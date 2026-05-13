from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch import nn

from .config import ExperimentConfig, config_to_dict, dump_config, experiment_config_from_dict
from .data import (
    NormalizationStats,
    compute_normalization_stats,
    load_sequence_records,
    make_dataloader,
    split_records,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def move_batch_to_device(
    batch: dict[str, Any], device: "torch.device"
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _metrics_template() -> dict[str, float]:
    return {"loss": 0.0, "mse": 0.0, "mae": 0.0, "n_elem": 0.0}


def get_output_mode(config: ExperimentConfig) -> str:
    mode = config.model.output_mode.lower()
    if mode not in {"normalized", "raw"}:
        raise ValueError(f"Unsupported model.output_mode: {config.model.output_mode}")
    return mode


def run_epoch(
    model: "nn.Module",
    loader,
    *,
    device: "torch.device",
    optimizer: "torch.optim.Optimizer | None",
    target_mean: float,
    target_std: float,
    output_mode: str,
    grad_clip: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.MSELoss()
    stats = _metrics_template()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        dynamic = batch["dynamic"]
        dynamic_raw = batch["dynamic_raw"]
        static = batch["static"]
        static_raw = batch.get("static_raw")
        target = batch["target"]
        target_raw = batch["target_raw"]

        target_for_loss = target if output_mode == "normalized" else target_raw

        with torch.set_grad_enabled(is_train):
            pred = model(dynamic, static, dynamic_raw, static_raw)
            loss = criterion(pred, target_for_loss)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        pred_raw = (
            pred * target_std + target_mean if output_mode == "normalized" else pred
        )
        diff = pred_raw - target_raw
        n_elem = float(diff.numel())
        stats["loss"] += float(loss.item()) * n_elem
        stats["mse"] += float((diff.pow(2)).sum().item())
        stats["mae"] += float(diff.abs().sum().item())
        stats["n_elem"] += n_elem

    denom = max(stats["n_elem"], 1.0)
    return {
        "loss": stats["loss"] / denom,
        "mse": stats["mse"] / denom,
        "mae": stats["mae"] / denom,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_data(
    config: ExperimentConfig,
) -> tuple[dict[str, list[Any]], NormalizationStats]:
    records = load_sequence_records(
        config.data.dataset_path,
        sim_id_col=config.data.sim_id_col,
        split_col=config.data.split_col,
        time_col=config.data.time_col,
        dynamic_features=config.data.dynamic_features,
        static_features=config.data.static_features,
        target_col=config.data.target,
    )
    splits = split_records(records)
    if "train" not in splits or "val" not in splits or "test" not in splits:
        raise RuntimeError("Expected train/val/test splits in the dataset")
    stats = compute_normalization_stats(
        splits["train"],
        dynamic_features=config.data.dynamic_features,
        static_features=config.data.static_features,
        target_col=config.data.target,
    )
    return splits, stats


def _make_loaders(
    config: ExperimentConfig,
    splits: dict[str, list[Any]],
    stats: NormalizationStats,
    *,
    shuffle_train: bool,
) -> dict[str, Any]:
    return {
        split: make_dataloader(
            splits[split],
            stats,
            batch_size=config.data.batch_size,
            shuffle=(shuffle_train and split == "train"),
            num_workers=config.data.num_workers,
        )
        for split in ["train", "val", "test"]
    }


def _evaluate_splits(
    model: "nn.Module",
    loaders: dict[str, Any],
    *,
    device: torch.device,
    stats: NormalizationStats,
    output_mode: str,
) -> dict[str, dict[str, float]]:
    return {
        split: run_epoch(
            model,
            loaders[split],
            device=device,
            optimizer=None,
            target_mean=stats.target_mean,
            target_std=stats.target_std,
            output_mode=output_mode,
        )
        for split in ["train", "val", "test"]
    }


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    out_dir = Path(config.training.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_mode = get_output_mode(config)
    splits, stats = _prepare_data(config)
    loaders = _make_loaders(config, splits, stats, shuffle_train=True)

    model = config.model.build(config.data).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    dump_config(config, out_dir / "config_used.yaml")
    save_json(out_dir / "normalization_stats.json", stats.to_dict())

    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device=device,
            optimizer=optimizer,
            target_mean=stats.target_mean,
            target_std=stats.target_std,
            output_mode=output_mode,
            grad_clip=config.training.grad_clip,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            device=device,
            optimizer=None,
            target_mean=stats.target_mean,
            target_std=stats.target_std,
            output_mode=output_mode,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "config": config_to_dict(config),
                    "stats": stats.to_dict(),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val,
                },
                out_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            break

    save_history(out_dir / "history.csv", history)
    checkpoint = torch.load(
        out_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])

    final_metrics = _evaluate_splits(
        model,
        loaders,
        device=device,
        stats=stats,
        output_mode=output_mode,
    )
    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "metrics": final_metrics,
    }
    save_json(out_dir / "metrics_summary.json", summary)
    return summary


def _config_from_checkpoint(checkpoint: dict[str, Any]) -> ExperimentConfig:
    return experiment_config_from_dict(checkpoint["config"])


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | None = None,
    batch_size: int | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _config_from_checkpoint(checkpoint)
    if dataset_path is not None:
        config.data.dataset_path = dataset_path
    if batch_size is not None:
        config.data.batch_size = batch_size

    output_mode = get_output_mode(config)
    splits, _ = _prepare_data(config)
    stats = NormalizationStats.from_dict(checkpoint["stats"])
    loaders = _make_loaders(config, splits, stats, shuffle_train=False)

    device = resolve_device(config.training.device)
    model = config.model.build(config.data).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = _evaluate_splits(
        model,
        loaders,
        device=device,
        stats=stats,
        output_mode=output_mode,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "metrics": metrics,
    }
    if out_path is not None:
        save_json(Path(out_path), summary)
    return summary
