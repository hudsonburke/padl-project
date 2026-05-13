from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .config import ExperimentConfig, experiment_config_from_dict
from .data import NormalizationStats, SequenceRecord, load_sequence_records, split_records
from .training import get_output_mode, resolve_device


def checkpoint_to_config(raw: dict[str, Any]) -> ExperimentConfig:
    return experiment_config_from_dict(raw)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - true
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_err))
    max_ae = float(np.max(abs_err))
    peak_true = float(np.max(true))
    peak_pred = float(np.max(pred))
    final_true = float(true[-1])
    final_pred = float(pred[-1])
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_ae": max_ae,
        "peak_true": peak_true,
        "peak_pred": peak_pred,
        "peak_abs_err": float(abs(peak_pred - peak_true)),
        "final_true": final_true,
        "final_pred": final_pred,
        "final_abs_err": float(abs(final_pred - final_true)),
    }


def predict_record(
    model: torch.nn.Module,
    record: SequenceRecord,
    stats: NormalizationStats,
    *,
    device: torch.device,
    output_mode: str,
) -> np.ndarray:
    dynamic_mean = np.asarray(stats.dynamic_mean, dtype=np.float32)
    dynamic_std = np.asarray(stats.dynamic_std, dtype=np.float32)
    static_mean = np.asarray(stats.static_mean, dtype=np.float32)
    static_std = np.asarray(stats.static_std, dtype=np.float32)

    dynamic = (record.dynamic - dynamic_mean) / dynamic_std
    static = (record.static - static_mean) / static_std

    dynamic_t = torch.from_numpy(dynamic.astype(np.float32)).unsqueeze(0).to(device)
    dynamic_raw_t = torch.from_numpy(record.dynamic.astype(np.float32)).unsqueeze(0).to(device)
    static_t = torch.from_numpy(static.astype(np.float32)).unsqueeze(0).to(device)
    static_raw_t = torch.from_numpy(record.static.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_out = model(dynamic_t, static_t, dynamic_raw_t, static_raw_t).squeeze(0).cpu().numpy()
    pred = pred_out * float(stats.target_std) + float(stats.target_mean) if output_mode == "normalized" else pred_out
    return pred.astype(np.float32)


def representative_indices(n_items: int, n_select: int) -> list[int]:
    if n_items <= 0 or n_select <= 0:
        return []
    if n_select >= n_items:
        return list(range(n_items))
    raw = np.linspace(0, n_items - 1, n_select)
    indices: list[int] = []
    seen = set()
    for value in raw:
        idx = int(round(float(value)))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    for idx in range(n_items):
        if len(indices) >= n_select:
            break
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return sorted(indices)


def make_plot(
    record: SequenceRecord,
    pred: np.ndarray,
    metrics: dict[str, float],
    *,
    static_feature_names: list[str],
    path: Path,
    label: str,
) -> None:
    true = record.target
    abs_err = np.abs(pred - true)
    static_desc = ", ".join(
        f"{name}={value:.3g}" for name, value in zip(static_feature_names, record.static, strict=False)
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1.4])
    axes[0].plot(record.time, true, label="true force", linewidth=2)
    axes[0].plot(record.time, pred, label="pred force", linewidth=2, linestyle="--")
    axes[0].set_ylabel("Force")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_title(
        f"{record.split.upper()} {label}: {record.sim_id}\n"
        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, max|e|={metrics['max_ae']:.4f}"
    )

    axes[1].plot(record.time, abs_err, color="tab:red", linewidth=1.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("|error|")
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.01, 0.95, static_desc, transform=axes[1].transAxes, va="top", ha="left", fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_prediction_diagnostics(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | None = None,
    out_dir: str | Path | None = None,
    splits: Sequence[str] = ("val", "test"),
    plots_per_split: int = 3,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint_to_config(checkpoint["config"])
    if dataset_path is not None:
        config.data.dataset_path = dataset_path

    out_dir = Path(out_dir) if out_dir else checkpoint_path.parent / "diagnostics"
    stats = NormalizationStats.from_dict(checkpoint["stats"])
    output_mode = get_output_mode(config)
    device = resolve_device(config.training.device)

    records = load_sequence_records(
        config.data.dataset_path,
        sim_id_col=config.data.sim_id_col,
        split_col=config.data.split_col,
        time_col=config.data.time_col,
        dynamic_features=config.data.dynamic_features,
        static_features=config.data.static_features,
        target_col=config.data.target,
    )
    split_map = split_records(records)

    model = config.model.build(config.data).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "dataset": config.data.dataset_path,
        "device": str(device),
        "splits": {},
    }

    selected_splits = [split for split in splits if split in split_map]
    if not selected_splits:
        raise RuntimeError("None of the requested splits were found in the dataset")

    for split in selected_splits:
        split_records_list = split_map[split]
        split_metrics: list[dict[str, Any]] = []

        for record in split_records_list:
            pred = predict_record(model, record, stats, device=device, output_mode=output_mode)
            metrics = compute_metrics(record.target, pred)
            metric_row: dict[str, Any] = {
                "sim_id": record.sim_id,
                "split": record.split,
                **{name: float(value) for name, value in zip(config.data.static_features, record.static, strict=False)},
                **metrics,
            }
            split_metrics.append(metric_row)
            metric_rows.append(metric_row)

            for i, t in enumerate(record.time):
                prediction_rows.append(
                    {
                        "sim_id": record.sim_id,
                        "split": record.split,
                        "t_idx": i,
                        "time": float(t),
                        "force_true": float(record.target[i]),
                        "force_pred": float(pred[i]),
                        "force_abs_err": float(abs(pred[i] - record.target[i])),
                    }
                )

        split_metrics_sorted = sorted(split_metrics, key=lambda row: row["mae"])
        mae_values = [row["mae"] for row in split_metrics_sorted]
        rmse_values = [row["rmse"] for row in split_metrics_sorted]
        summary["splits"][split] = {
            "n_sims": len(split_metrics_sorted),
            "mean_mae": float(np.mean(mae_values)),
            "median_mae": float(np.median(mae_values)),
            "mean_rmse": float(np.mean(rmse_values)),
            "best_sim": split_metrics_sorted[0]["sim_id"],
            "worst_sim": split_metrics_sorted[-1]["sim_id"],
        }

        chosen_indices = representative_indices(len(split_metrics_sorted), plots_per_split)
        sim_lookup = {record.sim_id: record for record in split_records_list}
        label_lookup: dict[int, str] = {}
        if len(chosen_indices) == 1:
            label_lookup[chosen_indices[0]] = "representative"
        elif len(chosen_indices) == 2:
            label_lookup[chosen_indices[0]] = "best"
            label_lookup[chosen_indices[1]] = "worst"
        elif len(chosen_indices) >= 3:
            label_lookup[chosen_indices[0]] = "best"
            label_lookup[chosen_indices[-1]] = "worst"
            mid = chosen_indices[len(chosen_indices) // 2]
            label_lookup[mid] = "median"

        for idx in chosen_indices:
            metric_row = split_metrics_sorted[idx]
            record = sim_lookup[metric_row["sim_id"]]
            pred = predict_record(model, record, stats, device=device, output_mode=output_mode)
            label = label_lookup.get(idx, f"rank_{idx+1}")
            plot_path = out_dir / "plots" / split / f"{label}_{record.sim_id}.png"
            make_plot(
                record,
                pred,
                metric_row,
                static_feature_names=config.data.static_features,
                path=plot_path,
                label=label,
            )

    save_json(out_dir / "prediction_summary.json", summary)
    write_csv(out_dir / "per_sim_metrics.csv", metric_rows)
    write_csv(out_dir / "predictions.csv", prediction_rows)
    write_parquet(out_dir / "predictions.parquet", prediction_rows)
    return summary
