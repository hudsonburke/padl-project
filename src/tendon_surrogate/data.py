from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SequenceRecord:
    sim_id: str
    split: str
    time: np.ndarray
    dynamic: np.ndarray
    static: np.ndarray
    target: np.ndarray


@dataclass
class NormalizationStats:
    dynamic_features: list[str]
    static_features: list[str]
    target: str
    dynamic_mean: list[float]
    dynamic_std: list[float]
    static_mean: list[float]
    static_std: list[float]
    target_mean: float
    target_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NormalizationStats":
        return cls(**raw)


class TendonSequenceDataset(Dataset):
    def __init__(
        self, records: list[SequenceRecord], stats: NormalizationStats
    ) -> None:
        self.records = records
        self.stats = stats
        self.dynamic_mean = np.asarray(stats.dynamic_mean, dtype=np.float32)
        self.dynamic_std = np.asarray(stats.dynamic_std, dtype=np.float32)
        self.static_mean = np.asarray(stats.static_mean, dtype=np.float32)
        self.static_std = np.asarray(stats.static_std, dtype=np.float32)
        self.target_mean = float(stats.target_mean)
        self.target_std = float(stats.target_std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        dynamic = (rec.dynamic - self.dynamic_mean) / self.dynamic_std
        static = (rec.static - self.static_mean) / self.static_std
        target = (rec.target - self.target_mean) / self.target_std
        return {
            "sim_id": rec.sim_id,
            "split": rec.split,
            "time": torch.from_numpy(rec.time.astype(np.float32)),
            "dynamic": torch.from_numpy(dynamic.astype(np.float32)),
            "dynamic_raw": torch.from_numpy(rec.dynamic.astype(np.float32)),
            "static": torch.from_numpy(static.astype(np.float32)),
            "static_raw": torch.from_numpy(rec.static.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
            "target_raw": torch.from_numpy(rec.target.astype(np.float32)),
        }


def _safe_std(
    values: np.ndarray, axis: int | tuple[int, ...] | None = None
) -> np.ndarray:
    std = values.std(axis=axis)
    return np.where(std < 1e-8, 1.0, std)


def load_sequence_records(
    path: str | Path,
    *,
    sim_id_col: str,
    split_col: str,
    time_col: str,
    dynamic_features: list[str],
    static_features: list[str],
    target_col: str,
) -> list[SequenceRecord]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sim_id = str(row[sim_id_col])
        grouped.setdefault(sim_id, []).append(row)

    records: list[SequenceRecord] = []
    for sim_id in sorted(grouped):
        sim_rows = sorted(grouped[sim_id], key=lambda r: float(r[time_col]))
        split = str(sim_rows[0][split_col])
        time = np.asarray([float(r[time_col]) for r in sim_rows], dtype=np.float32)
        dynamic = np.asarray(
            [[float(r[name]) for name in dynamic_features] for r in sim_rows],
            dtype=np.float32,
        )
        static = np.asarray(
            [float(sim_rows[0][name]) for name in static_features], dtype=np.float32
        )
        target = np.asarray([float(r[target_col]) for r in sim_rows], dtype=np.float32)
        records.append(
            SequenceRecord(
                sim_id=sim_id,
                split=split,
                time=time,
                dynamic=dynamic,
                static=static,
                target=target,
            )
        )
    return records


def split_records(records: list[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    splits: dict[str, list[SequenceRecord]] = {}
    for record in records:
        splits.setdefault(record.split, []).append(record)
    return splits


def compute_normalization_stats(
    train_records: list[SequenceRecord],
    *,
    dynamic_features: list[str],
    static_features: list[str],
    target_col: str,
) -> NormalizationStats:
    if not train_records:
        raise RuntimeError(
            "No training records available to compute normalization stats"
        )
    dynamic_stack = np.concatenate([r.dynamic for r in train_records], axis=0)
    static_stack = np.stack([r.static for r in train_records], axis=0)
    target_stack = np.concatenate([r.target for r in train_records], axis=0)
    return NormalizationStats(
        dynamic_features=dynamic_features,
        static_features=static_features,
        target=target_col,
        dynamic_mean=dynamic_stack.mean(axis=0).astype(float).tolist(),
        dynamic_std=_safe_std(dynamic_stack, axis=0).astype(float).tolist(),
        static_mean=static_stack.mean(axis=0).astype(float).tolist(),
        static_std=_safe_std(static_stack, axis=0).astype(float).tolist(),
        target_mean=float(target_stack.mean()),
        target_std=float(_safe_std(target_stack)),
    )


def make_dataloader(
    records: list[SequenceRecord],
    stats: NormalizationStats,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TendonSequenceDataset(records, stats)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
