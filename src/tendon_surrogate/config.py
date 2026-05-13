from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import yaml

from .baseline_model import BaselineModelConfig
from .maxwell_model import MaxwellModelConfig

import torch.nn as nn


@dataclass
class DataConfig:
    dataset_path: str = "data/processed/full_sweep_v2/resampled_dataset.parquet"
    sim_id_col: str = "sim_id"
    split_col: str = "split"
    time_col: str = "time"
    dynamic_features: list[str] = field(
        default_factory=lambda: ["strain", "strain_rate", "phase"]
    )
    static_features: list[str] = field(
        default_factory=lambda: ["tendon_l1", "visco_g1", "visco_t1", "ramp_time"]
    )
    target: str = "force"
    batch_size: int = 16
    num_workers: int = 0


class BuildableModelConfig(Protocol):
    model_name: str
    output_mode: str

    def build(self, data_config: "DataConfig") -> "nn.Module": ...


ModelConfig = BaselineModelConfig | MaxwellModelConfig


_MODEL_NAME_ALIASES = {
    "baseline": "baseline",
    "maxwell": "maxwell",
    # Legacy aliases kept so old checkpoints/configs still load.
    "tendon_coupled_maxwell_parc_v1p2": "maxwell",
    "coupled_maxwell_parc_v1p2": "maxwell",
    "tendon_maxwell_parc_v1p2": "maxwell",
    "coupled_maxwell_parc_explicit_tendon_scale": "maxwell",
}

_MODEL_CONFIG_TYPES: dict[str, type[ModelConfig]] = {
    "baseline": BaselineModelConfig,
    "maxwell": MaxwellModelConfig,
}


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 25
    out_dir: str = "artifacts/baseline"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=BaselineModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _section(raw: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if raw is None:
        return {}
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping")
    return value


def _filter_dataclass_kwargs(cls: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    valid = {item.name for item in fields(cls)}
    return {key: value for key, value in raw.items() if key in valid}


def normalize_model_name(model_name: str) -> str:
    key = model_name.lower()
    if key not in _MODEL_NAME_ALIASES:
        supported = ", ".join(sorted(_MODEL_NAME_ALIASES))
        raise ValueError(
            f"Unknown model_name: {model_name}. Supported names: {supported}"
        )
    return _MODEL_NAME_ALIASES[key]


def get_model_config_type(model_name: str) -> type[ModelConfig]:
    return _MODEL_CONFIG_TYPES[normalize_model_name(model_name)]


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a mapping")

    data_section = _filter_dataclass_kwargs(DataConfig, _section(raw, "data"))
    model_section = _section(raw, "model")
    training_section = _filter_dataclass_kwargs(
        TrainingConfig, _section(raw, "training")
    )

    model_name = normalize_model_name(str(model_section.get("model_name", "baseline")))
    model_cls = get_model_config_type(model_name)
    model_section = _filter_dataclass_kwargs(
        model_cls, {**model_section, "model_name": model_name}
    )

    return ExperimentConfig(
        data=DataConfig(**data_section),
        model=model_cls(**model_section),
        training=TrainingConfig(**training_section),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return experiment_config_from_dict(raw)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(asdict(config), sort_keys=False))


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
