#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tendon_surrogate.config import load_config
from tendon_surrogate.training import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tendon surrogate model.")
    parser.add_argument(
        "--config",
        default="configs/train_baseline.yaml",
        help="YAML config path",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--out-dir", default=None, help="Optional artifact directory override")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.out_dir is not None:
        config.training.out_dir = args.out_dir
    if args.device is not None:
        config.training.device = args.device
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    summary = run_training(config)
    print("Training complete")
    print(f"Artifacts: {config.training.out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
