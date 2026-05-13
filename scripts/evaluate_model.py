#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tendon_surrogate.training import evaluate_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained tendon surrogate model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", default=None, help="Optional dataset override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for metrics summary",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_checkpoint(
        args.checkpoint,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        out_path=args.output,
    )
    print(summary)


if __name__ == "__main__":
    main()
