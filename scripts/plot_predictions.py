#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tendon_surrogate.diagnostics import generate_prediction_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate prediction diagnostics for a trained tendon surrogate model.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", default=None, help="Optional dataset override")
    parser.add_argument("--out-dir", default=None, help="Optional diagnostics output directory")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["val", "test"],
        help="Dataset splits to include in diagnostics",
    )
    parser.add_argument(
        "--plots-per-split",
        type=int,
        default=3,
        help="How many representative trajectory plots to save per split",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = generate_prediction_diagnostics(
        args.checkpoint,
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        splits=args.splits,
        plots_per_split=args.plots_per_split,
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "diagnostics"
    print(f"Diagnostics written to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
