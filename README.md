# PADL Project Submission

This repository contains the final project code, report, processed datasets, and final trained artifacts for the tendon force surrogate study.

## Final models

The two report models are:

- `baseline`: GRU baseline model
- `maxwell`: physics-aware explicit Maxwell model with tendon stiffness scaling

## Repository layout

- `report.qmd` — source for the final report
- `report.pdf` — rendered report PDF
- `_freeze/` — committed Quarto frozen execution outputs for reproducible report rendering
- `configs/` — final training configs
- `src/tendon_surrogate/` — package code
- `scripts/` — thin CLI wrappers
- `artifacts/` — final trained model outputs and diagnostics
- `data/processed/full_sweep_v2/` — processed datasets used by the report

## Environment setup

This project uses `uv`.

From the repository root:

```bash
uv sync
```

## Rendering the report

This repo uses Quarto project-level freeze (`freeze: auto`).
The committed `_freeze/` directory allows a fresh clone to render the report without retraining the models.
Normal renders should reuse the frozen execution outputs.

### Normal render using frozen results

```bash
uv run quarto render --to pdf
```

or

```bash
uv run quarto render --to html
```

## Forcing a full rerun

The report contains hidden Python cells that can rerun the final training/evaluation/diagnostic pipelines.
Because the project uses `freeze: auto`, normal renders should reuse frozen results. In addition, the report now reuses existing model artifacts unless you explicitly request retraining.
To force a full recompute and refresh cached/frozen outputs, target `report.qmd` directly:

```bash
PADL_FORCE_RETRAIN=1 uv run quarto render report.qmd --execute --to pdf
```

or

```bash
PADL_FORCE_RETRAIN=1 uv run quarto render report.qmd --execute --to html
```

This will rerun the four final experiments:

- baseline interpolation
- baseline stiffness holdout
- maxwell interpolation
- maxwell stiffness holdout

and regenerate:

- `metrics_summary.json`
- `eval_metrics.json`
- diagnostics plots and CSV/Parquet outputs
- updated report figures/tables

## Training/evaluation scripts

The main CLI entry points are:

### Train

```bash
uv run python scripts/train_model.py --config configs/train_baseline.yaml
uv run python scripts/train_model.py --config configs/train_baseline_stiffness_holdout.yaml
uv run python scripts/train_model.py --config configs/train_maxwell.yaml
uv run python scripts/train_model.py --config configs/train_maxwell_stiffness_holdout.yaml
```

### Evaluate

```bash
uv run python scripts/evaluate_model.py --checkpoint artifacts/baseline/best_model.pt
uv run python scripts/evaluate_model.py --checkpoint artifacts/maxwell/best_model.pt
```

### Regenerate diagnostics

```bash
uv run python scripts/plot_predictions.py --checkpoint artifacts/baseline/best_model.pt
uv run python scripts/plot_predictions.py --checkpoint artifacts/maxwell/best_model.pt
```

## Notes

- Normal report renders should use the committed `_freeze/` outputs and should not retrain.
- Even if the document executes, existing experiment artifacts are reused by default.
- Use `PADL_FORCE_RETRAIN=1 uv run quarto render report.qmd --execute  ...` when you explicitly want to rerun training/evaluation/diagnostics.
- `.venv/`, `.jupyter_cache/`, `.quarto/`, and other local/generated files are gitignored.
- The final report compares the GRU baseline against the explicit Maxwell model.
