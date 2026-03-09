# Challenge1 Agent Guide

## Scope

- Applies to work inside `Challenge1/`.
- Main files to edit are `Challenge1/solution/model.py` and `Challenge1/solution/submission.py`.

## Goal

- Train a regression model that predicts `age` from methylation features and `gender`.
- Produce a valid submission file at `y_pred.csv`.

## Environment Rules

- Use `uv` as the package manager.
- Use the project virtual environment at `.venv` (repo root).
- Keep runtime dependencies in `pyproject.toml` and lock with `uv.lock`.

## Standard Commands (run from repo root)

- Sync environment:
  - `uv sync`
- Run submission pipeline end to end:
  - `uv run --python .venv/bin/python Challenge1/solution/submission.py Challenge1/input_data Challenge1/output_data`
- Run fixed final-submission script (ElasticNet a01-l03 on full train data):
  - `uv run --python .venv/bin/python Challenge1/solution/submission_elasticnet_a01_l03.py Challenge1/input_data Challenge1/output_submission_elasticnet_a01_l03`
- Run a specific optimization round (`1..6`) and keep reports separated:
  - `uv run --python .venv/bin/python Challenge1/solution/submission.py Challenge1/input_data Challenge1/output_data/round3 --validation-split 0.1 --max-round 3 --selection-metric rmse`
- Validate output contract:
  - `uv run --python .venv/bin/python - <<'PY'\nimport pandas as pd\np = pd.read_csv('Challenge1/output_data/y_pred.csv')\nassert list(p.columns) == ['age']\nassert len(p) == 200\nassert not p['age'].isna().any()\nprint('ok', p.shape)\nPY`

## Modeling Guidance

- This is a high-dimensional regression problem (`p >> n`), so prefer regularized models.
- Keep preprocessing explicit:
  - numeric imputation/scaling,
  - categorical handling for `gender`,
  - deterministic settings (`random_state=42`) when available.
- Keep `Model` API stable:
  - `fit(X, y)`
  - `predict(X)`

## Optimization Rounds (Model Search Space)

- Round 1: Linear tuning
  - Ridge / Lasso / ElasticNet variants.
- Round 2: Feature pre-selection
  - VarianceThreshold + ElasticNet, PCA + Ridge/ElasticNet.
- Round 3: Supervised dimensionality reduction
  - PLSRegression (`n_components` sweep).
- Round 4: Nonlinear sklearn models
  - SVR, RandomForestRegressor, GradientBoostingRegressor.
- Round 5: External boosting models
  - XGBoost and LightGBM.
- Round 6: Stacking
  - `StackingRegressor` built from top-3 models selected by validation metric.

## Tuning and Selection Rules

- Primary selection metric for optimization rounds is `rmse`.
- Secondary metrics tracked are `mae` and `custom_score`.
- Use `validation_split=0.1` for experiments.
- Set `validation_split=0.0` before final submission to train on full data.

## Submission Contract

- `submission.py` must accept exactly two positional arguments:
  - `input_dir`
  - `output_dir`
- Predictions must be written to:
  - `<output_dir>/y_pred.csv`
- Output format must be:
  - one column named `age`,
  - one row per sample in `X_test.csv`.

## Tracking Artifacts

- Every run writes:
  - `model_tuning_report.csv`
  - `model_tuning_report.json`
  - `model_tuning_report.md`
  - `y_pred.csv`
- When iterating rounds, write each run to its own output folder (for example `Challenge1/output_data/round5/`).

## Safety / Non-goals

- Do not modify files inside `Challenge1/input_data/`.
- Do not hardcode train/test row counts in model logic.
- Keep changes minimal and focused on predictive performance + submission correctness.

