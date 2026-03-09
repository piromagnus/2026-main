---
name: challenge1-epiclock-workflow
description: Improves the Challenge1 epigenetic clock regression solution with reproducible local testing. Use when working on Challenge1, epiclock, model.py, submission.py, local benchmark comparisons, or preparing y_pred.csv outputs for submission.
---

# Challenge1 Epiclock Workflow

## Purpose
Use this skill to improve and validate the Challenge1 solution while preserving submission compatibility.

## Required Inputs
- `Challenge1/Info.md`
- `Challenge1/solution/model.py`
- `Challenge1/solution/submission.py`
- `Challenge1/input_data/` CSV files

## Execution Rules
1. Read the nearest `AGENTS.md` before editing files.
2. Use `uv` as the package manager.
3. Run Python through the project virtual environment (`.venv`), preferably via:
   - `uv run --python .venv/bin/python ...`
4. Keep changes focused on model quality and submission correctness.

## Workflow

### 1) Environment sync
Run from repository root:

```bash
uv sync
```

### 2) Improve `model.py`
Checklist:
- Keep `Model.fit(X, y)` and `Model.predict(X)` signatures.
- Use a regularized regression approach suitable for `p >> n`.
- Include preprocessing for:
  - numeric methylation features (impute + scale),
  - categorical `gender` (impute + encode).
- Use deterministic settings where available (`random_state=42`).

### 3) Harden `submission.py`
Checklist:
- Preserve CLI arguments: `input_dir output_dir`.
- Read `X_train.csv`, `y_train.csv`, `X_test.csv` from `input_dir`.
- Write predictions to `<output_dir>/y_pred.csv`.
- Enforce output schema:
  - one column named `age`,
  - one row per test sample.

### 4) Validate locally
Run end-to-end submission generation:

```bash
uv run --python .venv/bin/python Challenge1/solution/submission.py Challenge1/input_data Challenge1/output_data
```

Validate output contract:

```bash
uv run --python .venv/bin/python - <<'PY'
import pandas as pd
p = pd.read_csv("Challenge1/output_data/y_pred.csv")
assert list(p.columns) == ["age"]
assert len(p) == 200
assert not p["age"].isna().any()
print("ok", p.shape)
PY
```

### 5) Compare against baseline
Use 5-fold CV MAE to compare:
- baseline: mean predictor,
- improved model: current `Model` implementation.

Report:
- fold-level MAE values,
- mean and standard deviation for both models.

## Output Format
When finishing, report:
1. Files changed.
2. CV benchmark results (baseline vs improved).
3. Submission file validation results.
4. Any dependency updates (`pyproject.toml` / `uv.lock`).
