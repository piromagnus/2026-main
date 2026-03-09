---
name: challenge1-model-report
description: Produces structured Challenge1 model reports from tuning artifacts and logs. Use when the user asks for model description, optimization steps, tuned hyperparameters, strengths/weaknesses, tested setups, improvement ideas, or next steps.
---

# Challenge1 Model Report

## Purpose
Generate a clear, repeatable report for Challenge1 model tuning and optimization progress.

## Inputs
- `Challenge1/solution/model.py`
- `Challenge1/solution/submission.py`
- `Challenge1/output_data/model_tuning_report.csv`
- `Challenge1/output_data/model_tuning_report.json`
- `Challenge1/output_data/model_tuning_report.md`
- Latest run logs from `submission.py`

## Required Sections
Always include all sections below:
1. Model description
2. Steps made to optimize it
3. Tuning hyperparameters
4. Usual use cases of the model
5. Good points
6. Bad points
7. What was tested
8. How to improve it
9. Next steps to keep improving

## Workflow
1. Ensure artifacts are current by running:
   - `uv run --python .venv/bin/python Challenge1/solution/submission.py Challenge1/input_data Challenge1/output_data`
2. Read `model_tuning_report.csv` and `model_tuning_report.json`.
3. Identify:
   - all candidate models evaluated,
   - selected best model,
   - ranking and metric values (`mae`, `rmse`, `custom_score`),
   - train/validation configuration (`validation_split`, `random_state`).
4. Build a concise report using the template below.
5. End with concrete next experiments (not generic advice).

## Report Template
Use this template:

```markdown
# Challenge1 Model Optimization Report

## 1) Model Description
- Main model:
- Pipeline summary:
- Data setup (features and target):

## 2) Optimization Steps Performed
- Step 1:
- Step 2:
- Step 3:

## 3) Hyperparameter Tuning
- Validation split:
- Random seed:
- Candidates evaluated:
- Selection metric:
- Best configuration:

## 4) Usual Use Cases
- Model A is usually used for:
- Model B is usually used for:

## 5) Good Points
- Point 1:
- Point 2:

## 6) Bad Points
- Limitation 1:
- Limitation 2:

## 7) What Was Tested
- Tested candidate list:
- Validation metrics observed:
- Output contract checks:

## 8) How to Improve
- Improvement idea 1:
- Improvement idea 2:

## 9) Next Steps
1. Next experiment:
2. Next experiment:
3. Next experiment:
```

## Quality Rules
- Include numeric evidence from tuning outputs.
- Explicitly name both the best model and rejected alternatives.
- Distinguish completed tests vs proposed future tests.
- Keep recommendations actionable (specific parameter ranges or model families).
