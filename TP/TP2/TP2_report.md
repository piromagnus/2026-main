# TP2 Report - Principal Components Regression in Genetics

## 1) Pedagogical Goal

This TP studies how to predict geographical coordinates (latitude, longitude) from high-dimensional genetic markers.

Main challenge:

- `N = 494` individuals
- `p = 5709` markers
- so `p >> N`, which makes naive multiple linear regression unstable

The key idea is to reduce dimensionality first with PCA, then perform linear regression on PCA scores (PCR).

---

## 2) Data and Problem Setup

The file `NAm2.txt` contains:

- metadata columns (individual, population, country, sex),
- target coordinates: `lat`, `long`,
- binary genetic markers from column 9 onward.

Modeling tasks:

1. Understand why direct normal equations fail in high dimension.
2. Apply PCA and interpret explained variance.
3. Build PCR models to predict `(lat, long)`.
4. Evaluate using realistic distance error in kilometers (haversine distance).
5. Use cross-validation to select the number of principal components.

---

## 3) Main Concepts to Teach

## 3.1 Linear regression in high dimension

For ordinary least squares:

`beta_hat = (X^T X)^(-1) X^T y`

This requires `X^T X` to be invertible. Here, `rank(X) <= N = 494 < p = 5709`, so `X^T X` is singular.

Important distinction:

- `numpy.linalg.solve` fails on singular systems.
- `numpy.linalg.lstsq` returns a least-squares (minimum-norm) solution using SVD and does not require invertibility.

## 3.2 PCA

PCA finds orthogonal directions maximizing variance:

- geometric interpretation: new orthogonal axes capturing dominant structure,
- statistical interpretation: eigendecomposition of covariance matrix.

Why useful here:

- compresses noisy/high-dimensional markers,
- reduces variance of regression estimates,
- enables stable modeling.

## 3.3 PCR

PCR pipeline:

1. center data and compute PCA,
2. keep first `k` components,
3. regress latitude/longitude on these `k` scores.

Choosing `k` is a bias-variance trade-off:

- too small: underfitting,
- too large: overfitting.

---

## 4) Key Results Obtained

From the executed notebook:

- `np.linalg.solve` on normal equations fails (`Singular matrix`).
- `rank(X) = 494`, confirming rank deficiency.
- First 2 PCs explain about:
  - raw markers: `3.57%`,
  - standardized markers: `3.39%`.
- Components needed for cumulative variance:
  - `80%`: 272 PCs
  - `90%`: 354 PCs
  - `95%`: 410 PCs

PCR with 250 PCs evaluated on the same data gives mean location error around `649 km` (optimistic because in-sample).

10-fold shuffled CV for model selection (`k = 2..440`, step 10):

- best around `k = 332`,
- mean CV train error: about `404 km`,
- mean CV test error: about `1176 km`.

The large train/test gap shows non-negligible overfitting/generalization difficulty even with PCR.

---

## 5) Common Issues and How to Solve Them

## Issue A - Singular matrix error

Symptom:

- error when using `np.linalg.solve(X.T @ X, X.T @ y)`.

Cause:

- `rank(X) < p`, so `X^T X` is not invertible.

Fix:

- use `np.linalg.lstsq` or `sklearn.linear_model.LinearRegression`.

## Issue B - Misleadingly optimistic map

Symptom:

- predicted coordinates look very close to true points.

Cause:

- evaluating on training data only.

Fix:

- use cross-validation and report test error in km.

## Issue C - Data leakage in PCR

Symptom:

- unrealistically good CV scores.

Cause:

- PCA fitted on full data before CV.

Fix:

- use a pipeline (`make_pipeline(PCA(...), LinearRegression())`) and cross-validate the full pipeline.

## Issue D - Wrong distance metric

Symptom:

- errors measured with Euclidean distance on degrees.

Cause:

- latitude/longitude are spherical coordinates.

Fix:

- use `haversine_distances` on radians and convert to km.

## Issue E - Biased fold construction

Symptom:

- unstable/unrepresentative CV depending on row order.

Cause:

- data rows are structured by geography/populations.

Fix:

- at minimum use shuffled `KFold`;
- for stricter evaluation on unseen populations, use grouped CV (`GroupKFold` by country/tribe).

---

## 6) How to Explain This TP to Students

Recommended teaching storyline:

1. Start with the real scientific question: infer geographic origin from markers.
2. Show why naive linear algebra fails when `p >> N`.
3. Introduce PCA as a compression/denoising step.
4. Build PCR and visualize maps.
5. Contrast in-sample vs cross-validated performance.
6. Emphasize methodology lessons: leakage control, metric choice, and split strategy.

Core takeaways:

- high-dimensional prediction needs regularization or dimensionality reduction,
- visualization alone is not enough; robust validation is mandatory,
- practical ML quality depends as much on evaluation protocol as on model choice.

---

## 7) Possible Extensions

- Compare PCR with Partial Least Squares (PLS).
- Evaluate per-country/per-tribe errors with confidence intervals.
- Try group-aware validation to estimate transfer to unseen populations.
- Explore sparse/regularized models (ridge, lasso, elastic net) on markers or PCA scores.
