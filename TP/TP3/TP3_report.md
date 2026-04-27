# TP3 Report - Benchmarking Classification Methods

## 1) Pedagogical Goal

This TP compares classification methods in two settings:

- a controlled simulated binary classification problem, where the true data-generating distribution is known;
- a real tabular classification problem, the Kaggle Titanic survival dataset, where preprocessing, validation, and submission protocol matter.

The main lesson is that classifier performance depends on both the model assumptions and the evaluation protocol. A classifier can be close to optimal when its assumptions match the data, but fail badly when the test distribution changes.

---

## 2) Part 1 - Simulated Gaussian Data

Each observation has a label `Y in {0, 1}` with

`P(Y = 1) = p`, `P(Y = 0) = 1 - p`.

Conditional on the class:

`X | Y = 0 ~ N(mu_0, Sigma_0)`, with `mu_0 = (0, 0)` and `Sigma_0 = 0.5 I`;

`X | Y = 1 ~ N(mu_1, Sigma_1)`, with `mu_1 = (epsilon, 0)` and `Sigma_1 = 0.4 I`.

For the main dataset:

- `D_train = D(50 | epsilon = 2, p = 0.30)`;
- `D_test = D(1000 | epsilon = 2, p = 0.30)`.

The numerical values reported below were computed from one reproducible simulation using seed `20260427`. They should be treated as representative values, not exact constants.

### 2.1 Question (a) - Plotting the data

The plot should show both train and test points:

- color = class label;
- marker shape = train or test split.

This plot is useful because students should see the two Gaussian clouds before fitting models. The class with label 1 is centered around `(2, 0)` and is slightly more concentrated because its covariance is `0.4 I` instead of `0.5 I`.

Common mistake: plotting only the training set. The question asks for `D_train union D_test`.

---

## 3) Question (b) - Bayes Classifier

The Bayes classifier predicts the class with the largest posterior probability. Here:

`h*(x) = 1` if

`P(Y = 1 | X = x) / P(Y = 0 | X = x) > 1`.

Using Bayes' formula:

`P(Y = 1 | X = x) / P(Y = 0 | X = x) = p f_1(x) / ((1 - p) f_0(x))`,

where `f_0` and `f_1` are the Gaussian densities of the two classes.

So the Bayes classifier is:

`h*(x) = 1{ log(p / (1 - p)) + log f_1(x) - log f_0(x) > 0 }`.

With the covariance matrices from the TP, this simplifies to:

`g(x_1, x_2) = log(p / (1 - p)) + log(1.25) - 0.25(x_1^2 + x_2^2) + 2.5 epsilon x_1 - 1.25 epsilon^2`.

The Bayes classifier is:

`h*(x) = 1{ g(x_1, x_2) > 0 }`.

The decision boundary is:

`g(x_1, x_2) = 0`.

Equivalently:

`(x_1 - 5 epsilon)^2 + x_2^2 = 20 epsilon^2 + 4 log(1.25 p / (1 - p))`.

For `epsilon = 2` and `p = 0.3`, the boundary is approximately:

`(x_1 - 10)^2 + x_2^2 = 77.50`.

This is a quadratic boundary, not a straight line.

---

## 4) Question (c) - Bayes Error and Effect of epsilon

A scikit-learn style implementation should inherit from `BaseEstimator` and `ClassifierMixin`, implement `fit`, and implement `predict` using the log-posterior ratio above.

Using a simulation with `10^4` points from `D(10^4 | 2, 0.3)`, the Bayes error is about:

`0.060`.

Representative empirical Bayes error rates for different values of `epsilon`:

| epsilon | Bayes error |
|---:|---:|
| 0.00 | 0.302 |
| 0.25 | 0.300 |
| 0.50 | 0.291 |
| 0.75 | 0.246 |
| 1.00 | 0.198 |
| 1.25 | 0.155 |
| 1.50 | 0.121 |
| 1.75 | 0.088 |
| 2.00 | 0.061 |
| 2.50 | 0.029 |
| 3.00 | 0.012 |
| 4.00 | 0.001 |

Interpretation:

- when `epsilon` is small, the two class distributions overlap strongly;
- when `epsilon` increases, the class means move apart and the Bayes error decreases;
- at `epsilon = 0`, the classes have the same mean, and the best rule mostly exploits the class prior and the small covariance difference.

---

## 5) Question (d) - Most Adequate Classifier

QDA is the most adequate classifier for this simulated setting.

Reason:

- LDA assumes Gaussian class-conditional distributions with equal covariance matrices;
- QDA assumes Gaussian class-conditional distributions with class-specific covariance matrices;
- logistic regression learns a linear decision boundary directly.

Here the data are generated from Gaussian distributions, but `Sigma_0 != Sigma_1`. Therefore the exact Bayes boundary is quadratic, so QDA is the correctly specified model.

LDA and logistic regression can still work well because `0.5 I` and `0.4 I` are close, so the boundary is not too far from linear in the region where most points lie.

---

## 6) Question (e) - LDA, QDA, Logistic Regression

Using one fixed simulation with `D_train = D(50 | 2, 0.3)` and `D_test = D(1000 | 2, 0.3)`, the observed test errors were:

| Classifier | Test error |
|---|---:|
| Bayes classifier | 0.068 |
| LDA | 0.072 |
| QDA | 0.067 |
| Logistic regression | 0.077 |

These numbers vary from one random sample to another, but the qualitative conclusion is stable:

- QDA is closest to the Bayes classifier because it matches the generative assumptions;
- LDA and logistic regression are close because the two covariance matrices are similar;
- with only 50 training points, QDA may sometimes be unstable because it estimates more parameters than LDA.

Empirical effect of increasing training size, averaged over repeated simulations:

| Training size | LDA | QDA | Logistic regression |
|---:|---:|---:|---:|
| 20 | 0.082 | 0.116 | 0.102 |
| 50 | 0.067 | 0.072 | 0.071 |
| 100 | 0.065 | 0.066 | 0.066 |
| 200 | 0.063 | 0.064 | 0.064 |
| 500 | 0.062 | 0.062 | 0.062 |
| 1000 | 0.062 | 0.061 | 0.062 |
| 5000 | 0.061 | 0.061 | 0.061 |

Interpretation:

- as training size increases, estimation variance decreases;
- QDA converges to the Bayes classifier because it is correctly specified;
- LDA and logistic regression converge to their best linear approximations, so any remaining gap is due to model misspecification;
- in this specific TP, that gap is small because the two covariance matrices are close.

---

## 7) Question (f) - Distribution Shift

Now evaluate the same classifiers trained on `D_train = D(50 | 2, 0.3)` on:

`D'_test = D(1000 | 0.5, 0.7)`.

Observed errors:

| Classifier trained on old distribution | Error on shifted test set |
|---|---:|
| Bayes classifier for old distribution | 0.602 |
| LDA | 0.592 |
| QDA | 0.578 |
| Logistic regression | 0.604 |

For comparison, the oracle Bayes classifier using the correct shifted parameters `epsilon = 0.5`, `p = 0.7` has error about:

`0.263`.

The problem is not that LDA, QDA, or logistic regression are implemented incorrectly. The problem is distribution shift:

- the class prior changes from `p = 0.3` to `p = 0.7`;
- the class separation changes from `epsilon = 2` to `epsilon = 0.5`;
- the training and test data are no longer identically distributed.

A model trained on the old distribution learns the wrong posterior probabilities for the new distribution.

How to solve it:

- retrain on data from the new distribution if labels are available;
- if the new class prior is known, recalibrate or adjust the decision threshold;
- if the feature distribution changes, use domain adaptation or collect representative training data;
- always check whether the train/test split is IID before trusting test errors.

---

## 8) Part 2 - Titanic Dataset

The repository does not contain the Kaggle Titanic CSV files, and a Kaggle leaderboard score cannot be verified locally without downloading the data and submitting predictions. The report students submit should include their actual Kaggle public score.

The expected workflow is still clear.

### 8.1 Question (a) - Feature Engineering

Recommended feature engineering:

- keep `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`;
- create `FamilySize = SibSp + Parch + 1`;
- create `IsAlone = 1{FamilySize = 1}`;
- extract `Title` from `Name`, for example `Mr`, `Mrs`, `Miss`, `Master`, and group rare titles;
- extract `Deck` from the first letter of `Cabin`, using `Unknown` for missing cabins;
- optionally create `FarePerPerson = Fare / FamilySize`;
- keep `PassengerId` only for the final submission file, not as a predictor.

Missing values:

- impute `Age` and `Fare` with a median;
- impute `Embarked`, `Cabin`-derived variables, and other categorical variables with a frequent or explicit missing category.

Encoding:

- numerical variables can be passed as numbers, with scaling if using logistic regression or distance-based methods;
- categorical variables should be one-hot encoded with `handle_unknown="ignore"`;
- if using `skrub`, explain that the tool automatically detects column types and applies suitable tabular encodings, but students should still inspect the produced pipeline and validate it.

Common mistake: fitting preprocessing on the full dataset before cross-validation. Imputation and encoding must be inside a `Pipeline` or `ColumnTransformer`, so each validation fold is processed without information leakage.

### 8.2 Question (b) - Classifier Choice

Good baseline classifiers:

- logistic regression with a preprocessing pipeline;
- random forest or gradient boosting;
- `HistGradientBoostingClassifier`;
- `skrub.tabular_learner` for a strong automatic tabular baseline.

A reasonable choice for students is gradient boosting or `skrub.tabular_learner`, because Titanic is a small mixed-type tabular dataset with nonlinear interactions such as sex, passenger class, age, and family size.

Validation should use stratified cross-validation, for example `StratifiedKFold`, because the survival labels are imbalanced. If students use family- or ticket-derived features, they should discuss possible dependence between passengers from the same family or ticket group.

The final Kaggle submission must contain exactly:

- `PassengerId`;
- `Survived`.

The public leaderboard score must be copied from Kaggle after submission. A simple well-built pipeline usually lands around the high `0.7x` accuracy range, but the exact value depends on the feature engineering, model, random seed, and Kaggle submission.

---

## 9) Common Student Issues

### Issue A - Forgetting the class prior in the Bayes classifier

Symptom:

- students use only `f_1(x) / f_0(x)` and ignore `p / (1 - p)`.

Why it is wrong:

- the prior is part of the posterior odds;
- here `p = 0.3`, so class 1 is less frequent.

Fix:

- use `p f_1(x) / ((1 - p) f_0(x))`.

### Issue B - Assuming the Bayes boundary is linear

Symptom:

- students draw or derive a straight-line boundary.

Why it is wrong:

- equal covariance Gaussians give a linear boundary;
- unequal covariance Gaussians give a quadratic boundary.

Fix:

- check whether `Sigma_0 = Sigma_1`;
- here they are different, so the correct boundary is quadratic.

### Issue C - Comparing classifiers on different test sets

Symptom:

- each model is evaluated on a newly generated random test set.

Why it is wrong:

- differences may come from test-set randomness, not classifier performance.

Fix:

- generate one test set and evaluate all classifiers on that same set.

### Issue D - Interpreting one simulation as a universal result

Symptom:

- students conclude that one classifier is always better from one run.

Why it is wrong:

- `D_train` has only 50 points, so sampling variability is large.

Fix:

- repeat simulations or vary training size;
- report mean and variability.

### Issue E - Ignoring distribution shift

Symptom:

- students are surprised that all classifiers fail on `D'_test`.

Why it is wrong:

- `D'_test` is not drawn from the same distribution as the training data.

Fix:

- identify what changed: prior `p` and class separation `epsilon`;
- explain that train and test are not IID;
- retrain, recalibrate, or collect representative data.

### Issue F - Data leakage in Titanic preprocessing

Symptom:

- imputation or encoding is fitted before cross-validation.

Why it is wrong:

- validation folds influence preprocessing statistics.

Fix:

- put feature engineering, imputation, encoding, and classification into one scikit-learn pipeline.

---

## 10) How to Explain This TP to Students

Recommended storyline:

1. Start from the Bayes classifier: if the full distribution is known, the optimal decision rule can be written explicitly.
2. Show that model assumptions determine the decision boundary: equal covariance gives LDA, unequal covariance gives QDA.
3. Compare Bayes, LDA, QDA, and logistic regression on the same test data.
4. Increase the training size to separate estimation error from approximation error.
5. Use the shifted test set to demonstrate why IID train/test assumptions matter.
6. Move to Titanic and emphasize practical ML: preprocessing, leakage control, validation strategy, and reproducible submissions.

Core takeaways:

- Bayes error is the irreducible error for a given data distribution;
- QDA is appropriate when class-conditional Gaussians have unequal covariance matrices;
- more training data reduces estimation error, but cannot fully fix model misspecification;
- distribution shift can dominate classifier choice;
- real tabular ML quality depends heavily on preprocessing and validation.
