import logging
from copy import deepcopy

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        validation_split=0.0,
        random_state=42,
        max_round=6,
        selection_metric="rmse",
    ):
        self.gender_column = "gender"
        self.validation_split = float(validation_split)
        self.random_state = int(random_state)
        self.max_round = int(max_round)
        self.selection_metric = selection_metric
        self.model = None
        self.best_params_ = None
        self.validation_results_ = []
        self.candidate_params = self._build_candidate_params()
        self._active_candidate_lookup = {}

    def _build_candidate_params(self):
        return [
            {
                "round": 1,
                "name": "elasticnet-default",
                "regressor": "elasticnet",
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "description": "Linear model with mixed L1/L2 regularization.",
                "usual_use": "High-dimensional data where feature selection helps.",
                "good_points": "Can shrink coefficients and set some to zero.",
                "bad_points": "Needs alpha/l1_ratio tuning and can be slow.",
            },
            {
                "round": 1,
                "name": "ridge-alpha-1",
                "regressor": "ridge",
                "alpha": 1.0,
                "description": "Linear model with L2 regularization.",
                "usual_use": "Correlated features and p >> n settings.",
                "good_points": "Stable, fast, and handles multicollinearity well.",
                "bad_points": "Does not perform feature selection.",
            },
            {
                "round": 1,
                "name": "ridge-alpha-10",
                "regressor": "ridge",
                "alpha": 10.0,
                "description": "Linear model with stronger L2 regularization.",
                "usual_use": "Noisy or very high-dimensional regression.",
                "good_points": "Stronger shrinkage can improve generalization.",
                "bad_points": "Can underfit if regularization is too strong.",
            },
            {
                "round": 1,
                "name": "elasticnet-a005-l09",
                "regressor": "elasticnet",
                "alpha": 0.05,
                "l1_ratio": 0.9,
                "description": "ElasticNet with stronger L1 component.",
                "usual_use": "Sparse high-dimensional feature selection.",
                "good_points": "Can remove many weak features.",
                "bad_points": "Can be unstable if over-sparse.",
            },
            {
                "round": 1,
                "name": "elasticnet-a01-l03",
                "regressor": "elasticnet",
                "alpha": 0.01,
                "l1_ratio": 0.3,
                "description": "ElasticNet with lighter regularization.",
                "usual_use": "Signal-rich settings with many useful features.",
                "good_points": "Less shrinkage can improve fit.",
                "bad_points": "Higher overfitting risk.",
            },
            {
                "round": 1,
                "name": "elasticnet-a03-l07",
                "regressor": "elasticnet",
                "alpha": 0.03,
                "l1_ratio": 0.7,
                "description": "ElasticNet with balanced shrinkage and sparsity.",
                "usual_use": "Correlated sparse predictors.",
                "good_points": "Good bias/variance compromise.",
                "bad_points": "Still needs robust tuning.",
            },
            {
                "round": 1,
                "name": "lasso-a01",
                "regressor": "lasso",
                "alpha": 0.01,
                "description": "Pure L1 regularized linear model.",
                "usual_use": "Automatic feature selection.",
                "good_points": "Very interpretable sparse coefficients.",
                "bad_points": "Can underperform when features are correlated.",
            },
            {
                "round": 1,
                "name": "ridge-alpha-01",
                "regressor": "ridge",
                "alpha": 0.1,
                "description": "Ridge model with lighter L2 penalty.",
                "usual_use": "Low-bias regularized linear baseline.",
                "good_points": "Stable and fast.",
                "bad_points": "No feature pruning.",
            },
            {
                "round": 2,
                "name": "varthresh-elasticnet",
                "regressor": "varthresh_elasticnet",
                "variance_threshold": 0.01,
                "alpha": 0.05,
                "l1_ratio": 0.5,
                "description": "Variance filtering then ElasticNet.",
                "usual_use": "Remove low-information CpGs before fitting.",
                "good_points": "Can reduce noise and dimensionality.",
                "bad_points": "Can discard weak but useful features.",
            },
            {
                "round": 2,
                "name": "pca50-ridge",
                "regressor": "pca_ridge",
                "n_components": 50,
                "alpha": 1.0,
                "description": "PCA(50) followed by Ridge.",
                "usual_use": "Compact latent representation for p >> n.",
                "good_points": "Fast and robust.",
                "bad_points": "Unsupervised reduction may lose target signal.",
            },
            {
                "round": 2,
                "name": "pca100-ridge",
                "regressor": "pca_ridge",
                "n_components": 100,
                "alpha": 1.0,
                "description": "PCA(100) followed by Ridge.",
                "usual_use": "Preserve more variance before regression.",
                "good_points": "Can retain richer signal.",
                "bad_points": "Higher complexity and overfitting risk.",
            },
            {
                "round": 2,
                "name": "pca200-elasticnet",
                "regressor": "pca_elasticnet",
                "n_components": 200,
                "alpha": 0.05,
                "l1_ratio": 0.5,
                "description": "PCA(200) followed by ElasticNet.",
                "usual_use": "Dimensionality reduction with sparse linear fit.",
                "good_points": "Combines latent compression and regularization.",
                "bad_points": "More expensive than plain linear models.",
            },
            {
                "round": 3,
                "name": "pls-10",
                "regressor": "pls",
                "n_components": 10,
                "description": "PLS regression with 10 components.",
                "usual_use": "Supervised dimensionality reduction for p >> n.",
                "good_points": "Uses target information during projection.",
                "bad_points": "Can overfit if too many components.",
            },
            {
                "round": 3,
                "name": "pls-20",
                "regressor": "pls",
                "n_components": 20,
                "description": "PLS regression with 20 components.",
                "usual_use": "Supervised latent factors with richer representation.",
                "good_points": "Can capture additional predictive structure.",
                "bad_points": "Higher model complexity.",
            },
            {
                "round": 3,
                "name": "pls-50",
                "regressor": "pls",
                "n_components": 50,
                "description": "PLS regression with 50 components.",
                "usual_use": "Maximal supervised compression in current sweep.",
                "good_points": "Flexible latent representation.",
                "bad_points": "More likely to overfit small validation sets.",
            },
            {
                "round": 4,
                "name": "svr-rbf",
                "regressor": "svr",
                "kernel": "rbf",
                "C": 1.0,
                "epsilon": 0.2,
                "gamma": "scale",
                "n_components": 100,
                "description": "SVR with RBF kernel on PCA-reduced features.",
                "usual_use": "Nonlinear methylation-age relationships.",
                "good_points": "Can model nonlinear effects well.",
                "bad_points": "Sensitive to hyperparameters and slower to train.",
            },
            {
                "round": 4,
                "name": "svr-linear",
                "regressor": "svr",
                "kernel": "linear",
                "C": 1.0,
                "epsilon": 0.2,
                "n_components": 100,
                "description": "Linear SVR on PCA-reduced features.",
                "usual_use": "Margin-based linear regression baseline.",
                "good_points": "Often robust in high dimensions.",
                "bad_points": "Still slower than Ridge/ElasticNet.",
            },
            {
                "round": 4,
                "name": "rf-500",
                "regressor": "rf",
                "n_estimators": 500,
                "max_features": "sqrt",
                "max_samples": 0.8,
                "max_depth": 10,
                "min_samples_leaf": 2,
                "description": "Random forest with feature and sample subsampling.",
                "usual_use": "Nonlinear interactions and robust averaging.",
                "good_points": "Less sensitive to scaling and outliers.",
                "bad_points": "Can overfit in p >> n if not constrained.",
            },
            {
                "round": 4,
                "name": "gbr-100",
                "regressor": "gbr",
                "n_estimators": 100,
                "max_depth": 3,
                "subsample": 0.8,
                "max_features": "sqrt",
                "learning_rate": 0.05,
                "description": "Gradient boosting regressor with subsampling.",
                "usual_use": "Structured nonlinear tabular relationships.",
                "good_points": "Strong predictive baseline on many tabular tasks.",
                "bad_points": "Can overfit and is sensitive to tuning.",
            },
            {
                "round": 5,
                "name": "xgb-base",
                "regressor": "xgb",
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "colsample_bytree": 0.3,
                "subsample": 0.8,
                "description": "XGBoost with aggressive feature subsampling.",
                "usual_use": "High-performance nonlinear tabular modeling.",
                "good_points": "Often strong predictive power with regularization.",
                "bad_points": "More computationally expensive.",
            },
            {
                "round": 5,
                "name": "lgbm-base",
                "regressor": "lgbm",
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "colsample_bytree": 0.3,
                "subsample": 0.8,
                "num_leaves": 31,
                "description": "LightGBM with aggressive feature subsampling.",
                "usual_use": "Fast boosted tree modeling on high-dimensional data.",
                "good_points": "Efficient and can capture nonlinear interactions.",
                "bad_points": "Can overfit without strict regularization.",
            },
            {
                "round": 6,
                "name": "stacking-top3",
                "regressor": "stacking",
                "description": "Stacking ensemble built from top-3 candidates.",
                "usual_use": "Combine complementary models for final accuracy.",
                "good_points": "Can improve robustness and generalization.",
                "bad_points": "Harder to interpret and slower to train.",
            },
        ]

    def _build_preprocess(self, X, scale_numeric=True):
        numeric_columns = [col for col in X.columns if col != self.gender_column]
        categorical_columns = [self.gender_column] if self.gender_column in X.columns else []

        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=numeric_steps), numeric_columns),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "one_hot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )

    def _build_stacking_model(self, X, params):
        base_names = params.get("base_candidates", [])
        if len(base_names) < 2:
            raise ValueError("Stacking requires at least two base candidates.")

        estimators = []
        for base_name in base_names:
            if base_name not in self._active_candidate_lookup:
                raise KeyError(f"Unknown stacking base candidate: {base_name}")
            base_params = deepcopy(self._active_candidate_lookup[base_name])
            estimator = self._build_model(X=X, params=base_params, allow_stacking=False)
            estimators.append((base_name, estimator))

        return StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0)),
            passthrough=False,
            n_jobs=1,
        )

    def _build_model(self, X, params, allow_stacking=True):
        regressor_name = params["regressor"]

        if regressor_name == "stacking":
            if not allow_stacking:
                raise ValueError("Nested stacking is not supported.")
            return self._build_stacking_model(X=X, params=params)

        if regressor_name == "ridge":
            regressor = Ridge(alpha=params["alpha"])
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", regressor),
                ]
            )

        if regressor_name == "lasso":
            regressor = Lasso(
                alpha=params["alpha"],
                max_iter=10000,
                tol=1e-3,
                selection="random",
                random_state=self.random_state,
            )
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", regressor),
                ]
            )

        if regressor_name == "elasticnet":
            regressor = ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=10000,
                tol=1e-3,
                selection="random",
                random_state=self.random_state,
            )
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", regressor),
                ]
            )

        if regressor_name == "varthresh_elasticnet":
            regressor = ElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=10000,
                tol=1e-3,
                selection="random",
                random_state=self.random_state,
            )
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    (
                        "variance_threshold",
                        VarianceThreshold(threshold=params["variance_threshold"]),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", regressor),
                ]
            )

        if regressor_name == "pca_ridge":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    (
                        "pca",
                        PCA(
                            n_components=params["n_components"],
                            svd_solver="randomized",
                            random_state=self.random_state,
                        ),
                    ),
                    ("regressor", Ridge(alpha=params["alpha"])),
                ]
            )

        if regressor_name == "pca_elasticnet":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    (
                        "pca",
                        PCA(
                            n_components=params["n_components"],
                            svd_solver="randomized",
                            random_state=self.random_state,
                        ),
                    ),
                    (
                        "regressor",
                        ElasticNet(
                            alpha=params["alpha"],
                            l1_ratio=params["l1_ratio"],
                            max_iter=10000,
                            tol=1e-3,
                            selection="random",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        if regressor_name == "pls":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", PLSRegression(n_components=params["n_components"])),
                ]
            )

        if regressor_name == "svr":
            steps = [("preprocess", self._build_preprocess(X, scale_numeric=True))]
            if params.get("n_components") is not None:
                steps.append(
                    (
                        "pca",
                        PCA(
                            n_components=params["n_components"],
                            svd_solver="randomized",
                            random_state=self.random_state,
                        ),
                    )
                )
            steps.append(
                (
                    "regressor",
                    SVR(
                        kernel=params["kernel"],
                        C=params["C"],
                        epsilon=params["epsilon"],
                        gamma=params.get("gamma", "scale"),
                    ),
                )
            )
            return Pipeline(steps=steps)

        if regressor_name == "rf":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=params["n_estimators"],
                            max_features=params["max_features"],
                            max_samples=params["max_samples"],
                            max_depth=params["max_depth"],
                            min_samples_leaf=params["min_samples_leaf"],
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        if regressor_name == "gbr":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    (
                        "regressor",
                        GradientBoostingRegressor(
                            n_estimators=params["n_estimators"],
                            max_depth=params["max_depth"],
                            subsample=params["subsample"],
                            max_features=params["max_features"],
                            learning_rate=params["learning_rate"],
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        if regressor_name == "xgb":
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed.")
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    (
                        "regressor",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=params["n_estimators"],
                            max_depth=params["max_depth"],
                            learning_rate=params["learning_rate"],
                            colsample_bytree=params["colsample_bytree"],
                            subsample=params["subsample"],
                            random_state=self.random_state,
                            n_jobs=4,
                        ),
                    ),
                ]
            )

        if regressor_name == "lgbm":
            if LGBMRegressor is None:
                raise ImportError("lightgbm is not installed.")
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    (
                        "regressor",
                        LGBMRegressor(
                            n_estimators=params["n_estimators"],
                            max_depth=params["max_depth"],
                            learning_rate=params["learning_rate"],
                            colsample_bytree=params["colsample_bytree"],
                            subsample=params["subsample"],
                            num_leaves=params["num_leaves"],
                            random_state=self.random_state,
                            n_jobs=4,
                            verbosity=-1,
                        ),
                    ),
                ]
            )

        raise ValueError(f"Unknown regressor type: {regressor_name}")

    @staticmethod
    def custom_evaluation(y_true, y_pred):
        y_true_vector = np.asarray(y_true).ravel()
        y_pred_vector = np.asarray(y_pred).ravel()
        residuals = y_true_vector - y_pred_vector
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        custom_score = 0.7 * mae + 0.3 * rmse
        return {
            "mae": mae,
            "rmse": rmse,
            "custom_score": custom_score,
        }

    def _get_active_candidates(self):
        if self.max_round < 1 or self.max_round > 6:
            raise ValueError("max_round must be in [1, 6].")
        active = [c for c in self.candidate_params if c["round"] <= self.max_round]
        if not active:
            raise RuntimeError("No active candidates found for current max_round.")
        return [deepcopy(candidate) for candidate in active]

    def _selection_value(self, metrics):
        return float(metrics[self.selection_metric])

    def _evaluate_candidate(self, X_fit, y_fit, X_valid, y_valid, params):
        try:
            candidate = self._build_model(X=X_fit, params=params)
            candidate.fit(X_fit, y_fit)
            y_valid_pred = np.asarray(candidate.predict(X_valid)).ravel()
            metrics = self.custom_evaluation(y_valid, y_valid_pred)
            result = {**params, **metrics, "status": "ok"}
            result["selection_metric"] = self.selection_metric
            result["selection_value"] = self._selection_value(metrics)
            return result
        except Exception as error:
            LOGGER.warning("Candidate %s failed: %s", params["name"], error)
            result = {**params}
            result.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "mae": float("inf"),
                    "rmse": float("inf"),
                    "custom_score": float("inf"),
                    "selection_metric": self.selection_metric,
                    "selection_value": float("inf"),
                }
            )
            return result

    def _tune_hyperparameters(self, X, y):
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X,
            y,
            test_size=self.validation_split,
            random_state=self.random_state,
            shuffle=True,
        )

        active_candidates = self._get_active_candidates()
        self._active_candidate_lookup = {c["name"]: deepcopy(c) for c in active_candidates}
        non_stacking = [c for c in active_candidates if c["regressor"] != "stacking"]
        stacking_templates = [c for c in active_candidates if c["regressor"] == "stacking"]

        LOGGER.info(
            "Starting tuning: train=%d valid=%d candidates=%d selection_metric=%s max_round=%d",
            len(X_fit),
            len(X_valid),
            len(active_candidates),
            self.selection_metric,
            self.max_round,
        )

        all_results = []
        for index, params in enumerate(non_stacking, start=1):
            LOGGER.info(
                "Evaluating candidate %d/%d: %s (%s)",
                index,
                len(non_stacking),
                params["name"],
                params["regressor"],
            )
            result = self._evaluate_candidate(
                X_fit=X_fit,
                y_fit=y_fit,
                X_valid=X_valid,
                y_valid=y_valid,
                params=params,
            )
            all_results.append(result)
            if result["status"] == "ok":
                LOGGER.info(
                    "Candidate %s metrics: mae=%.4f rmse=%.4f custom_score=%.4f",
                    result["name"],
                    result["mae"],
                    result["rmse"],
                    result["custom_score"],
                )

        successful_non_stacking = [
            result
            for result in all_results
            if result["status"] == "ok" and np.isfinite(result["selection_value"])
        ]

        if stacking_templates and len(successful_non_stacking) >= 3:
            successful_non_stacking.sort(key=lambda item: item["selection_value"])
            top3_names = [result["name"] for result in successful_non_stacking[:3]]
            LOGGER.info("Building stacking candidate from top3: %s", ", ".join(top3_names))
            for template in stacking_templates:
                stacking_params = deepcopy(template)
                stacking_params["base_candidates"] = top3_names
                stacking_result = self._evaluate_candidate(
                    X_fit=X_fit,
                    y_fit=y_fit,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    params=stacking_params,
                )
                all_results.append(stacking_result)
        elif stacking_templates:
            LOGGER.warning("Skipping stacking because fewer than 3 successful base models.")

        all_results.sort(key=lambda item: item["selection_value"])
        for rank, result in enumerate(all_results, start=1):
            result["rank"] = rank

        self.validation_results_ = all_results

        successful_results = [result for result in all_results if result["status"] == "ok"]
        if not successful_results:
            raise RuntimeError("No candidate model trained successfully.")

        best_result = successful_results[0]
        LOGGER.info(
            "Best candidate selected: %s (%s) %s=%.4f",
            best_result["name"],
            best_result["regressor"],
            self.selection_metric,
            best_result["selection_value"],
        )
        return best_result

    def fit(self, X, y):
        y_vector = np.asarray(y).ravel()
        LOGGER.info(
            "Model.fit samples=%d features=%d validation_split=%.3f max_round=%d selection_metric=%s",
            len(X),
            len(X.columns),
            self.validation_split,
            self.max_round,
            self.selection_metric,
        )

        if self.validation_split < 0.0 or self.validation_split >= 1.0:
            raise ValueError("validation_split must be in [0, 1).")
        if self.selection_metric not in {"rmse", "mae", "custom_score"}:
            raise ValueError("selection_metric must be one of: rmse, mae, custom_score.")

        if self.validation_split > 0.0:
            self.best_params_ = self._tune_hyperparameters(X, y_vector)
        else:
            self.validation_results_ = []
            active_candidates = self._get_active_candidates()
            self._active_candidate_lookup = {
                candidate["name"]: deepcopy(candidate) for candidate in active_candidates
            }
            self.best_params_ = active_candidates[0]
            LOGGER.info(
                "Validation disabled. Using default active model: %s (%s).",
                self.best_params_["name"],
                self.best_params_["regressor"],
            )

        self.model = self._build_model(X=X, params=self.best_params_)
        self.model.fit(X, y_vector)
        LOGGER.info(
            "Final model trained on full data with %s (%s).",
            self.best_params_["name"],
            self.best_params_["regressor"],
        )

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        LOGGER.info("Generating predictions for %d samples.", len(X))
        return np.asarray(self.model.predict(X)).ravel()

    def get_tuning_report(self):
        return {
            "tuning_enabled": self.validation_split > 0.0,
            "validation_split": self.validation_split,
            "random_state": self.random_state,
            "max_round": self.max_round,
            "selection_metric": self.selection_metric,
            "best_model": self.best_params_,
            "results": self.validation_results_,
        }

    def get_top_successful_candidates(self, top_k=3):
        if self.validation_results_:
            successful = [
                result
                for result in self.validation_results_
                if result.get("status") == "ok" and np.isfinite(result.get("selection_value", np.inf))
            ]
            return [deepcopy(result) for result in successful[:top_k]]

        if self.best_params_ is not None:
            return [deepcopy(self.best_params_)]

        return []

    def cross_validate_candidates(self, X, y, candidates, cv_folds=5):
        y_vector = np.asarray(y).ravel()
        cv_results = []
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for candidate in candidates:
            candidate_name = candidate["name"]
            params = deepcopy(self._active_candidate_lookup.get(candidate_name, {}))
            params.update(deepcopy(candidate))
            LOGGER.info("Cross-validating candidate %s with %d folds.", candidate_name, cv_folds)

            fold_metrics = []
            failure_reason = None
            for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
                X_train_fold = X.iloc[train_idx]
                X_valid_fold = X.iloc[valid_idx]
                y_train_fold = y_vector[train_idx]
                y_valid_fold = y_vector[valid_idx]

                try:
                    fold_model = self._build_model(X=X_train_fold, params=params)
                    fold_model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = np.asarray(fold_model.predict(X_valid_fold)).ravel()
                    metrics = self.custom_evaluation(y_valid_fold, y_pred_fold)
                    fold_metrics.append(metrics)
                    LOGGER.info(
                        "  CV fold %d/%d for %s: rmse=%.4f mae=%.4f custom_score=%.4f",
                        fold_idx,
                        cv_folds,
                        candidate_name,
                        metrics["rmse"],
                        metrics["mae"],
                        metrics["custom_score"],
                    )
                except Exception as error:
                    failure_reason = str(error)
                    LOGGER.warning(
                        "  CV fold %d/%d failed for %s: %s",
                        fold_idx,
                        cv_folds,
                        candidate_name,
                        error,
                    )
                    break

            if failure_reason is not None or not fold_metrics:
                cv_results.append(
                    {
                        "name": candidate_name,
                        "regressor": params["regressor"],
                        "status": "failed",
                        "cv_folds": cv_folds,
                        "error": failure_reason or "No valid fold results",
                        "rmse_mean": float("inf"),
                        "rmse_std": float("inf"),
                        "mae_mean": float("inf"),
                        "mae_std": float("inf"),
                        "custom_score_mean": float("inf"),
                        "custom_score_std": float("inf"),
                    }
                )
                continue

            rmse_values = [m["rmse"] for m in fold_metrics]
            mae_values = [m["mae"] for m in fold_metrics]
            custom_values = [m["custom_score"] for m in fold_metrics]
            cv_results.append(
                {
                    "name": candidate_name,
                    "regressor": params["regressor"],
                    "status": "ok",
                    "cv_folds": cv_folds,
                    "rmse_mean": float(np.mean(rmse_values)),
                    "rmse_std": float(np.std(rmse_values)),
                    "mae_mean": float(np.mean(mae_values)),
                    "mae_std": float(np.std(mae_values)),
                    "custom_score_mean": float(np.mean(custom_values)),
                    "custom_score_std": float(np.std(custom_values)),
                }
            )

        cv_results.sort(key=lambda item: item["rmse_mean"])
        for rank, result in enumerate(cv_results, start=1):
            result["cv_rank"] = rank

        return cv_results

    def build_candidate_estimator(self, X_reference, candidate):
        candidate_name = candidate["name"]
        params = deepcopy(self._active_candidate_lookup.get(candidate_name, {}))
        params.update(deepcopy(candidate))
        estimator = self._build_model(X=X_reference, params=params)
        return estimator, params

    def predict_candidates_on_test(self, X_train, y_train, X_test, candidates):
        y_vector = np.asarray(y_train).ravel()
        prediction_results = []

        for candidate in candidates:
            candidate_name = candidate["name"]
            try:
                estimator, params = self.build_candidate_estimator(
                    X_reference=X_train,
                    candidate=candidate,
                )
                estimator.fit(X_train, y_vector)
                y_pred = np.asarray(estimator.predict(X_test)).ravel()
                prediction_results.append(
                    {
                        "name": candidate_name,
                        "regressor": params["regressor"],
                        "status": "ok",
                        "predictions": y_pred,
                    }
                )
            except Exception as error:
                prediction_results.append(
                    {
                        "name": candidate_name,
                        "regressor": candidate.get("regressor", "unknown"),
                        "status": "failed",
                        "error": str(error),
                    }
                )

        return prediction_results
