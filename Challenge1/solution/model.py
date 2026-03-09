import logging
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoCV, Ridge, RidgeCV
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

try:
    import optuna
except ImportError:
    optuna = None

LOGGER = logging.getLogger(__name__)

DEFAULT_L1_RATIO_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 0.99]


class RelaxedElasticNetRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage relaxed ElasticNet:
    1) ElasticNet for sparse feature selection.
    2) Ridge refit on selected features to reduce shrinkage bias.
    """

    def __init__(
        self,
        alpha=0.01,
        l1_ratio=0.3,
        ridge_alpha=1.0,
        max_iter=10000,
        tol=1e-3,
        random_state=42,
        selection_threshold=1e-8,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selection_threshold = selection_threshold
        self.selector_ = None
        self.refit_model_ = None
        self.support_mask_ = None

    def fit(self, X, y):
        X_array = np.asarray(X)
        y_array = np.asarray(y).ravel()
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        self.selector_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="random",
            random_state=self.random_state,
        )
        self.selector_.fit(X_array, y_array)

        coefficients = np.asarray(self.selector_.coef_).ravel()
        support_mask = np.abs(coefficients) > self.selection_threshold
        if not support_mask.any():
            # Ensure at least one feature survives to avoid degenerate refit.
            best_idx = int(np.argmax(np.abs(coefficients)))
            support_mask = np.zeros_like(coefficients, dtype=bool)
            support_mask[best_idx] = True

        self.support_mask_ = support_mask
        X_selected = X_array[:, self.support_mask_]
        self.refit_model_ = Ridge(alpha=self.ridge_alpha)
        self.refit_model_.fit(X_selected, y_array)
        return self

    def predict(self, X):
        if self.refit_model_ is None or self.support_mask_ is None:
            raise RuntimeError("RelaxedElasticNetRegressor must be fitted before prediction.")
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        X_selected = X_array[:, self.support_mask_]
        return np.asarray(self.refit_model_.predict(X_selected)).ravel()


class Model:
    def __init__(
        self,
        validation_split=0.0,
        random_state=42,
        max_round=6,
        selection_metric="rmse",
        optuna_trials=60,
        optuna_cv_folds=3,
    ):
        self.gender_column = "gender"
        self.validation_split = float(validation_split)
        self.random_state = int(random_state)
        self.max_round = int(max_round)
        self.selection_metric = selection_metric
        self.optuna_trials = int(optuna_trials)
        self.optuna_cv_folds = int(optuna_cv_folds)
        self.model = None
        self.best_params_ = None
        self.validation_results_ = []
        self.candidate_params = self._build_candidate_params()
        self._active_candidate_lookup = {}
        self._max_available_round = max(candidate["round"] for candidate in self.candidate_params)

    def _build_candidate_params(self):
        return [
            {
                "round": 1,
                "name": "elasticnet-cv",
                "regressor": "elasticnet_cv",
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "ElasticNetCV baseline with automatic alpha/l1_ratio tuning.",
                "usual_use": "High-dimensional p >> n regression with correlated predictors.",
                "good_points": "Strong linear baseline and robust regularization path search.",
                "bad_points": "Still linear, may miss nonlinear interactions.",
            },
            {
                "round": 1,
                "name": "ridge-cv",
                "regressor": "ridge_cv",
                "alphas": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "description": "RidgeCV with broad alpha sweep.",
                "usual_use": "Correlated dense signals where sparse selection is not ideal.",
                "good_points": "Very stable and often strong in methylation tasks.",
                "bad_points": "No feature selection.",
            },
            {
                "round": 1,
                "name": "lasso-cv",
                "regressor": "lasso_cv",
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "LassoCV for sparse linear modeling.",
                "usual_use": "Feature selection in high-dimensional settings.",
                "good_points": "Interpretable sparse feature subset.",
                "bad_points": "Can be unstable with strongly correlated features.",
            },
            {
                "round": 1,
                "name": "relaxed-elasticnet",
                "regressor": "relaxed_elasticnet",
                "alpha": 0.01,
                "l1_ratio": 0.3,
                "ridge_alpha": 1.0,
                "description": "Two-stage relaxed ElasticNet (selection + Ridge refit).",
                "usual_use": "Reduce shrinkage bias after sparse selection.",
                "good_points": "Improves coefficient calibration on selected CpGs.",
                "bad_points": "Adds complexity and may overfit if selection is unstable.",
            },
            {
                "round": 2,
                "name": "selectk500-elasticnetcv",
                "regressor": "selectk_elasticnet_cv",
                "k_features": 500,
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "Top-500 age-correlated features + ElasticNetCV.",
                "usual_use": "Aggressive feature preselection before linear tuning.",
                "good_points": "Strong denoising and faster training.",
                "bad_points": "Can drop weak but jointly useful signals.",
            },
            {
                "round": 2,
                "name": "selectk1000-elasticnetcv",
                "regressor": "selectk_elasticnet_cv",
                "k_features": 1000,
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "Top-1000 age-correlated features + ElasticNetCV.",
                "usual_use": "Balanced preselection for p >> n methylation data.",
                "good_points": "Good trade-off between signal retention and denoising.",
                "bad_points": "Still linear.",
            },
            {
                "round": 2,
                "name": "selectk2000-elasticnetcv",
                "regressor": "selectk_elasticnet_cv",
                "k_features": 2000,
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "Top-2000 age-correlated features + ElasticNetCV.",
                "usual_use": "Retain broader CpG signal while reducing dimensionality.",
                "good_points": "Often robust in epigenetic age prediction.",
                "bad_points": "More compute than smaller k.",
            },
            {
                "round": 2,
                "name": "selectk5000-elasticnetcv",
                "regressor": "selectk_elasticnet_cv",
                "k_features": 5000,
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "Top-5000 age-correlated features + ElasticNetCV.",
                "usual_use": "Large preselection retaining richer feature space.",
                "good_points": "Less information loss from filtering.",
                "bad_points": "Higher risk of noise and overfitting.",
            },
            {
                "round": 2,
                "name": "selectk2000-relaxed-elasticnet",
                "regressor": "selectk_relaxed_elasticnet",
                "k_features": 2000,
                "alpha": 0.01,
                "l1_ratio": 0.3,
                "ridge_alpha": 1.0,
                "description": "Top-2000 feature filter + relaxed ElasticNet.",
                "usual_use": "Sparse selection with lower post-selection bias.",
                "good_points": "Combines denoising and relaxed refit.",
                "bad_points": "Two-stage model can be less stable fold-to-fold.",
            },
            {
                "round": 3,
                "name": "pls-10",
                "regressor": "pls",
                "n_components": 10,
                "description": "PLSRegression with 10 latent components.",
                "usual_use": "Supervised dimensionality reduction in p >> n.",
                "good_points": "Uses target information during projection.",
                "bad_points": "Can overfit if components are not controlled.",
            },
            {
                "round": 3,
                "name": "pls-20",
                "regressor": "pls",
                "n_components": 20,
                "description": "PLSRegression with 20 latent components.",
                "usual_use": "Richer supervised latent representation.",
                "good_points": "Captures additional age-related variance.",
                "bad_points": "Higher model complexity.",
            },
            {
                "round": 3,
                "name": "pls-30",
                "regressor": "pls",
                "n_components": 30,
                "description": "PLSRegression with 30 latent components.",
                "usual_use": "Extended supervised latent modeling.",
                "good_points": "Flexible representation of methylation-age signal.",
                "bad_points": "Increased overfitting risk.",
            },
            {
                "round": 3,
                "name": "pca200-elasticnetcv",
                "regressor": "pca_elasticnet_cv",
                "n_components": 200,
                "l1_ratio_grid": DEFAULT_L1_RATIO_GRID,
                "n_alphas": 100,
                "cv_folds": 5,
                "description": "PCA(<=200 safe-capped) + ElasticNetCV.",
                "usual_use": "Unsupervised compression followed by regularized regression.",
                "good_points": "Controls dimensionality aggressively.",
                "bad_points": "Unsupervised PCA may drop age-informative variance.",
            },
            {
                "round": 4,
                "name": "svr-linear-pca100",
                "regressor": "svr",
                "kernel": "linear",
                "C": 1.0,
                "epsilon": 0.2,
                "n_components": 100,
                "description": "Linear SVR on safe-capped PCA features.",
                "usual_use": "Margin-based linear modeling with latent compression.",
                "good_points": "Robust objective and regularized fit.",
                "bad_points": "Slower than linear regressions.",
            },
            {
                "round": 5,
                "name": "selectk500-lgbm",
                "regressor": "selectk_lgbm",
                "k_features": 500,
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.6,
                "num_leaves": 31,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "description": "Feature-preselected LightGBM on top-500 CpGs.",
                "usual_use": "Nonlinear tabular patterns after aggressive denoising.",
                "good_points": "Captures interactions missed by linear models.",
                "bad_points": "Requires careful regularization with small n.",
            },
            {
                "round": 5,
                "name": "selectk500-xgb",
                "regressor": "selectk_xgb",
                "k_features": 500,
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.6,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "description": "Feature-preselected XGBoost on top-500 CpGs.",
                "usual_use": "Nonlinear effects in reduced feature space.",
                "good_points": "Strong gradient-boosted tree baseline.",
                "bad_points": "Potentially slower and sensitive to tuning.",
            },
            {
                "round": 6,
                "name": "stacking-diverse",
                "regressor": "stacking",
                "preferred_base_candidates": [
                    "selectk2000-elasticnetcv",
                    "pls-20",
                    "selectk500-lgbm",
                    "selectk500-xgb",
                ],
                "description": "Diverse stacking of linear, latent, and boosted models.",
                "usual_use": "Blend complementary error profiles.",
                "good_points": "Can improve robustness vs single-model solutions.",
                "bad_points": "Harder to interpret and slower to train.",
            },
            {
                "round": 7,
                "name": "optuna-selectk-elasticnet",
                "regressor": "optuna_elasticnet",
                "k_min": 200,
                "k_max": 5000,
                "alpha_min": 1e-4,
                "alpha_max": 1.0,
                "l1_ratio_min": 0.05,
                "l1_ratio_max": 0.99,
                "optuna_cv_folds": self.optuna_cv_folds,
                "n_trials": self.optuna_trials,
                "description": "Bayesian optimization (Optuna) for SelectK + ElasticNet.",
                "usual_use": "Efficient joint tuning of feature count and regularization.",
                "good_points": "Explores broader space than manual grids.",
                "bad_points": "Requires Optuna and adds optimization runtime.",
            },
        ]

    def _build_preprocess(self, X, scale_numeric=True):
        numeric_columns = [col for col in X.columns if col != self.gender_column]
        categorical_columns = [self.gender_column] if self.gender_column in X.columns else []

        transformers = []
        if numeric_columns:
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(steps=numeric_steps), numeric_columns))

        if categorical_columns:
            transformers.append(
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
                )
            )

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @staticmethod
    def _normalize_metadata(params):
        clean = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                clean[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                clean[key] = value.item()
            else:
                clean[key] = value
        return clean

    def _resolve_k_features(self, requested_k, X):
        max_features = max(1, len(X.columns))
        resolved = max(1, int(requested_k))
        return min(resolved, max_features)

    def _resolve_n_components(self, requested_components, X):
        max_components = max(1, min(len(X), len(X.columns)) - 1)
        resolved = max(1, int(requested_components))
        return min(resolved, max_components)

    def _build_elasticnet_cv_regressor(self, params):
        alphas = params.get("alphas")
        if alphas is None:
            # sklearn 1.7+ accepts an integer here and deprecates n_alphas.
            alphas = int(params.get("n_alphas", 100))

        kwargs = {
            "l1_ratio": params.get("l1_ratio_grid", DEFAULT_L1_RATIO_GRID),
            "cv": int(params.get("cv_folds", 5)),
            "max_iter": int(params.get("max_iter", 10000)),
            "tol": float(params.get("tol", 1e-3)),
            "random_state": self.random_state,
            "n_jobs": -1,
            "alphas": alphas,
        }
        return ElasticNetCV(**kwargs)

    def _build_lasso_cv_regressor(self, params):
        alphas = params.get("alphas")
        if alphas is None:
            # sklearn 1.7+ accepts an integer here and deprecates n_alphas.
            alphas = int(params.get("n_alphas", 100))

        kwargs = {
            "cv": int(params.get("cv_folds", 5)),
            "max_iter": int(params.get("max_iter", 10000)),
            "tol": float(params.get("tol", 1e-3)),
            "random_state": self.random_state,
            "n_jobs": -1,
            "alphas": alphas,
        }
        return LassoCV(**kwargs)

    def _build_relaxed_regressor(self, params):
        return RelaxedElasticNetRegressor(
            alpha=float(params.get("alpha", 0.01)),
            l1_ratio=float(params.get("l1_ratio", 0.3)),
            ridge_alpha=float(params.get("ridge_alpha", 1.0)),
            max_iter=int(params.get("max_iter", 10000)),
            tol=float(params.get("tol", 1e-3)),
            random_state=self.random_state,
        )

    def _build_stacking_model(self, X, params):
        base_names = list(params.get("base_candidates", []))
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
            final_estimator=RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)),
            passthrough=False,
            n_jobs=1,
        )

    def _build_model(self, X, params, allow_stacking=True):
        regressor_name = params["regressor"]

        if regressor_name == "stacking":
            if not allow_stacking:
                raise ValueError("Nested stacking is not supported.")
            return self._build_stacking_model(X=X, params=params)

        if regressor_name == "elasticnet_cv":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", self._build_elasticnet_cv_regressor(params)),
                ]
            )

        if regressor_name == "ridge_cv":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", RidgeCV(alphas=params.get("alphas", (0.1, 1.0, 10.0)))),
                ]
            )

        if regressor_name == "lasso_cv":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", self._build_lasso_cv_regressor(params)),
                ]
            )

        if regressor_name == "relaxed_elasticnet":
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", self._build_relaxed_regressor(params)),
                ]
            )

        if regressor_name == "selectk_elasticnet_cv":
            k_features = self._resolve_k_features(params["k_features"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("select_kbest", SelectKBest(score_func=f_regression, k=k_features)),
                    ("regressor", self._build_elasticnet_cv_regressor(params)),
                ]
            )

        if regressor_name == "selectk_relaxed_elasticnet":
            k_features = self._resolve_k_features(params["k_features"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("select_kbest", SelectKBest(score_func=f_regression, k=k_features)),
                    ("regressor", self._build_relaxed_regressor(params)),
                ]
            )

        if regressor_name == "pca_elasticnet_cv":
            n_components = self._resolve_n_components(params["n_components"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    (
                        "pca",
                        PCA(
                            n_components=n_components,
                            svd_solver="randomized",
                            random_state=self.random_state,
                        ),
                    ),
                    ("regressor", self._build_elasticnet_cv_regressor(params)),
                ]
            )

        if regressor_name == "pls":
            n_components = self._resolve_n_components(params["n_components"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("regressor", PLSRegression(n_components=n_components)),
                ]
            )

        if regressor_name == "svr":
            steps = [("preprocess", self._build_preprocess(X, scale_numeric=True))]
            if params.get("n_components") is not None:
                n_components = self._resolve_n_components(params["n_components"], X)
                steps.append(
                    (
                        "pca",
                        PCA(
                            n_components=n_components,
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
                        C=float(params["C"]),
                        epsilon=float(params["epsilon"]),
                        gamma=params.get("gamma", "scale"),
                    ),
                )
            )
            return Pipeline(steps=steps)

        if regressor_name == "selectk_lgbm":
            if LGBMRegressor is None:
                raise ImportError("lightgbm is not installed.")
            k_features = self._resolve_k_features(params["k_features"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    ("select_kbest", SelectKBest(score_func=f_regression, k=k_features)),
                    (
                        "regressor",
                        LGBMRegressor(
                            n_estimators=int(params["n_estimators"]),
                            max_depth=int(params["max_depth"]),
                            learning_rate=float(params["learning_rate"]),
                            subsample=float(params["subsample"]),
                            colsample_bytree=float(params["colsample_bytree"]),
                            num_leaves=int(params["num_leaves"]),
                            min_child_samples=int(params.get("min_child_samples", 20)),
                            reg_alpha=float(params.get("reg_alpha", 0.0)),
                            reg_lambda=float(params.get("reg_lambda", 0.0)),
                            random_state=self.random_state,
                            n_jobs=4,
                            verbosity=-1,
                        ),
                    ),
                ]
            )

        if regressor_name == "selectk_xgb":
            if XGBRegressor is None:
                raise ImportError("xgboost is not installed.")
            k_features = self._resolve_k_features(params["k_features"], X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=False)),
                    ("select_kbest", SelectKBest(score_func=f_regression, k=k_features)),
                    (
                        "regressor",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=int(params["n_estimators"]),
                            max_depth=int(params["max_depth"]),
                            learning_rate=float(params["learning_rate"]),
                            subsample=float(params["subsample"]),
                            colsample_bytree=float(params["colsample_bytree"]),
                            reg_alpha=float(params.get("reg_alpha", 0.0)),
                            reg_lambda=float(params.get("reg_lambda", 1.0)),
                            random_state=self.random_state,
                            n_jobs=4,
                        ),
                    ),
                ]
            )

        if regressor_name == "optuna_elasticnet":
            k_features = self._resolve_k_features(params.get("k_features", 2000), X)
            return Pipeline(
                steps=[
                    ("preprocess", self._build_preprocess(X, scale_numeric=True)),
                    ("select_kbest", SelectKBest(score_func=f_regression, k=k_features)),
                    (
                        "regressor",
                        ElasticNet(
                            alpha=float(params.get("alpha", 0.01)),
                            l1_ratio=float(params.get("l1_ratio", 0.3)),
                            max_iter=int(params.get("max_iter", 10000)),
                            tol=float(params.get("tol", 1e-3)),
                            selection="random",
                            random_state=self.random_state,
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
        if self.max_round < 1 or self.max_round > self._max_available_round:
            raise ValueError(f"max_round must be in [1, {self._max_available_round}].")
        active = [c for c in self.candidate_params if c["round"] <= self.max_round]
        if not active:
            raise RuntimeError("No active candidates found for current max_round.")
        return [deepcopy(candidate) for candidate in active]

    def _selection_value(self, metrics):
        return float(metrics[self.selection_metric])

    def _evaluate_optuna_candidate(self, X_fit, y_fit, X_valid, y_valid, params):
        if optuna is None:
            raise ImportError("optuna is not installed.")

        n_trials = int(params.get("n_trials", self.optuna_trials))
        cv_folds = int(params.get("optuna_cv_folds", self.optuna_cv_folds))
        if n_trials <= 0:
            raise ValueError("Optuna candidate requires n_trials > 0.")
        if cv_folds < 2:
            raise ValueError("optuna_cv_folds must be >= 2.")

        k_min = int(params.get("k_min", 200))
        k_max = int(params.get("k_max", 5000))
        k_min = max(10, min(k_min, k_max))
        alpha_min = float(params.get("alpha_min", 1e-4))
        alpha_max = float(params.get("alpha_max", 1.0))
        l1_ratio_min = float(params.get("l1_ratio_min", 0.05))
        l1_ratio_max = float(params.get("l1_ratio_max", 0.99))

        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        def objective(trial):
            trial_k = self._resolve_k_features(
                trial.suggest_int("k_features", k_min, k_max, log=True),
                X_fit,
            )
            trial_alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
            trial_l1_ratio = trial.suggest_float("l1_ratio", l1_ratio_min, l1_ratio_max)

            fold_rmse = []
            trial_params = {
                "regressor": "optuna_elasticnet",
                "k_features": trial_k,
                "alpha": trial_alpha,
                "l1_ratio": trial_l1_ratio,
            }
            for train_idx, valid_idx in splitter.split(X_fit):
                X_train_fold = X_fit.iloc[train_idx]
                X_valid_fold = X_fit.iloc[valid_idx]
                y_train_fold = np.asarray(y_fit)[train_idx]
                y_valid_fold = np.asarray(y_fit)[valid_idx]
                fold_model = self._build_model(X=X_train_fold, params=trial_params)
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = np.asarray(fold_model.predict(X_valid_fold)).ravel()
                fold_metrics = self.custom_evaluation(y_valid_fold, y_pred_fold)
                fold_rmse.append(fold_metrics["rmse"])
            return float(np.mean(fold_rmse))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        tuned_params = deepcopy(params)
        tuned_params.update(
            {
                "alpha": float(study.best_params["alpha"]),
                "l1_ratio": float(study.best_params["l1_ratio"]),
                "k_features": int(study.best_params["k_features"]),
                "optuna_n_trials": n_trials,
                "optuna_best_value": float(study.best_value),
            }
        )

        candidate = self._build_model(X=X_fit, params=tuned_params)
        candidate.fit(X_fit, y_fit)
        y_valid_pred = np.asarray(candidate.predict(X_valid)).ravel()
        metrics = self.custom_evaluation(y_valid, y_valid_pred)
        result = {**self._normalize_metadata(tuned_params), **metrics, "status": "ok"}
        result["selection_metric"] = self.selection_metric
        result["selection_value"] = self._selection_value(metrics)
        return result

    def _evaluate_candidate(self, X_fit, y_fit, X_valid, y_valid, params):
        try:
            if params["regressor"] == "optuna_elasticnet":
                return self._evaluate_optuna_candidate(
                    X_fit=X_fit,
                    y_fit=y_fit,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    params=params,
                )

            candidate = self._build_model(X=X_fit, params=params)
            candidate.fit(X_fit, y_fit)
            y_valid_pred = np.asarray(candidate.predict(X_valid)).ravel()
            metrics = self.custom_evaluation(y_valid, y_valid_pred)
            result = {**self._normalize_metadata(params), **metrics, "status": "ok"}
            result["selection_metric"] = self.selection_metric
            result["selection_value"] = self._selection_value(metrics)
            return result
        except Exception as error:
            LOGGER.warning("Candidate %s failed: %s", params["name"], error)
            result = self._normalize_metadata(params)
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

    @staticmethod
    def _prefer_diverse_models(sorted_successful_results, initial_names):
        selected = list(initial_names)
        selected_set = set(selected)
        regressor_by_name = {r["name"]: r["regressor"] for r in sorted_successful_results}
        used_regressors = {regressor_by_name.get(name) for name in selected if name in regressor_by_name}

        for result in sorted_successful_results:
            if len(selected) >= 3:
                break
            name = result["name"]
            if name in selected_set:
                continue
            regressor = result["regressor"]
            if regressor not in used_regressors or len(selected) < 2:
                selected.append(name)
                selected_set.add(name)
                used_regressors.add(regressor)

        for result in sorted_successful_results:
            if len(selected) >= 3:
                break
            name = result["name"]
            if name not in selected_set:
                selected.append(name)
                selected_set.add(name)

        return selected[:3]

    def _resolve_stacking_base_candidates(self, template, sorted_successful_results):
        successful_lookup = {result["name"]: result for result in sorted_successful_results}
        preferred_names = template.get("preferred_base_candidates", [])
        initial = [name for name in preferred_names if name in successful_lookup]
        return self._prefer_diverse_models(sorted_successful_results, initial)

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
                self._active_candidate_lookup[result["name"]] = deepcopy(result)
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
        successful_non_stacking.sort(key=lambda item: item["selection_value"])

        for template in stacking_templates:
            base_candidates = self._resolve_stacking_base_candidates(
                template,
                successful_non_stacking,
            )
            if len(base_candidates) < 2:
                LOGGER.warning(
                    "Skipping stacking candidate %s because fewer than two base models are available.",
                    template["name"],
                )
                continue
            stacking_params = deepcopy(template)
            stacking_params["base_candidates"] = base_candidates
            self._active_candidate_lookup[stacking_params["name"]] = deepcopy(stacking_params)
            LOGGER.info(
                "Evaluating stacking candidate %s with bases: %s",
                stacking_params["name"],
                ", ".join(base_candidates),
            )
            stacking_result = self._evaluate_candidate(
                X_fit=X_fit,
                y_fit=y_fit,
                X_valid=X_valid,
                y_valid=y_valid,
                params=stacking_params,
            )
            all_results.append(stacking_result)
            if stacking_result["status"] == "ok":
                self._active_candidate_lookup[stacking_result["name"]] = deepcopy(stacking_result)

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

            metadata = self._normalize_metadata(params)
            if failure_reason is not None or not fold_metrics:
                cv_results.append(
                    {
                        **metadata,
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
                    **metadata,
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
