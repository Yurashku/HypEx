from typing import Any, Dict, Optional
import numpy as np
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole
from ..executor import MLExecutor
from ..utils import ExperimentDataEnum


class CUPACExecutor(MLExecutor):
    def __init__(
        self,
        cupac_features: Dict[str, list],
        key: Any = "",
        models: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_features = cupac_features
        self.models = models
        self.n_folds = n_folds
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.variance_reduction = None
        self.feature_importances_ = None
        self.is_fitted = False
        self.model_results_ = {}

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        cupac_features: Dict[str, list],
        models: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        instance = cls(
            cupac_features=cupac_features,
            models=models,
            n_folds=n_folds,
            random_state=random_state,
        )
        instance.fit(data)
        return instance.predict(data)

    def fit(self, X: Dataset) -> "CUPACExecutor":
        import pandas as pd
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            CatBoostRegressor = None

        # Supported models
        all_models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=0.5),
            "lasso": Lasso(alpha=0.01, max_iter=10000),
        }
        if CatBoostRegressor:
            all_models["catboost"] = CatBoostRegressor(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                silent=True,
                random_state=self.random_state,
                allow_writing_files=False,
            )

        # Check for explicit model selection
        explicit_model = None
        if "model" in self.cupac_features:
            model_name = self.cupac_features["model"].lower()
            if model_name not in all_models:
                raise ValueError(f"Unknown model '{model_name}'. Supported: {list(all_models.keys())}")
            explicit_model = all_models[model_name]

        df = X.data.copy()
        self.fitted_models = {}
        self.best_model_names = {}
        for target_col, covariates in self.cupac_features.items():
            if target_col == "model":
                continue
            X_cov = df[covariates]
            y = df[target_col]
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            if explicit_model is not None:
                # Use only the specified model
                model = explicit_model.__class__(**explicit_model.get_params())
                model.fit(X_cov, y)
                self.fitted_models[target_col] = model
                self.best_model_names[target_col] = model_name
            else:
                # Auto-select best model
                best_score = -np.inf
                best_model = None
                best_model_name = None
                for name, model in all_models.items():
                    fold_var_reductions = []
                    for train_idx, val_idx in kf.split(X_cov):
                        X_train, X_val = X_cov.iloc[train_idx], X_cov.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        m = model.__class__(**model.get_params())
                        m.fit(X_train, y_train)
                        pred = m.predict(X_val)
                        fold_var_reductions.append(self._calculate_variance_reduction(y_val, pred))
                    score = np.nanmean(fold_var_reductions)
                    if score > best_score:
                        best_score = score
                        best_model = model.__class__(**model.get_params())
                        best_model_name = name
                best_model.fit(X_cov, y)
                self.fitted_models[target_col] = best_model
                self.best_model_names[target_col] = best_model_name
        self.is_fitted = True
        return self

    def predict(self, X: Dataset) -> Dict[str, np.ndarray]:
        df = X.data.copy()
        result = {}
        for target_col, covariates in self.cupac_features.items():
            if target_col == "model":
                continue
            model = self.fitted_models.get(target_col)
            if model is None:
                raise RuntimeError(f"Model for {target_col} not fitted. Call fit() first.")
            X_cov = df[covariates]
            y = df[target_col]
            pred = model.predict(X_cov)
            y_adj = y - pred + np.mean(y)
            result[f"{target_col}_cupac"] = y_adj
        return result

    @staticmethod
    def _calculate_variance_reduction(y, pred):
        pred_centered = pred - np.mean(pred)
        if np.var(pred_centered) < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
        y_adj = y - theta * pred_centered
        return max(0, (1 - np.var(y_adj) / np.var(y)) * 100)

    def execute(self, data: ExperimentData) -> ExperimentData:
        self.fit(data.ds)
        cupac_result = self.predict(data.ds)
        for col, values in cupac_result.items():
            ds_ml = Dataset.from_dict(
                {col: values},
                roles={col: TargetRole()},
                index=data.ds.index,
            )
            data.set_value(
                ExperimentDataEnum.ml,
                executor_id=col,
                value=ds_ml,
                role=TargetRole(),
            )
            # Добавить колонку в основной Dataset и назначить ей роль TargetRole
            data.ds.add_column(values, {col: TargetRole()})
        return data
