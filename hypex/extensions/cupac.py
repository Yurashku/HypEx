from __future__ import annotations
from typing import Any, Sequence, Optional, Dict
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..dataset import Dataset, TargetRole
from .abstract import MLExtension
from ..utils.models import CUPAC_MODELS

class CupacExtension(MLExtension):
    """
    Extension for CUPAC variance reduction. Supports multiple models and auto-selection.
    """
    def __init__(
        self,
        cupac_features: Dict[str, Sequence[str]],
        cupac_model: Optional[str | Sequence[str]] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.cupac_features = cupac_features
        self.cupac_model = cupac_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_models: Dict[str, Any] = {}
        self.best_model_names: Dict[str, str] = {}
        self.is_fitted = False

    def fit(self, X: Dataset, Y: Dataset = None) -> 'CupacExtension':
        all_models = {k.lower(): v for k, v in CUPAC_MODELS.items()}
        df = X.data.copy()
        self.fitted_models = {}
        self.best_model_names = {}
        explicit_models = self._select_explicit_models(all_models)
        for target_feature, pre_target_features in self.cupac_features.items():
            X_cov = df[pre_target_features]
            y = df[target_feature]
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            if len(explicit_models) == 1:
                model_proto = all_models[explicit_models[0]]
                model = clone(model_proto)
                model.fit(X_cov, y)
                self.fitted_models[target_feature] = model
                self.best_model_names[target_feature] = explicit_models[0]
                continue
            best_score = -np.inf
            best_model = None
            best_model_name = None
            for name in explicit_models:
                model_proto = all_models[name]
                fold_var_reductions: list[float] = []
                for train_idx, val_idx in kf.split(X_cov):
                    X_train, X_val = X_cov.iloc[train_idx], X_cov.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    m = clone(model_proto)
                    m.fit(X_train, y_train)
                    pred = m.predict(X_val)
                    fold_var_reductions.append(self._calculate_variance_reduction(y_val.to_numpy(), pred))
                score = float(np.nanmean(fold_var_reductions))
                if score > best_score:
                    best_score = score
                    best_model = clone(model_proto)
                    best_model_name = name
            if best_model is None:
                raise RuntimeError("No model was selected during model search")
            best_model.fit(X_cov, y)
            self.fitted_models[target_feature] = best_model
            self.best_model_names[target_feature] = best_model_name
        self.is_fitted = True
        return self

    def predict(self, X: Dataset) -> Dict[str, np.ndarray]:
        df = X.data.copy()
        result = {}
        for target_feature, pre_target_features in self.cupac_features.items():
            model = self.fitted_models.get(target_feature)
            if model is None:
                raise RuntimeError(f"Model for {target_feature} not fitted. Call fit() first.")
            X_cov = df[pre_target_features]
            y = df[target_feature]
            pred = model.predict(X_cov)
            y_adj = y - pred + np.mean(y)
            result[f"{target_feature}_cupac"] = y_adj
        return result

    def calc(self, data: Dataset, **kwargs):
        self.fit(data)
        return self.predict(data)

    def _select_explicit_models(self, all_models: dict[str, Any]) -> Sequence[str]:
        if self.cupac_model:
            if isinstance(self.cupac_model, str):
                names = [self.cupac_model.lower()]
            else:
                names = [m.lower() for m in self.cupac_model]
            for name in names:
                if name not in all_models:
                    raise ValueError(f"Unknown model '{name}'. Supported: {list(all_models.keys())}")
            return names
        return list(all_models.keys())

    @staticmethod
    def _calculate_variance_reduction(y: np.ndarray, pred: np.ndarray) -> float:
        pred_centered = pred - np.mean(pred)
        if np.var(pred_centered) < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
        y_adj = y - theta * pred_centered
        return float(max(0, (1 - np.var(y_adj) / np.var(y)) * 100))
