from __future__ import annotations
from typing import Any, Sequence, Optional, Dict
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from ..dataset import Dataset, TargetRole
from .abstract import MLExtension
from ..utils.models import CUPAC_MODELS


# --- Pandas-specific extension ---
class CupacPandasExtension:
    def __init__(self, cupac_features, cupac_model, n_folds, random_state):
        self.cupac_features = cupac_features
        self.cupac_model = cupac_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_models = {}
        self.best_model_names = {}
        self.is_fitted = False

    def fit(self, df):
        all_models = {k.lower(): v for k, v in CUPAC_MODELS.items()}
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
                fold_var_reductions = []
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

    def predict(self, df):
        import numpy as np
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

    def _select_explicit_models(self, all_models):
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
    def _calculate_variance_reduction(y, pred):
        import numpy as np
        pred_centered = pred - np.mean(pred)
        if np.var(pred_centered) < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
        y_adj = y - theta * pred_centered
        return float(max(0, (1 - np.var(y_adj) / np.var(y)) * 100))

# --- Main extension ---
class CupacExtension(MLExtension):
    """
    Extension for CUPAC variance reduction. Delegates to backend-specific extension.
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
        self._backend_ext = None

    def _get_backend_ext(self, data: Dataset):
        # Only pandas backend supported for now
        if hasattr(data, 'backend') and hasattr(data.backend, 'data'):
            import pandas as pd
            if isinstance(data.backend.data, pd.DataFrame):
                if self._backend_ext is None:
                    self._backend_ext = CupacPandasExtension(
                        self.cupac_features, self.cupac_model, self.n_folds, self.random_state
                    )
                return self._backend_ext
        raise NotImplementedError("CUPAC only supports pandas backend for now.")

    def fit(self, X: Dataset, Y: Dataset = None) -> 'CupacExtension':
        ext = self._get_backend_ext(X)
        ext.fit(X.backend.data.copy())
        return self

    def predict(self, X: Dataset) -> Dict[str, Any]:
        ext = self._get_backend_ext(X)
        return ext.predict(X.backend.data.copy())

    def calc(self, data: Dataset, **kwargs):
        self.fit(data)
        return self.predict(data)
