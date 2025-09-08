from typing import Any, Optional
import numpy as np
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole
from ..executor import MLExecutor
from ..utils import ExperimentDataEnum


from ..utils.models import CUPAC_MODELS

from typing import Union, Sequence
from ..utils.models import CUPAC_MODELS

class CUPACExecutor(MLExecutor):
    """Executor that fits predictive models to pre-period covariates and adjusts target
    features using the CUPAC approach (model-based prediction adjustment similar to CUPED).

    cupac_features should be a mapping: {target_feature: list[pre_target_feature, ...]}.
    It may also include a top-level key 'model' to request a specific model name.
    """

    def __init__(
        self,
        cupac_features: dict[str, list],
        cupac_model: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            cupac_features: dict[str, list] — parameters for CUPAC, e.g. {"target": ["cov1", "cov2"]}
            cupac_model: str or list of str — model name (e.g. 'linear', 'ridge', 'lasso', 'catboost') or list of model names to try.
            key: key for executor.
            n_folds: number of folds for cross-validation.
            random_state: random seed.
        """
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_features = cupac_features
        self.cupac_model = cupac_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_models: dict[str, Any] = {}
        self.best_model_names: dict[str, str] = {}
        self.is_fitted = False

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        cupac_features: dict[str, list],
        n_folds: int = 5,
        random_state: Optional[int] = None,
        cupac_model: Optional[str] = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        instance = cls(
            cupac_features=cupac_features,
            n_folds=n_folds,
            random_state=random_state,
            cupac_model=cupac_model,
        )
        instance.fit(data)
        return instance.predict(data)

    def _select_explicit_models(self, all_models: dict[str, Any]) -> Sequence[str]:
        """
        Returns a list of model names explicitly specified by the user (or all available if None).
        If cupac_model is a string, returns [cupac_model]. If a list, returns the list.
        If None, returns all available models.
        """
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

    def fit(self, X: Dataset) -> "CUPACExecutor":
        from sklearn.base import clone
        from sklearn.model_selection import KFold
        all_models = {k.lower(): v for k, v in CUPAC_MODELS.items()}

        df = X.data.copy()
        self.fitted_models = {}
        self.best_model_names = {}

        explicit_models = self._select_explicit_models(all_models)

        for target_feature, pre_target_features in self.cupac_features.items():
            if target_feature == "model":
                continue

            X_cov = df[pre_target_features]
            y = df[target_feature]
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

            # If only one model is specified, use it
            if len(explicit_models) == 1:
                model_proto = all_models[explicit_models[0]]
                model = clone(model_proto)
                model.fit(X_cov, y)
                self.fitted_models[target_feature] = model
                self.best_model_names[target_feature] = explicit_models[0]
                continue

            # If a list of models is specified, try only those
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

    def predict(self, X: Dataset) -> dict[str, np.ndarray]:
        df = X.data.copy()
        result = {}
        for target_feature, pre_target_features in self.cupac_features.items():
            if target_feature == "model":
                continue
            model = self.fitted_models.get(target_feature)
            if model is None:
                raise RuntimeError(f"Model for {target_feature} not fitted. Call fit() first.")
            X_cov = df[pre_target_features]
            y = df[target_feature]
            pred = model.predict(X_cov)
            y_adj = y - pred + np.mean(y)
            result[f"{target_feature}_cupac"] = y_adj
        return result

    @staticmethod
    def _calculate_variance_reduction(y: np.ndarray, pred: np.ndarray) -> float:
        """Calculate variance reduction percentage between y and prediction pred.

        Args:
            y: true values (1D numpy array)
            pred: predictions (1D numpy array)

        Returns:
            Variance reduction percentage (float, >= 0).
        """
        pred_centered = pred - np.mean(pred)
        if np.var(pred_centered) < 1e-10:
            return 0.0
        theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
        y_adj = y - theta * pred_centered
        return float(max(0, (1 - np.var(y_adj) / np.var(y)) * 100))

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
            # Add column to main Dataset and set its role
            data.ds.add_column(values, {col: TargetRole()})
        return data
