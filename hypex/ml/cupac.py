from typing import Any, Optional
import numpy as np
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole
from ..executor import MLExecutor
from ..utils import ExperimentDataEnum



from ..extensions.cupac import CupacExtension

from typing import Union, Sequence
from ..utils.models import CUPAC_MODELS

class CUPACExecutor(MLExecutor):
    """Executor that fits predictive models to pre-period covariates and adjusts target
    features using the CUPAC approach (model-based prediction adjustment similar to CUPED).
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
        self.extension = CupacExtension(
            cupac_features=self.cupac_features,
            cupac_model=self.cupac_model,
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        self.extension.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: Dataset) -> dict[str, np.ndarray]:
        if not hasattr(self, "extension"):
            raise RuntimeError("CUPACExecutor not fitted. Call fit() first.")
        return self.extension.predict(X)

    # @staticmethod
    # def _calculate_variance_reduction(y: np.ndarray, pred: np.ndarray) -> float:
    #     """Calculate variance reduction percentage between y and prediction pred.

    #     Args:
    #         y: true values (1D numpy array)
    #         pred: predictions (1D numpy array)

    #     Returns:
    #         Variance reduction percentage (float, >= 0).
    #     """
    #     pred_centered = pred - np.mean(pred)
    #     if np.var(pred_centered) < 1e-10:
    #         return 0.0
    #     theta = np.cov(y, pred_centered)[0, 1] / np.var(pred_centered)
    #     y_adj = y - theta * pred_centered
    #     return float(max(0, (1 - np.var(y_adj) / np.var(y)) * 100))

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
            data.ds.add_column(values, {col: TargetRole()})
        return data
