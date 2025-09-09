from typing import Any, Optional
import numpy as np
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole, StatisticRole
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

    def get_variance_reductions(self):
        if not hasattr(self, "extension"):
            raise RuntimeError("CUPACExecutor not fitted. Call fit() first.")
        return self.extension.get_variance_reductions()



    def execute(self, data: ExperimentData) -> ExperimentData:
        self.fit(data.ds)
        predictions = self.predict(data.ds)
        new_ds = data.ds
        for key, pred in predictions.items():
            if hasattr(pred, 'values'):
                pred = pred.values
            new_ds = new_ds.add_column(data=pred, role={key: TargetRole()})
        # Save variance reductions to additional_fields
        variance_reductions = self.get_variance_reductions()
        for key, reduction in variance_reductions.items():
            data.additional_fields = data.additional_fields.add_column(
                data=[reduction],
                role={key: StatisticRole()}
            )
        return data.copy(data=new_ds)
