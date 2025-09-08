from typing import Any, Sequence
from copy import deepcopy
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole, PreTargetRole
from ..utils.adapter import Adapter
from .abstract import Transformer


class CUPEDTransformer(Transformer):
    def __init__(
        self,
        cuped_features: dict[str, str],
        key: Any = "",
    ):
        """
        Transformer that applies the CUPED adjustment to target features.

        Args:
            cuped_features (dict[str, str]): A mapping {target_feature: pre_target_feature}.
        """
        super().__init__(key=key)
        self.cuped_features = cuped_features

    @staticmethod
    def _inner_function(
        data: Dataset,
        cuped_features: dict[str, str],
    ) -> Dataset:
        # cuped_features: {target_feature: pre_target_feature}
        # Work on a deepcopy so original Dataset isn't mutated by the transformer.
        result_ds = deepcopy(data)
        for target_feature, pre_target_feature in cuped_features.items():
            # Используем Series для вычислений
            target_series = result_ds.data[target_feature]
            covariate_series = result_ds.data[pre_target_feature]
            cov_xy = result_ds.data[[target_feature, pre_target_feature]].cov().loc[target_feature, pre_target_feature]
            std_y = target_series.std()
            std_x = covariate_series.std()
            theta = cov_xy / (std_y * std_x)
            result_ds[target_feature] = target_series - theta * (covariate_series - covariate_series.mean())
            result_ds = result_ds.astype({target_feature: result_ds.roles[target_feature].data_type or float})
        return result_ds

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute transformer using the instance's configured cuped_features.

        The base Transformer.execute calls calc without kwargs which fails for
        CUPEDTransformer because the inner function needs the mapping of features.
        We therefore override execute to provide the configured cuped_features.
        """
        result = data.copy(
            data=self.calc(data=data.ds, cuped_features=self.cuped_features)
        )
        return result