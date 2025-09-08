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
            # Все вычисления через Dataset
            cov_xy = result_ds.get_matrix_value('cov', target_feature, pre_target_feature)
            std_y = result_ds[target_feature].std()
            std_x = result_ds[pre_target_feature].std()
            theta = cov_xy / (std_y * std_x)
            # mean для pre_target_feature
            pre_target_mean = result_ds[pre_target_feature].mean()
            # CUPED корректировка: все операции между Dataset
            new_values_ds = result_ds[target_feature] - theta * (result_ds[pre_target_feature] - pre_target_mean)
            # Присваиваем скорректированные значения
            result_ds[target_feature] = new_values_ds
            result_ds = result_ds.astype({target_feature: result_ds.roles[target_feature].data_type or float})
        return result_ds

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(data=data.ds, cuped_features=self.cuped_features)
        )
        return result