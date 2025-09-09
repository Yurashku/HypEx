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
        result = deepcopy(data)
        for target_feature, pre_target_feature in cuped_features.items():
            cov_xy = result.get_matrix_value('cov', target_feature, pre_target_feature)
            std_y = result[target_feature].std()
            std_x = result[pre_target_feature].std()
            theta = cov_xy / (std_y * std_x)
            pre_target_mean = result[pre_target_feature].mean()
            new_values_ds = result[target_feature] - (result[pre_target_feature] - pre_target_mean) * theta
            result = result.add_column(
                data=new_values_ds,
                role={f"{target_feature}_cuped": TargetRole()}
            )
        return result

    def execute(self, data: ExperimentData) -> ExperimentData:
        new_ds = self.calc(data=data.ds, cuped_features=self.cuped_features)
        return data.copy(data=new_ds)