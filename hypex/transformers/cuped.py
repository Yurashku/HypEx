from typing import Any, Sequence
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
        Transformer для применения метода CUPED.

        Args:
            cuped_features (dict[str, str]): Словарь {target_feature: pre_target_feature}.
        """
        super().__init__(key=key)
        self.cuped_features = cuped_features

    @staticmethod
    def _inner_function(
        data: Dataset,
        cuped_features: dict[str, str],
    ) -> Dataset:
        # cuped_features: {target_col: covariate_col}
        for target_col, covariate_col in cuped_features.items():
            # Используем Series для вычислений
            target_series = data.data[target_col]
            covariate_series = data.data[covariate_col]
            cov_xy = data.data[[target_col, covariate_col]].cov().loc[target_col, covariate_col]
            std_y = target_series.std()
            std_x = covariate_series.std()
            theta = cov_xy / (std_y * std_x)
            data[target_col] = target_series - theta * (covariate_series - covariate_series.mean())
            data = data.astype({target_col: data.roles[target_col].data_type or float})
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        result = data.copy(
            data=self.calc(
                data=data.ds,
                cuped_features=self.cuped_features,
            )
        )
        return result