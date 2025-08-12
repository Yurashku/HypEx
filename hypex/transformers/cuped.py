from ..dataset import ExperimentData, TargetRole

from ..utils import (
    ExperimentDataEnum,
)


class CUPEDTransformer:
    def __init__(self, cuped_feature: str | dict[str, str]):
        """
        Transformer для применения метода CUPED.

        Args:
            cuped_feature (str | dict[str, str]): Название ковариаты для CUPED (строка) 
                                                  или словарь {target_feature: pre_target_feature}.
        """
        self.cuped_feature = cuped_feature
        self.cuped_features = None  # Словарь {target_feature: pre_target_feature}, будет обработан позже

    def _prepare_cuped_features(self, data: ExperimentData):
        """
        Подготавливает словарь {target_feature: pre_target_feature}.

        Args:
            data (Dataset): Данные эксперимента.

        Raises:
            ValueError: Если не удается определить целевые метрики или ковариаты.
        """
        if isinstance(self.cuped_feature, str):
            # Если передана строка, ищем единственную колонку TargetRole
            target_columns = data.field_search(TargetRole())
            if len(target_columns) != 1:
                raise ValueError(
                    "Для использования CUPED с одной ковариатой должна быть ровно одна TargetRole колонка."
                )
            self.cuped_features = {target_columns[0]: self.cuped_feature}
        elif isinstance(self.cuped_feature, dict):
            # Если передан словарь, используем его напрямую
            self.cuped_features = self.cuped_feature
        else:
            raise ValueError(
                "cuped_feature должен быть строкой или словарем {target_feature: pre_target_feature}."
            )

    @staticmethod
    def _apply_cuped(data: ExperimentData, target_col: str, covariate_col: str) -> None:
        """
        Применяет метод CUPED к одной целевой метрике.

        Args:
            data (Dataset): Данные эксперимента.
            target_col (str): Название столбца с метрикой.
            covariate_col (str): Название столбца с ковариатой.
        """
        #TODO: Измнеить ExperimentData чтобы можно было выполнять в нём
        cov_xy = data._data[[target_col, covariate_col]].cov().loc[target_col, covariate_col].iloc[0, 0]
        std_y = data._data[target_col].std()
        std_x = data._data[covariate_col].std()

        # Вычисляем theta
        theta = cov_xy / (std_y * std_x)
        theta = theta.get_values(0, 0)


        # Применяем CUPED
        adjusted_values = data._data.data[target_col] - theta * (data._data.data[covariate_col] - data._data.data[covariate_col].mean())
        data._data.data[target_col] = adjusted_values

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Выполняет трансформацию данных с использованием CUPED.

        Args:
            data (Dataset): Данные эксперимента.

        Returns:
            Dataset: Трансформированные данные.
        """
        # Подготавливаем словарь {target_feature: pre_target_feature}, если он еще не подготовлен
        if self.cuped_features is None:
            self._prepare_cuped_features(data)

        # Применяем CUPED для каждой пары target_feature -> pre_target_feature
        for target_col, covariate_col in self.cuped_features.items():
            self._apply_cuped(data, target_col, covariate_col)

        return data