import sys
import traceback

from hypex import ABTest
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole


def run_abtest_tutorial():
    try:
        # Генерация синтетических данных как в туториале
        from hypex.utils.tutorial_data_creation import DataGenerator
        import numpy as np
        import pandas as pd

        gen1 = DataGenerator(
            n_samples=2000,
            distributions={
                "X1": {"type": "normal", "mean": 0, "std": 1},
                "X2": {"type": "bernoulli", "p": 0.5},
                "y0": {"type": "normal", "mean": 5, "std": 1},
            },
            time_correlations={"X1": 0.2, "X2": 0.1, "y0": 0.6},
            effect_size=2.0,
            seed=7
        )
        df = gen1.generate()
        df = df.drop(columns=['y0', 'z', 'U', 'D', 'y1', 'y0_lag_2'])

        from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole
        data = Dataset(
            roles={
                "d": TreatmentRole(),
                "y": TargetRole(),
            },
            data=df,
            default_role=InfoRole()
        )

        # Обычный ABTest
        test = ABTest()
        result = test.execute(data)
        print("ABTest:", result.resume.data.head())

        # ABTest с CUPED
        test_cuped = ABTest(cuped_features={"y": "y0_lag_1"})
        result_cuped = test_cuped.execute(data)
        print("ABTest CUPED:", result_cuped.resume.data.head())

        # ABTest с CUPAC
        test_cupac = ABTest(cupac_features={"y": ["y0_lag_1", "X1_lag", "X2_lag"]}, cupac_model="linear")
        result_cupac = test_cupac.execute(data)
        print("ABTest CUPAC:", result_cupac.resume.data.head())

        print("Все проверки прошли успешно!")
        return 0
    except Exception:
        print("Ошибка при выполнении теста туториала:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_abtest_tutorial())
