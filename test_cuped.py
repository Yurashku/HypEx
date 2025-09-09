import sys
import traceback

from hypex import ABTest
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole


def run_abtest_cuped():
    try:
        # Генерация синтетических данных
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

        data = Dataset(
            roles={
                "d": TreatmentRole(),
                "y": TargetRole(),
            },
            data=df,
            default_role=InfoRole()
        )

        # ABTest с CUPED
        test_cuped = ABTest(cuped_features={"y": "y0_lag_1"})
        result_cuped = test_cuped.execute(data)

        print("ABTest CUPED resume:")
        print(result_cuped.resume)

        print("\nVariance Reduction Report:")
        print(result_cuped.variance_reduction_report())

        print("\nВсе проверки прошли успешно!")
        return 0
    except Exception:
        print("Ошибка при выполнении теста CUPED:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_abtest_cuped())
