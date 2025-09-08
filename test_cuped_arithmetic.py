import pandas as pd
from hypex.dataset.dataset import Dataset, InfoRole

def test_cuped_arithmetic():
    # Дублируем структуру датасета
    df = pd.DataFrame({
        'y': [1.0, 2.0, 3.0, 4.0],
        'y0_lag_1': [0.5, 1.5, 2.5, 3.5]
    })
    roles = {'y': InfoRole(), 'y0_lag_1': InfoRole()}
    ds = Dataset(data=df, roles=roles)

    cov_xy = ds.get_matrix_value('cov', 'y', 'y0_lag_1')
    std_y = ds['y'].std()
    std_x = ds['y0_lag_1'].std()
    theta = cov_xy / (std_y * std_x)
    pre_target_mean = ds['y0_lag_1'].mean()

    print('ds["y0_lag_1"]:')
    print(ds['y0_lag_1'].data)
    print('pre_target_mean:')
    print(pre_target_mean)

    # Этап 1: ds['y0_lag_1'] - pre_target_mean
    try:
        step1 = ds['y0_lag_1'] - pre_target_mean
        print('step1 (ds["y0_lag_1"] - pre_target_mean):')
        print(step1.data)
        print('step1 roles:', step1.roles)
    except Exception as e:
        print('Ошибка на этапе 1:', type(e), e)
        return

    # Этап 2: theta * step1
    try:
        step2 = theta * step1
        print('step2 (theta * step1):')
        print(step2.data)
    except Exception as e:
        print('Ошибка на этапе 2:', type(e), e)
        return

    # Этап 3: ds['y'] - step2
    try:
        step3 = ds['y'] - step2
        print('step3 (ds["y"] - step2):')
        print(step3.data)
    except Exception as e:
        print('Ошибка на этапе 3:', type(e), e)
        return

if __name__ == '__main__':
    test_cuped_arithmetic()
