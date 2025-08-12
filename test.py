import pandas as pd
from hypex.dataset.dataset import ExperimentData, ExperimentDataEnum
from hypex.dataset import Dataset, InfoRole, TargetRole, TreatmentRole

initial_data = pd.DataFrame({
    "target_col": [1.0, 2.0, 3.0],
    "covariate_col": [0.5, 1.5, 2.5]
})

data = Dataset(
    roles={
        "covariate_col": TargetRole(),
        "target_col": TargetRole(),
    },
    data=initial_data,
    default_role=InfoRole()
)

target_col = 'target_col'
covariate_col = 'covariate_col'

data = ExperimentData(data=data)

cov_xy = data.ds[[target_col, covariate_col]].cov().loc[target_col, covariate_col]
std_y = data.ds[target_col].std()
std_x = data.ds[covariate_col].std()

# Вычисляем theta
theta = cov_xy / (std_y * std_x)

# Применяем CUPED data.ds[target_col] - 
adjusted_values = theta.get_values(0,0) * (data.ds[covariate_col] - data.ds[covariate_col].mean())
print(adjusted_values)