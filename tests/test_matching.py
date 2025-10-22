# python3 -m pytest -s -v -rA tests/test_matching.py
import pytest
import pandas as pd
import numpy as np
from itertools import product

from hypex import Matching
from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole, TreatmentRole, GroupingRole

from causalinference import CausalModel
from causalinference.utils import tools


@pytest.fixture
def matching_data():
    df = pd.read_csv("examples/tutorials/synthetic_data.csv").bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"].clip(0, 1)

    roles = {
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "pre_spends": TargetRole(float),
        "gender": FeatureRole(str),
        "post_spends": FeatureRole(float),
        "industry": FeatureRole(str),
        "signup_month": FeatureRole(int),
        "age": FeatureRole(float),
    }
    return Dataset(roles=roles, data=df)


@pytest.fixture
def matching_data_with_group():
    df = pd.read_csv("examples/tutorials/synthetic_data.csv").bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"].clip(0, 1)

    roles = {
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "pre_spends": TargetRole(float),
        "gender": GroupingRole(int),
        "post_spends": FeatureRole(float),
        "industry": FeatureRole(int),
        "signup_month": FeatureRole(int),
        "age": FeatureRole(float),
    }
    return Dataset(roles=roles, data=df)


feature_subsets = [
    ["post_spends"],
    ["age", "gender"],
    ["post_spends", "gender", "industry"],
]

distances = ["mahalanobis"]
k_values = [1, 3, 5]
scenarios = [
    "default",
    "group_match_gender",
    "group_match_industry",
    "custom_weights_1",
    "custom_weights_2",
]

param_combinations = list(product(feature_subsets, distances, k_values, scenarios))

custom_weights_dict = {
    "custom_weights_1": {"post_spends": 0.5, "gender": 0.3, "industry": 0.2},
    "custom_weights_2": {"post_spends": 0.2, "gender": 0.4, "industry": 0.4},
}


def get_causalinference_pvalue(data_df, features, k, effect="ate"):
    Y = data_df["pre_spends"].values
    D = data_df["treat"].values
    X = data_df[features].astype(float).values
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    model = CausalModel(Y, D, X)
    model.est_via_matching(matches=k, bias_adj=True)

    try:
        est = model.estimates.get("matching", model.estimates)
        coef = est[effect.lower()]
        se = est[f"{effect.lower()}_se"]
    except Exception:
        return np.nan

    if se == 0 or np.isnan(se):
        return 0.0 if abs(coef) > 1e-6 else 1.0

    entries = tools.gen_reg_entries(effect.upper(), float(coef), float(se))
    return float(entries[4])


def calculate_relative_ratio(p1, p2, eps=1e-9):
    return (p1 + eps) / (p2 + eps)


@pytest.mark.parametrize(
    "features, distance, k, scenario",
    param_combinations,
    ids=lambda val: (
        f"features={','.join(val[0])}_dist={val[1]}_k={val[2]}_scenario={val[3]}"
        if isinstance(val, tuple) else str(val)
    ),
)


def test_matching_scenario(matching_data, matching_data_with_group, features, distance, k, scenario):
    print(f"\n=== Scenario: {scenario}, k={k}, features={features} ===")

    if distance != "mahalanobis":
        pytest.skip(f"Skipping distance={distance}")

    if "group_match" in scenario:
        data_subset = matching_data_with_group
        group_match_flag = True
        scenario_weights = None
        if "gender" in scenario:
            data_subset.roles["gender"] = GroupingRole(int)
        elif "industry" in scenario:
            data_subset.roles["industry"] = GroupingRole(int)
    else:
        current_roles = {
            "user_id": InfoRole(),
            "treat": TreatmentRole(),
            "pre_spends": TargetRole(),
        }
        for f in features:
            current_roles[f] = FeatureRole()
        data_subset = Dataset(roles=current_roles, data=matching_data.data)
        group_match_flag = False
        scenario_weights = custom_weights_dict.get(scenario, None)

    df = data_subset.data
    if len(df) < 30 or df["treat"].nunique() < 2:
        pytest.skip(f"Skipping scenario={scenario}, insufficient data")

    matcher_hypex = Matching(
        distance=distance,
        n_neighbors=k,
        weights=scenario_weights,
        group_match=group_match_flag,
        quality_tests=["t-test", "ks-test", "chi2-test"],
    )

    result_hypex = matcher_hypex.execute(data_subset)
    pval_hypex = result_hypex.resume.data.loc["ATE", "P-value"]

    pval_causal = get_causalinference_pvalue(df, features, k)

    print(f"Hypex p-value: {pval_hypex:.6f}")
    print(f"Causal p-value: {pval_causal:.6f}")

    rel_ratio = calculate_relative_ratio(pval_hypex, pval_causal)
    print(f"Relative ratio: {rel_ratio:.4f}")

    assert 0.95 <= rel_ratio <= 1.05, (
        f"p-value relative ratio out of range: {rel_ratio:.2f} "
        f"for features {features}, scenario={scenario}, k={k}"
    )
