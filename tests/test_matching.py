import pytest
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy import stats

from hypex import Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
    GroupingRole,
)
from causalinference import CausalModel


@pytest.fixture
def matching_data():
    df = pd.read_csv("examples/tutorials/data.csv")
    df = df.bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"].clip(0, 1)

    full_roles = {
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "post_spends": TargetRole(float),
        "gender": FeatureRole(str),
        "pre_spends": FeatureRole(float),
        "industry": FeatureRole(str),
        "signup_month": FeatureRole(int),
        "age": FeatureRole(float),
    }

    data = Dataset(
        roles=full_roles,
        data=df,
    )
    return data


@pytest.fixture
def matching_data_with_group():
    df = pd.read_csv("examples/tutorials/data.csv")
    df = df.bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"].clip(0, 1)

    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "post_spends": TargetRole(float),
            "gender": GroupingRole(int),
            "pre_spends": FeatureRole(float),
            "industry": FeatureRole(int),
            "signup_month": FeatureRole(int),
            "age": FeatureRole(float),
        },
        data=df,
    )
    return data


def get_causalinference_pvalue(data_df, feature_subset, k, effect="att"):
    Y = data_df['post_spends'].values
    D = data_df['treat'].values
    X = data_df[feature_subset].astype(float).values

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    model = CausalModel(Y, D, X)
    model.est_via_matching(matches=k, bias_adj=True)
    results = model.estimates['matching']
    
    effect_key = effect.lower()
    se_key = f'{effect_key}_se'
    
    if effect_key not in results or se_key not in results:
        return np.nan 

    effect_estimate = results[effect_key]
    se_estimate = results[se_key]
    
    if se_estimate == 0:
        return 0.0 if abs(effect_estimate) > 1e-6 else 1.0
    
    t_stat = effect_estimate / se_estimate
    df = len(data_df) - X.shape[1] 
    p_val = stats.t.sf(abs(t_stat), df) * 2
    return p_val


def calculate_relative_difference(v1, v2, tolerance=1e-9):
    if pd.isna(v1) or pd.isna(v2):
        return np.nan
    if abs(v1) < tolerance and abs(v2) < tolerance:
        return 0.0
    denominator = (abs(v1) + abs(v2)) / 2
    if denominator < tolerance:
        return abs(v1 - v2)
    return abs(v1 - v2) / denominator


def test_feature_subsets_vs_causalinference(matching_data):
    feature_subsets = [["pre_spends"], ["age", "gender"], ["pre_spends", "gender", "industry"]]
    distances = ["mahalanobis", "l2"]
    effects = ["att"] 
    k_values = [1, 5]

    total_tests = len(feature_subsets) * len(distances) * len(effects) * len(k_values)
    pbar = tqdm(total=total_tests, desc="Feature subset comparison", unit="test")

    param_combinations = product(feature_subsets, distances, effects, k_values)

    for feature_subset, distance, effect, k in param_combinations:
        current_roles = {
            "user_id": InfoRole(), "treat": TreatmentRole(), "post_spends": TargetRole()
        }
        for feature in feature_subset:
            current_roles[feature] = FeatureRole()
        data_subset = Dataset(roles=current_roles, data=matching_data.data)
        
        matcher_hypex = Matching(distance=distance, n_neighbors=k, quality_tests=["t-test"])
        result_hypex = matcher_hypex.execute(data_subset)
        pval_hypex = result_hypex.resume.data.loc[effect.upper(), "P-value"]

        if distance == "mahalanobis":
            pval_causal = get_causalinference_pvalue(matching_data.data, feature_subset, k, effect=effect)
            rel_diff = calculate_relative_difference(pval_hypex, pval_causal)
        else:
            pval_causal = np.nan
            rel_diff = np.nan

        pbar.set_postfix({
            'dist': distance, 'k': k, 'hypex': f"{pval_hypex:.4f}",
            'causal': f"{pval_causal:.4f}", 'diff': f"{rel_diff:.2%}" if not pd.isna(rel_diff) else "N/A"
        })
        pbar.update(1)

        assert 0 <= pval_hypex <= 1
        if not pd.isna(pval_causal):
            assert 0 <= pval_causal <= 1
            assert rel_diff <= 0.05, f"Relative difference is too high: {rel_diff:.2%} for features {feature_subset}"

    pbar.close()


def test_matching_group_match(matching_data_with_group):
    pbar = tqdm(total=1, desc="Group match test", unit="test")
    matcher = Matching(group_match=True, n_neighbors=1, quality_tests=["t-test", "ks-test"])
    result = matcher.execute(matching_data_with_group)
    pval = result.resume.data.loc["ATT", "P-value"]
    pbar.set_postfix({'pval': f"{pval:.4f}"})
    pbar.update(1)
    assert 0 <= pval <= 1
    pbar.close()


def test_matching_quality_tests_all(matching_data):
    pbar = tqdm(total=1, desc="Quality tests", unit="test")
    matcher = Matching(n_neighbors=1, quality_tests=["t-test", "ks-test", "chi2-test"])
    result = matcher.execute(matching_data)
    pval = result.resume.data.loc["ATT", "P-value"]
    pbar.set_postfix({'pval': f"{pval:.4f}"})
    pbar.update(1)
    assert 0 <= pval <= 1
    pbar.close()


def test_matching_custom_weights_functionality(matching_data):
    pbar = tqdm(total=1, desc="Custom weights functionality test", unit="test")
    
    features = ["pre_spends", "gender", "industry", "signup_month", "age"]
    weights = {
        "pre_spends": 0.3, "gender": 0.2, "industry": 0.3,
        "signup_month": 0.1, "age": 0.1,
    }
    k = 1
    
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    matcher_hypex = Matching(n_neighbors=k, weights=weights, quality_tests=["t-test"])
    result_hypex = matcher_hypex.execute(matching_data)
    pval_hypex = result_hypex.resume.data.loc["ATT", "P-value"]
    
    pval_causal = get_causalinference_pvalue(matching_data.data, features, k, effect="att")

    rel_diff = calculate_relative_difference(pval_hypex, pval_causal)
    
    pbar.set_postfix({
        'hypex': f"{pval_hypex:.4f}", 
        'causal_mahalanobis': f"{pval_causal:.4f}", 
        'diff': f"{rel_diff:.2%}"
    })
    pbar.update(1)
    
    assert 0 <= pval_hypex <= 1
    if not pd.isna(pval_causal):
        assert 0 <= pval_causal <= 1
        assert rel_diff <= 0.05, f"Relative difference is too high: {rel_diff:.2%} (Weighted L2 vs Mahalanobis)"
        
    pbar.close()