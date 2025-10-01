import pytest
import pandas as pd
from hypex import Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)

@pytest.fixture
def matching_data():
    df = pd.read_csv("examples/tutorials/data.csv")

    df = df.fillna(method="bfill").fillna(0)

    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes

    df["treat"] = df["treat"].clip(0, 1)

    print("Data prepared for HypEx:")
    print(df[["treat", "post_spends", "gender", "industry"]].head())

    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),          
            "post_spends": TargetRole(float),     
            "gender": FeatureRole(int),           
            "pre_spends": FeatureRole(float),     
            "industry": FeatureRole(int),        
            "signup_month": FeatureRole(int),     
            "age": FeatureRole(float),            
        },
        data=df,
    )
    return data

def get_hypex_pvalue(matching_data, distance, k, effect, feature_subset):
    matcher = Matching(distance=distance, n_neighbors=k, quality_tests=["t-test", "ks-test"])
    result = matcher.execute(matching_data)

    assert effect.upper() in result.resume.data.index
    assert "P-value" in result.resume.data.columns
    hypex_pval = result.resume.data.loc[effect.upper(), "P-value"]
    return hypex_pval

@pytest.mark.parametrize("feature_subset", [
    ["pre_spends"],
    ["gender"],
    ["pre_spends", "gender", "industry"]
])
@pytest.mark.parametrize("distance", ["mahalanobis", "l2"])
@pytest.mark.parametrize("effect", ["att", "atc", "ate"])
@pytest.mark.parametrize("k", [1, 3])
def test_matching_pvalue_is_valid(matching_data, feature_subset, distance, effect, k):
    hypex_pval = get_hypex_pvalue(matching_data, distance, k, effect, feature_subset)

    assert isinstance(hypex_pval, (int, float)), f"P-value {hypex_pval} is not a number"
    assert 0 <= hypex_pval <= 1, f"P-value {hypex_pval} is out of range [0, 1]"


    print(f" distance={distance}, k={k}, effect={effect}, features={feature_subset}: p-value={hypex_pval}")