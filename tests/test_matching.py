import pytest
import numpy as np
import pandas as pd

from hypex import Matching
from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole, TreatmentRole
from causalinference import CausalModel


@pytest.fixture
def matching_data():
    """
    Dataset fixture для тестов Matching.
    Содержит числовые и категориальные признаки.
    """
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "post_spends": TargetRole(float),
            "pre_spends": FeatureRole(float),
            "age": FeatureRole(float),
            "gender": FeatureRole(str),
            "industry": FeatureRole(str),
        },
        data="examples/tutorials/data.csv",
    )
    return data.fillna(method="bfill")


def run_causal_inference(df, features, outcome="post_spends", treat="treat"):
    if not isinstance(df, pd.DataFrame):
        df = df.data

    X_df = df[features].copy()

    non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_df = pd.get_dummies(X_df, columns=non_numeric, drop_first=True)

    X = X_df.select_dtypes(include=[np.number]).to_numpy()
    y = df[outcome].to_numpy()
    D = df[treat].to_numpy().astype(int)

    cm = CausalModel(Y=y, D=D, X=X)
    cm.est_via_matching(bias_adj=True)

    return {
        "att": float(cm.estimates["matching"].pvalue_att),
        "atc": float(cm.estimates["matching"].pvalue_atc),
        "ate": float(cm.estimates["matching"].pvalue_ate),
    }


@pytest.mark.parametrize("metric", ["att", "atc", "ate"])
@pytest.mark.parametrize("distance", ["mahalanobis", "l2"])
def test_hypex_vs_causal_pvalues(matching_data, metric, distance):
    matcher = Matching(metric=metric, distance=distance)
    result = matcher.execute(matching_data)
    actual_data = result.resume.data

    assert actual_data.index.isin(["ATT", "ATC", "ATE"]).all()
    assert all(
        actual_data.iloc[:, :-1].dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))
    ), "Есть нечисловые колонки!"