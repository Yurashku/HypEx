import pytest
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from hypex.dataset import Dataset, FeatureRole, InfoRole, TargetRole, TreatmentRole
from hypex import Matching
from causalinference import CausalModel


@pytest.fixture
def matching_data():
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "post_spends": TargetRole(float),
            "gender": FeatureRole(str),
            "pre_spends": FeatureRole(float),
            "industry": FeatureRole(str),
        },
        data="examples/tutorials/data.csv",
    )
    return data.fillna(method="bfill")


def get_causal_att_and_se(dataset: Dataset):
    df = dataset.data if hasattr(dataset, "data") else pd.DataFrame(dataset)

    Y = df["post_spends"].values
    D = df["treat"].values

    exclude = {"user_id", "treat", "post_spends"}
    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    if categorical_cols:
        df_dummies = pd.get_dummies(df[categorical_cols], drop_first=True)
        X_df = pd.concat([df[numeric_cols], df_dummies], axis=1)
    else:
        X_df = df[numeric_cols]

    X_df = X_df.astype(float)
    X = X_df.values

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cm = CausalModel(Y=Y, D=D, X=X)
    cm.est_via_matching(bias_adj=False)

    return float(cm.estimates["matching"]["att"]), float(cm.estimates["matching"]["att_se"])


def compute_pvalue(effect: float, std_error: float) -> float:
    if std_error == 0:
        return 0.0
    t_stat = abs(effect / std_error)
    return 2 * (1 - norm.cdf(t_stat))


def test_matching_pvalue_consistency_with_causalinference(matching_data):
    distances = ["mahalanobis", "l2"]
    effects = ["att", "atc", "ate"]
    neighbors = [1, 5]

    scenarios = [
        {"distance": d, "effect": e, "n_neighbors": k}
        for d in distances for e in effects for k in neighbors
    ]

    for scenario in tqdm(scenarios, desc="Matching", unit="scenario"):
        distance = scenario["distance"]
        effect = scenario["effect"]
        k = scenario["n_neighbors"]

        causal_att, causal_se = get_causal_att_and_se(matching_data)
        causal_pval = compute_pvalue(causal_att, causal_se)

        matcher = Matching(
            distance=distance,
            n_neighbors=k,
            quality_tests=["t-test", "ks-test"]
        )
        result = matcher.execute(matching_data)

        hypex_pval = result.resume.data.loc[effect.upper(), "P-value"]

        diff = abs(hypex_pval - causal_pval)
        assert diff <= 0.05, (
            f"  distance={distance}, k={k}, effect={effect}:\n"
            f"  hypex: {hypex_pval:.6f}\n"
            f"  causalinference (рассчитан): {causal_pval:.6f}\n"
            f"  разница: {diff:.6f} > 0.05"
        )

        actual_data = result.resume.data
        assert actual_data.index.isin(["ATT", "ATC", "ATE"]).all()
        assert all(
            actual_data.iloc[:, :-1].dtypes.apply(
                lambda x: pd.api.types.is_numeric_dtype(x)
            )
        ), "Есть нечисловые колонки!"

        if hasattr(result, "quality_tests_results"):
            for test_name, test_df in result.quality_tests_results.items():
                assert all(
                    test_df["p-value"].apply(lambda x: isinstance(x, (int, float)))
                ), f"Некорректные p-value в quality_test {test_name}"
