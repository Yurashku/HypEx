import pandas as pd
import pytest
from pyspark.sql import SparkSession

from hypex.dataset import Dataset
from hypex.dataset.dataset import ExperimentData, SmallDataset
from hypex.dataset.roles import DefaultRole
from hypex.utils import ExperimentDataEnum


@pytest.fixture(scope="session")
def spark_session():
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("hypex-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_small_dataset_from_dict():
    roles = {"a": DefaultRole(), "b": DefaultRole()}
    data = {"data": {"a": [1, 2], "b": ["x", "y"]}, "index": [0, 1]}
    ds = SmallDataset.from_dict(data=data["data"], roles=roles)
    assert isinstance(ds, SmallDataset)
    assert list(ds.columns) == ["a", "b"]


def test_experiment_data_set_value_analysis_tables_accept_dataset():
    roles = {"a": DefaultRole()}
    dataset = Dataset(roles=roles, data=pd.DataFrame({"a": [1, 2]}))
    exp = ExperimentData(dataset)

    exp.set_value(
        ExperimentDataEnum.analysis_tables,
        executor_id="test_table",
        value=dataset,
        role=roles,
    )

    assert "test_table" in exp.analysis_tables
    assert isinstance(exp.analysis_tables["test_table"], SmallDataset)


def test_spark_backend_fillna_sort_and_records(spark_session):
    roles = {"a": DefaultRole(), "b": DefaultRole()}
    data = pd.DataFrame({"a": [2, None, 1], "b": ["x", None, "z"]})
    dataset = Dataset(roles=roles, data=data, session=spark_session)

    filled = dataset.backend.fillna(values={"a": 0, "b": "missing"})
    out = filled.orderBy("a").collect()
    assert out[0]["a"] == 0
    assert any(r["b"] == "missing" for r in out)

    sorted_df = dataset.backend.sort_values(by="a", ascending=True)
    assert sorted_df.columns == ["a", "b"]

    records = dataset.backend.to_records()
    assert len(records) == 3
    assert set(records[0].keys()) == {"a", "b"}
