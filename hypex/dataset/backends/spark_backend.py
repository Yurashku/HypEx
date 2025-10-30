import os
import sys

from functools import reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Sequence,
    Sized,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
import pyspark.sql as spark
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SparkSession, Window, functions as F, types as T
from pyspark.sql.functions import lit, monotonically_increasing_id

from ...utils import FromDictTypes, MergeOnError, ScalarType
from .abstract import DatasetBackendCalc, DatasetBackendNavigation


class SparkNavigation(DatasetBackendNavigation):
    @staticmethod
    def _read_file(
        filename: Union[str, Path], session: SparkSession
    ) -> spark.DataFrame:
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return (
                session.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "true")  # TODO: find faster solution in future
                .load(filename)
            )
        elif file_extension == ".parquet":
            return session.read.parquet(filename)
        else:
            try:
                return session.read.table(filename)
            except:
                raise ValueError(f"Unsupported file extension {file_extension}")

    @staticmethod
    def _get_spark_session(
        app_name: str = "HypEx",
        python_path: Optional[str] = None,
        dynamic_allocation: bool = True,
        mode: Optional[str] = None,
    ):
        if python_path is None:
            python_path = sys.executable

        os.environ["PYSPARK_PYTHON"] = python_path
        os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

        if mode == "local":
            conf = (
                SparkConf()
                .setAppName(app_name)
                .setMaster("local[*]")
                .set("spark.driver.memory", "6g")
                .set("spark.executor.memory", "6g")
            )
        else:
            conf = (
                SparkConf()
                .setAppName(app_name)
                .
                # setMaster("yarn").
                set("spark.executor.cores", "8")
                .set("spark.executor.memory", "8g")
                .set("spark.executor.memoryOverhead", "8g")
                .set("spark.driver.cores", "12")
                .set("spark.driver.memory", "16g")
                .set("spark.driver.maxResultSize", "32g")
                .set("spark.shuffle.service.enabled", "true")
                .set("spark.dynamicAllocation.enabled", dynamic_allocation)
                .set("spark.dynamicAllocation.initialExecutors", "6")
                .set("spark.dynamicAllocation.maxExecutors", "32")
                .set("spark.dynamicAllocation.executorIdleTimeout", "120s")
                .set("spark.dynamicAllocation.cachedExecutorIdleTimeout", "600s")
                .set("spark.port.maxRetries", "150")
            )

        return SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()

    def __init__(
        self,
        data: Optional[Union[spark.DataFrame, pd.DataFrame, dict, str]] = None,
        session: SparkSession = None,
    ):
        if session is None:
            if isinstance(data, spark.DataFrame):
                self.session = data.spark
            else:
                self.session = self._get_spark_session()
        else:
            if isinstance(session, SparkSession):
                self.session = session
            else:
                raise TypeError("Session must be an instance of SparkSession")

        if isinstance(data, dict):
            if "index" in data.keys():
                data = pd.DataFrame(data=data["data"], index=data["index"])
            else:
                data = pd.DataFrame(data=data["data"])

        if isinstance(data, spark.DataFrame):
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.data = SparkSession.createDataFrame(data)
        elif isinstance(data, str):
            self.data = self._read_file(data, self.session)
        else:
            self.data = SparkSession.emptyDataFrame

    def __getitem__(self, item):
        if isinstance(item, (str, list, tuple)):
            columns = [item] if isinstance(item, str) else list(item)
            return self.data.select(*columns)

        if isinstance(item, slice):
            start = item.start or 0
            stop = item.stop
            if stop is None:
                raise ValueError("Slice stop must be defined for Spark navigation")
            step = item.step or 1
            if step <= 0:
                raise ValueError("Slice step must be positive for Spark navigation")

            indexed = self._with_row_index(self.data)
            condition = (F.col("__row_id") >= start) & (F.col("__row_id") < stop)
            filtered = indexed.filter(condition & ((F.col("__row_id") - start) % step == 0))
            return filtered.drop("__row_id")

        if isinstance(item, int):
            idx = item
            if idx < 0:
                length = self.__len__()
                idx = length + idx
            row = self._with_row_index(self.data).filter(F.col("__row_id") == idx).drop("__row_id")
            collected = row.limit(1).collect()
            if not collected:
                raise IndexError("Spark dataset index out of range")
            return collected[0]

        raise KeyError("Unsupported index type for SparkDataset")

    def __len__(self):
        return 0 if self.data is None else self.data.count()

    @staticmethod
    def __magic_determine_other(other) -> Any:
        if isinstance(other, SparkDataset):
            return other.data
        if isinstance(other, spark.DataFrame):
            return other
        return other

    @staticmethod
    def _with_row_index(df: spark.DataFrame, column_name: str = "__row_id") -> spark.DataFrame:
        schema = df.schema.add(T.StructField(column_name, T.LongType(), False))
        rdd = df.rdd.zipWithIndex().map(lambda row_idx: row_idx[0] + (row_idx[1],))
        return df.sql_ctx.createDataFrame(rdd, schema)

    def _binary_operation(
        self,
        other: Any,
        operator: Callable[[F.Column, F.Column], F.Column],
        reverse: bool = False,
    ) -> spark.DataFrame:
        if self.data is None:
            empty_schema = T.StructType([])
            empty_rdd = self.session.sparkContext.emptyRDD()
            return self.session.createDataFrame(empty_rdd, empty_schema)

        other_value = self.__magic_determine_other(other)
        if isinstance(other_value, spark.DataFrame):
            left = self._with_row_index(self.data).alias("left")
            right = self._with_row_index(other_value).alias("right")
            joined = left.join(right, "__row_id", "inner")
            columns = self.data.columns
            expressions = []
            for column in columns:
                left_col = F.col(f"left.{column}")
                right_col = F.col(f"right.{column}")
                lhs, rhs = (right_col, left_col) if reverse else (left_col, right_col)
                expressions.append(operator(lhs, rhs).alias(column))
            return joined.select(*expressions)

        literal = F.lit(other_value)
        expressions = []
        for column in self.data.columns:
            column_expr = F.col(column)
            lhs, rhs = (literal, column_expr) if reverse else (column_expr, literal)
            expressions.append(operator(lhs, rhs).alias(column))
        return self.data.select(*expressions)

    # comparison operators:
    def __eq__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left == right)

    def __ne__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left != right)

    def __le__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left <= right)

    def __lt__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left < right)

    def __ge__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left >= right)

    def __gt__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left > right)

    # Unary operations:
    def __pos__(self) -> Any:
        return self.data

    def __neg__(self) -> Any:
        return self.data.select(
            *[(F.col(column) * -1).alias(column) for column in self.data.columns]
        )

    def __abs__(self) -> Any:
        return self.data.select(
            *[F.abs(F.col(column)).alias(column) for column in self.data.columns]
        )

    def __invert__(self) -> Any:
        def invert_column(column_name: str) -> F.Column:
            column = F.col(column_name)
            return F.when(column.isNull(), None).otherwise(~column)

        return self.data.select(
            *[invert_column(column).alias(column) for column in self.data.columns]
        )

    def __round__(self, ndigits: int = 0) -> Any:
        return self.data.select(
            *(F.round(F.col(column), ndigits).alias(column) for column in self.data.columns)
        )

    # Binary operations:
    def __add__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left + right)

    def __sub__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left - right)

    def __mul__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left * right)

    def __floordiv__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left // right)

    def __div__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left / right)

    def __truediv__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left / right)

    def __mod__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left % right)

    def __pow__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left ** right)

    def __and__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left & right)

    def __or__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left | right)

    # Right arithmetic operators:
    def __radd__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left + right, reverse=True)

    def __rsub__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left - right, reverse=True)

    def __rmul__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left * right, reverse=True)

    def __rfloordiv__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left // right, reverse=True)

    def __rdiv__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left / right, reverse=True)

    def __rtruediv__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left / right, reverse=True)

    def __rmod__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left % right, reverse=True)

    def __rpow__(self, other) -> Any:
        return self._binary_operation(other, lambda left, right: left ** right, reverse=True)

    def __repr__(self):
        if self.data is None:
            return "SparkDataset(empty)"
        try:
            return self.data._jdf.showString(20, 20, False)
        except Exception:
            return f"SparkDataset({self.data.schema.simpleString()})"

    def _repr_html_(self):
        if self.data is None:
            return "<div>Empty SparkDataset</div>"
        preview = self.data.limit(20).toPandas()
        return preview.to_html()

    def create_empty(
        self,
        index: Optional[Iterable] = None,
        columns: Optional[Iterable[str]] = None,
    ):
        columns = list(columns or [])
        schema = T.StructType(
            [T.StructField(column, T.StringType(), True) for column in columns]
        )
        empty_rdd = self.session.sparkContext.emptyRDD()
        self.data = self.session.createDataFrame(empty_rdd, schema)
        if index is not None:
            index_rows = [(value,) for value in index]
            index_df = self.session.createDataFrame(index_rows, ["index"])
            self.data = index_df.join(self.data, how="cross") if columns else index_df
        return self

    @property
    def index(self):
        if self.data is None:
            return pd.Index([])
        if "index" in self.data.columns:
            return pd.Index(self.data.select("index").toPandas()["index"])
        if "__index__" in self.data.columns:
            return pd.Index(self.data.select("__index__").toPandas()["__index__"])
        return pd.Index(
            [row["__row_id"] for row in self._with_row_index(self.data).select("__row_id").collect()]
        )

    @property
    def columns(self):
        return self.data.columns

    @property
    def shape(self):
        return (self.__len__(), len(self.data.columns))

    def _get_column_index(
        self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        if isinstance(column_name, str):
            return self.data.columns.index(column_name)
        return [self.data.columns.index(name) for name in column_name]

    def get_column_type(
        self, column_name: Union[List[str], str]
    ) -> Optional[Union[Dict[str, type], type]]:
        dtypes = {}
        for k, v in self.data.select(column_name).dtypes:
            if pd.api.types.is_integer_dtype(v):
                dtypes[k] = int
            elif pd.api.types.is_float_dtype(v):
                dtypes[k] = float
            # elif pd.api.types.is_object_dtype(v) and pd.api.types.is_list_like(
            #     self.data[column_name].iloc[0]
            # ):
            #     dtypes[k] = object
            elif (
                pd.api.types.is_string_dtype(v)
                or pd.api.types.is_object_dtype(v)
                or v == "category"
            ):
                dtypes[k] = str
            elif pd.api.types.is_bool_dtype(v):
                dtypes[k] = bool
        if isinstance(column_name, list):
            return dtypes
        else:
            if column_name in dtypes:
                return dtypes[column_name]
        return None

    def astype(
        self, dtype: Dict[str, type], errors: Literal["raise", "ignore"] = "raise"
    ) -> spark.DataFrame:
        return self.data.astype(dtype)

    def update_column_type(self, dtype: Dict[str, type]):
        if len(dtype) > 0:
            self.data = self.astype(dtype)
        return self

    def add_column(self, data: Union[spark.DataFrame, List], name: Optional[str] = None, index = None) -> spark.DataFrame:
        def _add_columns_from_dataframe(df: spark.DataFrame, new_df: spark.DataFrame) -> spark.DataFrame:
            if df.count() != new_df.count():
                raise ValueError(
                    f"Row count mismatch: original DF has {df.count()} rows, new DF has {new_df.count()} rows")

            df_with_index = df.withColumn("__join_id", monotonically_increasing_id())
            new_df_with_index = new_df.withColumn("__join_id", monotonically_increasing_id())

            result = df_with_index.join(new_df_with_index, "__join_id", "inner")

            return result.drop("__join_id")

        def _add_columns_from_list(df: spark.DataFrame, data_list: List, column_names: str) -> spark.DataFrame:
            if len(data_list) != df.count():
                raise ValueError(f"Data length {len(data_list)} doesn't match DataFrame row count {df.count()}")

            original_rdd = df.rdd
            zipped_rdd = original_rdd.zip(spark.sparkContext.parallelize(data_list))
            new_df = zipped_rdd.map(lambda x: x[0] + (x[1],)).toDF(df.columns + [column_names])

            return new_df

        if isinstance(data, spark.DataFrame):
            return _add_columns_from_dataframe(self.data, data)
        elif isinstance(data, list):
            return _add_columns_from_list(self.data, data, name)
        else:
            raise ValueError("new_data must be Spark DataFrame, list of values, or list of lists")

    def append(
        self, other, reset_index: bool = False, axis: int = 0
    ) -> spark.DataFrame:
        if axis != 0:
            raise NotImplementedError("Spark backend only supports axis=0 for append")
        frames = [self.data]
        for dataset in other:
            if isinstance(dataset, SparkDataset):
                frames.append(dataset.data)
            elif isinstance(dataset, spark.DataFrame):
                frames.append(dataset)
            else:
                raise TypeError("Unsupported dataset type for append")
        result = frames[0]
        for frame in frames[1:]:
            result = result.unionByName(frame, allowMissingColumns=True)
        if reset_index:
            result = self._with_row_index(result, "index").drop("index")
        return result

    def from_dict(
        self, data: FromDictTypes, index: Optional[Union[Iterable, Sized]] = None
    ):
        if isinstance(data, dict):
            pdf = pd.DataFrame().from_records(data, columns=list(data.keys()))
        else:
            pdf = pd.DataFrame().from_records(data)
        if index is not None:
            pdf.index = index
        self.data = self.session.createDataFrame(pdf.reset_index())
        return self

    def to_dict(self) -> dict[str, Any]:
        pdf = self.data.toPandas()
        return {"data": {column: pdf[column].tolist() for column in pdf.columns}, "index": list(pdf.index)}

    def to_records(self) -> list[dict]:
        return self.data.toPandas().to_dict(orient="records")

    def loc(self, items: Iterable) -> Iterable:
        pdf = self.data.toPandas()
        data = pdf.loc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def iloc(self, items: Iterable) -> Iterable:
        pdf = self.data.toPandas()
        data = pdf.iloc[items]
        if not isinstance(data, Iterable) or isinstance(data, str):
            data = [data]
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


class SparkDataset(SparkNavigation, DatasetBackendCalc):
    def __init__(
        self,
        data: Optional[Union[spark.DataFrame, pd.DataFrame, dict, str]] = None,
        session: SparkSession = None,
    ):
        super().__init__(data, session)

    def _numeric_columns(self, include_boolean: bool = True) -> List[str]:
        if self.data is None:
            return []
        numeric_types = {
            "byte",
            "short",
            "int",
            "bigint",
            "long",
            "float",
            "double",
            "tinyint",
            "smallint",
        }
        numeric_columns: List[str] = []
        for name, dtype in self.data.dtypes:
            dtype_lower = dtype.lower()
            if (
                dtype_lower in numeric_types
                or dtype_lower.startswith("decimal")
                or (include_boolean and dtype_lower == "boolean")
            ):
                numeric_columns.append(name)
        return numeric_columns

    def _collect_aggregation(self, expressions: List[F.Column], row_label: str) -> pd.DataFrame:
        aggregated = self.data.agg(*expressions)
        as_dict = aggregated.limit(1).toPandas()
        as_dict.insert(0, "aggregation", row_label)
        return as_dict

    @staticmethod
    def _ensure_pandas_frame(
        value: Union[spark.DataFrame, pd.DataFrame, float, int],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(value, spark.DataFrame):
            return value.toPandas()
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, (int, float)):
            column_names = columns or [0]
            return pd.DataFrame([[value]], columns=column_names)
        raise TypeError("Unsupported value type for conversion to pandas DataFrame")

    @staticmethod
    def _convert_agg_result(result):
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return result
            if result.shape[0] == 1 and result.shape[1] == 1:
                value = result.iat[0, 0]
                return float(value) if value is not None else float("nan")
            return result
        if isinstance(result, spark.DataFrame):
            columns = result.columns
            if len(columns) == 0:
                return result
            if len(columns) == 1:
                first_row = result.select(columns[0]).head(1)
                if not first_row:
                    return float("nan")
                value = first_row[0][0]
                return float(value) if value is not None else float("nan")
            return result
        return result

    def get_values(
        self,
        row: Optional[str] = None,
        column: Optional[str] = None,
    ) -> Any:
        pdf = self.data.toPandas()
        if (column is not None) and (row is not None):
            return pdf.loc[row, column]
        elif column is not None:
            result = pdf.loc[:, column]
        elif row is not None:
            result = pdf.loc[row, :]
        else:
            result = pdf
        return result.values.tolist()

    def iget_values(
        self,
        row: Optional[int] = None,
        column: Optional[int] = None,
    ) -> Any:
        pdf = self.data.toPandas()
        if (column is not None) and (row is not None):
            return pdf.iloc[row, column]
        elif column is not None:
            result = pdf.iloc[:, column]
        elif row is not None:
            result = pdf.iloc[row, :]
        else:
            result = pdf
        return result.values.tolist()

    def apply(self, func: Callable, **kwargs) -> spark.DataFrame:
        column_name = kwargs.pop("column_name", None)
        pdf = self.data.toPandas()
        result = pdf.apply(func, **kwargs)
        if not isinstance(result, pd.DataFrame):
            column_name = column_name or "result"
            result = result.to_frame(name=column_name)
        return self.session.createDataFrame(result.reset_index(drop=True))

    def map(self, func: Callable, **kwargs) -> spark.DataFrame:
        pdf = self.data.toPandas()
        result = pdf.map(func, **kwargs)
        return self.session.createDataFrame(result.reset_index(drop=True))

    def is_empty(self) -> bool:
        return self.data.rdd.isEmpty()

    def unique(self):
        return {
            column: [row[0] for row in self.data.select(column).distinct().collect()]
            for column in self.data.columns
        }

    def nunique(self, dropna: bool = True):
        counts = {}
        for column in self.data.columns:
            col_expr = F.col(column)
            data = self.data.select(column)
            if dropna:
                data = data.filter(col_expr.isNotNull())
            counts[column] = data.distinct().count()
        return counts

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> list[tuple]:
        columns = [by] if isinstance(by, str) else list(by)
        grouped = self.data.groupBy(*columns, **kwargs)
        return grouped

    def agg(self, func: Union[str, list], **kwargs) -> Union[spark.DataFrame, float]:
        functions = func if isinstance(func, list) else [func]
        rows: List[pd.DataFrame] = []
        for func_name in functions:
            func_lower = func_name.lower()
            columns = (
                self.data.columns if func_lower == "count" else self._numeric_columns()
            )
            if not columns:
                continue
            expressions: List[F.Column] = []
            for column in columns:
                col_expr = F.col(column)
                if func_lower in {"mean", "avg"}:
                    expressions.append(F.avg(col_expr).alias(column))
                elif func_lower == "sum":
                    expressions.append(F.sum(col_expr).alias(column))
                elif func_lower == "max":
                    expressions.append(F.max(col_expr).alias(column))
                elif func_lower == "min":
                    expressions.append(F.min(col_expr).alias(column))
                elif func_lower == "count":
                    expressions.append(F.count(col_expr).alias(column))
                elif func_lower in {"std", "stddev"}:
                    ddof = kwargs.get("ddof", 1)
                    expressions.append(
                        (F.stddev_samp if ddof == 1 else F.stddev_pop)(col_expr).alias(column)
                    )
                elif func_lower in {"var", "variance"}:
                    ddof = kwargs.get("ddof", 1)
                    expressions.append(
                        (F.var_samp if ddof == 1 else F.var_pop)(col_expr).alias(column)
                    )
                else:
                    raise ValueError(f"Unsupported aggregation function: {func_name}")
            rows.append(self._collect_aggregation(expressions, func_name))
        if not rows:
            return pd.DataFrame()
        result = pd.concat(rows, ignore_index=True)
        result = result.set_index("aggregation")
        return self._convert_agg_result(result)

    def max(self) -> Union[spark.DataFrame, float]:
        numeric_columns = self._numeric_columns()
        expressions = [F.max(F.col(column)).alias(column) for column in numeric_columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*numeric_columns))

    def idxmax(self) -> Union[spark.DataFrame, float]:
        pdf = self.data.toPandas()
        result = pdf.idxmax()
        if isinstance(result, pd.Series):
            return result.to_frame().T
        return result

    def min(self) -> Union[spark.DataFrame, float]:
        numeric_columns = self._numeric_columns()
        expressions = [F.min(F.col(column)).alias(column) for column in numeric_columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*numeric_columns))

    def count(self) -> Union[spark.DataFrame, float]:
        expressions = [F.count(F.col(column)).alias(column) for column in self.data.columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*self.data.columns))

    def sum(self) -> Union[spark.DataFrame, float]:
        numeric_columns = self._numeric_columns()
        expressions = [F.sum(F.col(column)).alias(column) for column in numeric_columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*numeric_columns))

    def mean(self) -> Union[spark.DataFrame, float]:
        if self.data is None:
            return float("nan")

        numeric_types = {
            "byte",
            "short",
            "int",
            "bigint",
            "long",
            "float",
            "double",
            "tinyint",
            "smallint",
        }

        numeric_columns: List[str] = []
        for name, dtype in self.data.dtypes:
            dtype_lower = dtype.lower()
            if (
                dtype_lower in numeric_types
                or dtype_lower.startswith("decimal")
                or dtype_lower == "boolean"
            ):
                numeric_columns.append(name)

        if not numeric_columns:
            empty_schema = T.StructType([])
            return self.session.createDataFrame(
                self.session.sparkContext.emptyRDD(), empty_schema
            )

        agg_expressions = [F.avg(F.col(column)).alias(column) for column in numeric_columns]
        aggregated = self.data.select(*numeric_columns).agg(*agg_expressions)

        if len(numeric_columns) == 1:
            value_row = aggregated.select(numeric_columns[0]).head(1)
            if not value_row:
                return float("nan")
            value = value_row[0][0]
            return float(value) if value is not None else float("nan")

        return aggregated.select(*numeric_columns)

    def mode(
        self, numeric_only: bool = False, dropna: bool = True
    ) -> Union[spark.DataFrame, float]:
        columns = self._numeric_columns() if numeric_only else self.data.columns
        modes: Dict[str, List[Any]] = {}
        for column in columns:
            col_expr = F.col(column)
            df = self.data.select(column)
            if dropna:
                df = df.filter(col_expr.isNotNull())
            counts = df.groupBy(column).agg(F.count(lit(1)).alias("count"))
            if counts.rdd.isEmpty():
                modes[column] = []
                continue
            max_count = counts.agg(F.max("count")).collect()[0][0]
            top_rows = counts.filter(F.col("count") == max_count).select(column).collect()
            modes[column] = [row[column] for row in top_rows]
        max_len = max((len(values) for values in modes.values()), default=0)
        if max_len == 0:
            return self.session.createDataFrame([], schema=T.StructType([]))
        rows = []
        for i in range(max_len):
            row = {}
            for column, values in modes.items():
                row[column] = values[i] if i < len(values) else None
            rows.append(row)
        pdf = pd.DataFrame(rows)
        return pdf

    def var(
        self, skipna: bool = True, ddof: int = 1, numeric_only: bool = False
    ) -> Union[spark.DataFrame, float]:
        columns = self._numeric_columns() if numeric_only else self.data.columns
        variance_fn = F.var_samp if ddof == 1 else F.var_pop
        expressions = [variance_fn(F.col(column)).alias(column) for column in columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*columns))

    def log(self) -> spark.DataFrame:
        return self.data.select(
            *[F.log(F.col(column)).alias(column) for column in self.data.columns]
        )

    def std(self, skipna: bool = True, ddof: int = 1) -> Union[spark.DataFrame, float]:
        columns = self._numeric_columns()
        std_fn = F.stddev_samp if ddof == 1 else F.stddev_pop
        expressions = [std_fn(F.col(column)).alias(column) for column in columns]
        aggregated = self.data.agg(*expressions)
        return self._convert_agg_result(aggregated.select(*columns))

    def cov(self):
        columns = self._numeric_columns()
        data = []
        for column in columns:
            row = {"column": column}
            for other in columns:
                row[other] = self.data.stat.cov(column, other)
            data.append(row)
        pdf = pd.DataFrame(data).set_index("column")
        return pdf

    def quantile(self, q: float = 0.5) -> spark.DataFrame:
        columns = self._numeric_columns()
        quantiles = {}
        for column in columns:
            value = self.data.approxQuantile(column, [q], 1e-3)
            quantiles[column] = value[0] if value else None
        pdf = pd.DataFrame([quantiles], index=[q])
        return pdf

    def coefficient_of_variation(self) -> Union[spark.DataFrame, float]:
        mean_value = self.mean()
        std_value = self.std()
        if isinstance(mean_value, (int, float)) and isinstance(std_value, (int, float)):
            return std_value / mean_value if mean_value else float("nan")
        columns = self.data.columns
        mean_pdf = self._ensure_pandas_frame(mean_value, columns)
        std_pdf = self._ensure_pandas_frame(std_value, columns)
        mean_series = mean_pdf.iloc[0]
        std_series = std_pdf.iloc[0]
        cv = (std_series / mean_series).to_frame().T
        cv.index = ["cv"]
        return cv

    def sort_index(self, ascending: bool = True, **kwargs) -> spark.DataFrame:
        if "index" in self.data.columns:
            order_col = F.col("index").asc() if ascending else F.col("index").desc()
            return self.data.orderBy(order_col)
        indexed = self._with_row_index(self.data, "__row_id")
        order_col = F.col("__row_id").asc() if ascending else F.col("__row_id").desc()
        sorted_df = indexed.orderBy(order_col)
        return sorted_df.drop("__row_id")

    def corr(
        self,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        numeric_only: bool = False,
    ) -> Union[spark.DataFrame, float]:
        if method != "pearson":
            pdf = self.data.toPandas()
            return pdf.corr(method=method, numeric_only=numeric_only)
        columns = self._numeric_columns() if numeric_only else self._numeric_columns(include_boolean=False)
        data = []
        for column in columns:
            row = {"column": column}
            for other in columns:
                row[other] = self.data.stat.corr(column, other)
            data.append(row)
        pdf = pd.DataFrame(data).set_index("column")
        return pdf

    def isna(self) -> spark.DataFrame:
        return self.data.select(
            *[F.col(column).isNull().alias(column) for column in self.data.columns]
        )

    def sort_values(
        self, by: Union[str, list[str]], ascending: bool = True, **kwargs
    ) -> spark.DataFrame:
        columns = [by] if isinstance(by, str) else list(by)
        sort_columns = [F.col(column).asc() if ascending else F.col(column).desc() for column in columns]
        return self.data.sort(*sort_columns)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> spark.DataFrame:
        df = self.data
        if dropna:
            conditions = [F.col(column).isNotNull() for column in df.columns]
            if conditions:
                df = df.filter(reduce(lambda a, b: a & b, conditions))
        grouped = df.groupBy(*df.columns).agg(F.count(lit(1)).alias("count"))
        if normalize:
            total_row = grouped.agg(F.sum("count").alias("total")).collect()[0]
            total = total_row["total"] or 1
            grouped = grouped.withColumn("count", F.col("count") / F.lit(total))
        order_cols = [F.col("count").asc() if ascending else F.col("count").desc()]
        if sort:
            grouped = grouped.sort(*order_cols)
        return grouped

    def fillna(
        self,
        values: Optional[Union[ScalarType, dict[str, ScalarType]]] = None,
        method: Optional[Literal["bfill", "ffill"]] = None,
        **kwargs,
    ) -> spark.DataFrame:
        if method is not None:
            pdf = self.data.toPandas()
            if method == "bfill":
                result = pdf.bfill(**kwargs)
            elif method == "ffill":
                result = pdf.ffill(**kwargs)
            else:
                raise ValueError(f"Wrong fill method: {method}")
            return self.session.createDataFrame(result.reset_index(drop=True))
        return self.data.fillna(values)

    def na_counts(self) -> Union[spark.DataFrame, int]:
        expressions = [
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in self.data.columns
        ]
        aggregated = self.data.agg(*expressions)
        total = aggregated.select(*(F.col(column) for column in aggregated.columns))
        result = total.toPandas()
        result.index = ["na_counts"]
        if result.shape[0] == 1 and result.shape[1] == 1:
            return int(result.iloc[0, 0])
        return result

    def dot(self, other: Union["SparkDataset", np.ndarray]) -> spark.DataFrame:
        pdf = self.data.toPandas()
        if isinstance(other, np.ndarray):
            other_df = pd.DataFrame(other)
            result = pdf.dot(other_df.T)
        else:
            other_pdf = other.data.toPandas()
            result = pdf.dot(other_pdf.T)
        return self.session.createDataFrame(result.reset_index(drop=True))

    def dropna(
        self,
        how: Literal["any", "all"] = "any",
        subset: Optional[Union[str, Iterable[str]]] = None,
        axis: Union[Literal["index", "rows", "columns"], int] = 0,
    ) -> spark.DataFrame:
        if axis not in (0, "index", "rows"):
            raise NotImplementedError("Spark backend supports dropping NA along rows only")
        if isinstance(subset, str):
            subset = [subset]
        return self.data.dropna(how=how, subset=subset)

    def transpose(self, names: Optional[Sequence[str]] = None) -> spark.DataFrame:
        pdf = self.data.toPandas().transpose()
        if names is not None:
            pdf.columns = names
        return self.session.createDataFrame(pdf.reset_index())

    def sample(
        self,
        frac: Optional[float] = None,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> spark.DataFrame:
        if frac is not None:
            return self.data.sample(withReplacement=False, fraction=frac, seed=random_state)
        if n is not None:
            total = self.__len__()
            if total == 0:
                return self.data.limit(0)
            fraction = min(1.0, n / total)
            sampled = self.data.sample(withReplacement=False, fraction=fraction, seed=random_state)
            return sampled.limit(n)
        raise ValueError("Either frac or n must be provided")

    def select_dtypes(
        self,
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> spark.DataFrame:
        include = include.lower() if include else None
        exclude = exclude.lower() if exclude else None
        selected = []
        for name, dtype in self.data.dtypes:
            dtype_lower = dtype.lower()
            if include and include not in dtype_lower:
                continue
            if exclude and exclude in dtype_lower:
                continue
            selected.append(name)
        return self.data.select(*selected)

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self.data.isin(values)

    def merge(
        self,
        right: "SparkDataset",
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        left_index: Optional[bool] = None,
        right_index: Optional[bool] = None,
        suffixes: tuple[str, str] = ("_x", "_y"),
        how: Literal["left", "right", "inner", "outer", "cross"] = "inner",
    ) -> spark.DataFrame:
        if suffixes != ("_x", "_y"):
            raise NotImplementedError("Custom suffixes are not supported in the Spark backend yet")
        for on_ in [on, left_on, right_on]:
            if on_ and (
                on_ not in [*self.columns, *right.columns]
                if isinstance(on_, str)
                else any(c not in [*self.columns, *right.columns] for c in on_)
            ):
                raise MergeOnError(on_)

        left_df = self.data
        right_df = right.data
        if left_index:
            left_df = self._with_row_index(left_df, "index")
        if right_index:
            right_df = self._with_row_index(right_df, "index")
        if on is not None:
            join_condition = on
        elif left_on is not None and right_on is not None:
            if isinstance(left_on, str) and isinstance(right_on, str):
                join_condition = left_df[left_on] == right_df[right_on]
            else:
                raise NotImplementedError("Merge with multiple join keys is not yet supported")
        else:
            join_condition = None
        return left_df.join(right_df, on=join_condition, how=how)

    def drop(self, labels: str = "", axis: int = 1) -> spark.DataFrame:
        if axis == 1:
            labels = [labels] if isinstance(labels, str) else list(labels)
            return self.data.drop(*labels)
        raise NotImplementedError("Spark backend supports only axis=1 for drop")

    def filter(
        self,
        items: Optional[list] = None,
        like: Optional[str] = None,
        regex: Optional[str] = None,
        axis: int = 0,
    ) -> spark.DataFrame:
        if items is not None:
            return self.data.select(*items)
        columns = self.data.columns
        if like is not None:
            filtered = [column for column in columns if like in column]
            return self.data.select(*filtered)
        if regex is not None:
            import re

            pattern = re.compile(regex)
            filtered = [column for column in columns if pattern.search(column)]
            return self.data.select(*filtered)
        return self.data

    def rename(self, columns: dict[str, str]) -> spark.DataFrame:
        renamed = self.data
        for old_name, new_name in columns.items():
            renamed = renamed.withColumnRenamed(old_name, new_name)
        return renamed

    def replace(
        self, to_replace: Any = None, value: Any = None, regex: bool = False
    ) -> spark.DataFrame:
        if isinstance(to_replace, spark.DataFrame) and len(to_replace.columns) == 1:
            to_replace = [row[0] for row in to_replace.collect()]
        return self.data.replace(to_replace=to_replace, value=value, subset=None)

    def reindex(
        self, labels: str = "", fill_value: Optional[str] = None
    ) -> spark.DataFrame:
        pdf = self.data.toPandas().reindex(labels, fill_value=fill_value)
        return self.session.createDataFrame(pdf.reset_index())

    def list_to_columns(self, column: str) -> spark.DataFrame:
        sample_row = self.data.select(column).limit(1).collect()
        if not sample_row:
            return self.data
        first_value = sample_row[0][column]
        if not isinstance(first_value, (list, tuple)):
            return self.data
        n_cols = len(first_value)
        expanded_columns = [F.col(column)[i].alias(f"{column}_{i}") for i in range(n_cols)]
        return self.data.select(*expanded_columns)
