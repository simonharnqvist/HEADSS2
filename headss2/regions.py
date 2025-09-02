from pyspark import sql
from pyspark.sql.functions import col, lit
from pyspark.sql.types import FloatType, StructType, StructField
from typing import List
import numpy as np
import itertools
import pandas as pd


def cut_df_by_region(
    df: sql.DataFrame, minima: np.ndarray, split_columns: List[str], step: np.ndarray
) -> sql.DataFrame:
    for i, value in enumerate(minima):
        df = df.filter(
            (col(split_columns[i]) >= value)
            & (col(split_columns[i]) <= value + step[i])
        )
    return df


def split_dataframes(
    df: sql.DataFrame, limits: np.ndarray, split_columns: List[str], step: np.ndarray
) -> List[sql.DataFrame]:
    return [
        cut_df_by_region(df, minima=mins, split_columns=split_columns, step=step)
        for mins in limits
    ]


def concat_dataframes(
    dfs: List[sql.DataFrame], add_pos: bool = True, pos_col: str = "region"
) -> sql.DataFrame:
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")

    if add_pos:
        dfs = [df.withColumn(pos_col, lit(idx)) for idx, df in enumerate(dfs)]

    result = dfs[0]
    for df in dfs[1:]:
        result = result.unionByName(df)

    return result


def get_limits(
    df: sql.DataFrame, step: np.ndarray, split_columns: List[str]
) -> np.ndarray:
    stats = df.select([col(c) for c in split_columns]).summary("min", "max").toPandas()
    mins = [
        np.arange(
            float(stats[col][0]),
            float(stats[col][1]) - val + float(stats[col][1]) / 100,
            val / 2,
        )
        for col, val in zip(split_columns, step)
    ]
    return np.array(list(itertools.product(*mins)))


def get_step(df: sql.DataFrame, split_columns: List[str], n: int) -> np.ndarray:
    assert isinstance(df, sql.DataFrame), print(f"DF is of unexpected type {type(df)}")
    stats = df.select([col(c) for c in split_columns]).summary("min", "max").toPandas()
    return np.array(
        [(float(stats[col][1]) - float(stats[col][0])) / n for col in split_columns]
    )


def get_split_data(
    df: sql.DataFrame, limits: np.ndarray, split_columns: List[str], step: np.ndarray
) -> sql.DataFrame:
    dfs = split_dataframes(df, limits=limits, split_columns=split_columns, step=step)
    return concat_dataframes(dfs, add_pos=True, pos_col="region")


def get_split_regions(
    spark_session: sql.SparkSession,
    limits: np.ndarray,
    split_columns: List[str],
    step: np.ndarray,
):
    rows = []
    for region_idx, mins in enumerate(limits):
        region_dict = {"region": region_idx}
        for idx, col in enumerate(split_columns):
            region_dict[f"{col}_min"] = float(mins[idx])
            region_dict[f"{col}_max"] = float(mins[idx] + step[idx])
        rows.append(sql.Row(**region_dict))

    return spark_session.createDataFrame(rows)


def get_stitch_regions(
    spark_session: sql.SparkSession,
    low_cuts: np.ndarray,
    high_cuts: np.ndarray,
    split_columns: List[str],
) -> sql.DataFrame:
    rows = []
    for idx, (low, high) in enumerate(zip(low_cuts, high_cuts)):
        region_dict = {"region": idx}
        for i, col_name in enumerate(split_columns):
            region_dict[f"{col_name}_min"] = float(low[i])
            region_dict[f"{col_name}_max"] = float(high[i])
        rows.append(sql.Row(**region_dict))

    return spark_session.createDataFrame(rows)


def get_n_regions(n: int, split_columns: List[str]) -> int:
    return (2 * n - 1) ** len(split_columns)


def get_minima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    low = limits + step * 0.25
    for i, minimum in enumerate(np.min(limits, axis=0)):
        low[:, i][low[:, i] == low[:, i].min()] = minimum
    return low


def get_maxima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    high = limits + step * 0.75
    for i, maximum in enumerate(np.max(limits, axis=0)):
        high[:, i][high[:, i] == high[:, i].max()] = maximum + step[i]
    return high


class Regions:
    def __init__(
        self,
        split_data: sql.DataFrame,
        split_regions: sql.DataFrame,
        stitch_regions: sql.DataFrame,
    ):
        self.split_data = split_data
        self.split_regions = split_regions
        self.stitch_regions = stitch_regions


def make_regions(
    spark_session: sql.SparkSession,
    df: sql.DataFrame | pd.DataFrame,
    n: int,
    split_columns: List[str],
) -> Regions:

    if isinstance(df, pd.DataFrame):
        df = spark_session.createDataFrame(df)

    step = get_step(df, split_columns=split_columns, n=n)
    limits = get_limits(df, step=step, split_columns=split_columns)
    low_cuts = get_minima(limits, step=step)
    high_cuts = get_maxima(limits, step=step)

    split_data = get_split_data(
        df, limits=limits, split_columns=split_columns, step=step
    )
    split_regions = get_split_regions(
        spark_session=spark_session,
        limits=limits,
        split_columns=split_columns,
        step=step,
    )
    stitch_regions = get_stitch_regions(
        spark_session=spark_session,
        low_cuts=low_cuts,
        high_cuts=high_cuts,
        split_columns=split_columns,
    )

    return Regions(
        split_data=split_data,
        split_regions=split_regions,
        stitch_regions=stitch_regions,
    )
