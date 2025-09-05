from pyspark import sql
from pyspark.sql.functions import col, lit, floor, when
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, StructType, StructField
from typing import List
import numpy as np
import itertools
import pandas as pd
from functools import reduce
import operator


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


def get_step_and_limits(
    df: sql.DataFrame, split_columns: List[str], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        step: np.ndarray of step sizes for each dimension
        limits: np.ndarray of region start coordinates (shape: [num_regions, num_dims])
    """
    agg_exprs = []
    for c in split_columns:
        agg_exprs += [F.min(c).alias(f"{c}_min"), F.max(c).alias(f"{c}_max")]
    stats = df.agg(*agg_exprs).collect()[0]

    mins = []
    steps = []
    for c in split_columns:
        c_min = stats[f"{c}_min"]
        c_max = stats[f"{c}_max"]
        step = (c_max - c_min) / n
        mins.append(c_min)
        steps.append(step)

    # Generate grid of minima for each region
    limits_per_dim = [
        np.linspace(m, m + (n - 1) * s, 2 * n - 1) for m, s in zip(mins, steps)
    ]
    limits = (
        np.array(np.meshgrid(*limits_per_dim, indexing="ij"))
        .reshape(len(split_columns), -1)
        .T
    )

    return np.array(steps), limits


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


def assign_regions(
    df: sql.DataFrame, split_regions: sql.DataFrame, split_columns: list[str]
) -> sql.DataFrame:
    joined = df.crossJoin(split_regions)

    conditions = [
        (col(col_name) >= col(f"{col_name}_min"))
        & (col(col_name) < col(f"{col_name}_max"))
        for col_name in split_columns
    ]

    filtered = joined.filter(reduce(operator.and_, conditions))
    selected_columns = split_columns + ["region"]
    return filtered.select(*selected_columns)


class Regions:
    def __init__(
        self,
        split_data: sql.DataFrame,
        split_regions: pd.DataFrame,
        stitch_regions: pd.DataFrame,
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
    """
    Computes regions for a DataFrame by assigning each row a region ID based on spatial bins,
    and returns metadata (split + stitch regions) for those spatial divisions.

    Args:
        spark_session: Active SparkSession.
        df: Input Spark or Pandas DataFrame.
        n: Number of bins per dimension.
        split_columns: List of column names to split on.

    Returns:
        Regions: Contains region-annotated DataFrame and region metadata as pandas DataFrames.
    """
    if isinstance(df, pd.DataFrame):
        df = spark_session.createDataFrame(df)

    # Compute step size and all region minima
    step, limits = get_step_and_limits(df, split_columns, n)

    # Compute region boundaries
    low_cuts = get_minima(limits, step)
    high_cuts = get_maxima(limits, step)

    # Generate metadata tables
    split_regions = get_split_regions(
        spark_session=spark_session,
        limits=limits,
        split_columns=split_columns,
        step=step,
    )

    # Assign region ID to each row
    split_data = assign_regions(
        df=df, split_regions=split_regions, split_columns=["x", "y"]
    )
    stitch_regions = get_stitch_regions(
        spark_session=spark_session,
        low_cuts=low_cuts,
        high_cuts=high_cuts,
        split_columns=split_columns,
    )

    return Regions(
        split_data=split_data,
        split_regions=split_regions.toPandas(),
        stitch_regions=stitch_regions.toPandas(),
    )
