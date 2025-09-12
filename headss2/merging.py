from pyspark.sql import SparkSession, Row, functions as F
from pyspark import sql
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterable, Iterator
import numpy as np

spark = SparkSession.builder.getOrCreate()


def get_cluster_bounds(
    clustered: sql.DataFrame, split_columns: list[str], group_col="group"
) -> sql.DataFrame:
    """Compute bounding boxes (min/max) for each cluster."""

    mins = clustered.groupBy("group").min(*split_columns)
    max_ = clustered.groupBy("group").max(*split_columns)
    min_max = mins.join(max_, on="group")
    return min_max


def find_overlapping_pairs(
    cluster_bounds: sql.DataFrame, split_columns: list[str]
) -> pd.DataFrame:
    """Find pairs of clusters with overlapping bounding boxes."""
    bounds_1 = cluster_bounds.alias("bounds_1")
    bounds_2 = cluster_bounds.alias("bounds_2")

    join_cond = F.col("bounds_1.group") < F.col("bounds_2.group")

    overlap_conditions = []
    for c in split_columns:
        cond = (F.col(f"bounds_1.min({c})") <= F.col(f"bounds_2.max({c})")) & (
            F.col(f"bounds_1.max({c})") >= F.col(f"bounds_2.min({c})")
        )
        overlap_conditions.append(cond)

    full_overlap_cond = overlap_conditions[0]
    for cond in overlap_conditions[1:]:
        full_overlap_cond = full_overlap_cond & cond

    overlapping_pairs = bounds_1.join(bounds_2, join_cond & full_overlap_cond).select(
        F.col("bounds_1.group").alias("group1"), F.col("bounds_2.group").alias("group2")
    )
    return overlapping_pairs.toPandas()


# @pandas_udf("int")
def get_n_overlaps(df, group_bounds, group1, group2):
    """Pandas UDF to calculate number of points in overlap between two clusters"""

    group_bounds = group_bounds.toPandas()
    print(group_bounds)

    bounds2 = group_bounds[group_bounds["group"] == group2]
    bounds1 = group_bounds[group_bounds["group"] == group1]

    mask = (
        (df["group"] == group1)
        & (df["x"] >= float(bounds2["min(x)"]))
        & (df["x"] < float(bounds2["max(x)"]))
        & (df["y"] >= float(bounds2["min(y)"]))
        & (df["y"] < float(bounds2["max(y)"]))
    ) | (
        (df["group"] == group2)
        & (df["x"] >= float(bounds1["min(x)"]))
        & (df["x"] < float(bounds1["max(x)"]))
        & (df["y"] >= float(bounds1["min(y)"]))
        & (df["y"] < float(bounds1["max(y)"]))
    )

    return pd.Series(len(df[mask]))


def apply_get_n_overlaps(
    clustered: sql.DataFrame,
    overlapping_groups: pd.DataFrame,
    group_bounds: sql.DataFrame,
):
    results = []
    for _, row in overlapping_groups.iterrows():
        group1, group2 = int(row["group1"]), int(row["group2"])

        clustered_subset = clustered.filter(
            (clustered.group == group1) | (clustered.group == group2)
        ).toPandas()

        count = get_n_overlaps(
            df=clustered_subset, group_bounds=group_bounds, group1=group1, group2=group2
        )
        results.append((group1, group2, count))

    return results


def main():
    """temp function for dev"""

    clustered = spark.read.csv(
        "tests/ground_truth/t48k_clustered.csv", header=True, inferSchema=True
    )
    cluster_bounds = get_cluster_bounds(clustered, ["x", "y"])
    overlapping_groups = find_overlapping_pairs(cluster_bounds, ["x", "y"])
    res = apply_get_n_overlaps(
        clustered=clustered,
        overlapping_groups=overlapping_groups,
        group_bounds=cluster_bounds,
    )

    print(res)


main()
