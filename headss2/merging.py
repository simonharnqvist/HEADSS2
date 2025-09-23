from pyspark.sql import SparkSession, Row, functions as F
from pyspark import sql
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterable, Iterator
import numpy as np
from headss2.union_find import UnionFind
from itertools import chain


def _get_cluster_bounds(
    clustered: sql.DataFrame, cluster_columns: list[str]
) -> sql.DataFrame:
    """Compute bounding boxes (min/max) for each cluster."""

    mins = clustered.groupBy("cluster").min(*cluster_columns)
    max_ = clustered.groupBy("cluster").max(*cluster_columns)
    min_max = mins.join(max_, on="cluster")

    #rename for Pandas compatibility:
    for col in cluster_columns:
        min_col = f"min({col})"
        max_col = f"max({col})"
        min_max = min_max.withColumnRenamed(min_col, f"{col}_min")
        min_max = min_max.withColumnRenamed(max_col, f"{col}_max")
    return min_max


def _find_overlapping_pairs(
    cluster_bounds: sql.DataFrame, cluster_columns: list[str]
) -> pd.DataFrame:
    """Find pairs of clusters with overlapping bounding boxes."""
    bounds_1 = cluster_bounds.alias("bounds_1")
    bounds_2 = cluster_bounds.alias("bounds_2")

    join_cond = F.col("bounds_1.cluster") < F.col("bounds_2.cluster")

    overlap_conditions = []
    for c in cluster_columns:
        cond = (F.col(f"bounds_1.{c}_min") <= F.col(f"bounds_2.{c}_max")) & (
            F.col(f"bounds_1.{c}_max") >= F.col(f"bounds_2.{c}_min")
        )
        overlap_conditions.append(cond)

    full_overlap_cond = overlap_conditions[0]
    for cond in overlap_conditions[1:]:
        full_overlap_cond = full_overlap_cond & cond

    overlapping_pairs = bounds_1.join(bounds_2, join_cond & full_overlap_cond).select(
        F.col("bounds_1.cluster").alias("cluster1"), F.col("bounds_2.cluster").alias("cluster2")
    )
    return overlapping_pairs.toPandas()


def _get_n_overlaps(df: pd.DataFrame, cluster_bounds: sql.DataFrame, cluster1: str, cluster2: str):
    """Pandas UDF to calculate number of points in overlap between two clusters"""

    cluster_bounds = cluster_bounds.toPandas()

    if not isinstance(cluster1, str) or not isinstance(cluster2, str):
        raise ValueError("Cluster names must be string")

    bounds2 = cluster_bounds[cluster_bounds["cluster"] == cluster2]
    bounds1 = cluster_bounds[cluster_bounds["cluster"] == cluster1]


    if bounds1.empty or bounds2.empty:
        return 0, 0
    
    b1 = bounds1.iloc[0]
    b2 = bounds2.iloc[0]


    cluster1_in_2 = (
        (df["cluster"] == cluster1)
        & (df["x"] >= b2["x_min"])
        & (df["x"] < b2["x_max"])
        & (df["y"] >= b2["y_min"])
        & (df["y"] < b2["y_max"])
    )

    cluster2_in_1 = (
        (df["cluster"] == cluster2)
        & (df["x"] >= b1["x_min"])
        & (df["x"] < b1["x_max"])
        & (df["y"] >= b1["y_min"])
        & (df["y"] < b1["y_max"])
    )
    return len(df[cluster1_in_2]), len(df[cluster2_in_1])


def _apply_get_n_overlaps(
    clustered: sql.DataFrame,
    overlapping_clusters: pd.DataFrame,
    cluster_bounds: sql.DataFrame,
):
    """Perform overlap calculations per cluster pair"""
    results = []
    for _, row in overlapping_clusters.iterrows():
        cluster1, cluster2 = row["cluster1"], row["cluster2"]

        clustered_subset = clustered.filter(
            (clustered.cluster == cluster1) | (clustered.cluster == cluster2)
        ).toPandas()

        n_cluster1, n_cluster2 = (len(clustered_subset[clustered_subset["cluster"] == cluster1]), 
                              len(clustered_subset[clustered_subset["cluster"] == cluster2]))

        n_cluster1_in_2, n_cluster2_in_1 = _get_n_overlaps(
            df=clustered_subset, cluster_bounds=cluster_bounds, cluster1=cluster1, cluster2=cluster2
        )
        results.append((cluster1, cluster2, n_cluster1, n_cluster2, n_cluster1_in_2, n_cluster2_in_1))

    return pd.DataFrame(results, columns=["cluster1", "cluster2", "n_cluster1", "n_cluster2", "n_cluster1_in2", "n_cluster2_in1"])

def _calculate_overlap_stats(n_overlaps_df: pd.DataFrame) -> pd.DataFrame:
    """Compute overlap stats per cluster pair"""
    n_overlaps_df["n_in_overlap"] = n_overlaps_df["n_cluster1_in2"] + n_overlaps_df["n_cluster2_in1"]
    n_overlaps_df["n_total"] = n_overlaps_df["n_cluster1"] + n_overlaps_df["n_cluster2"]

    n_overlaps_df["total_overlap_fraction"] = n_overlaps_df.apply(
    lambda row: (row["n_in_overlap"] / row["n_total"]) if row["n_total"]> 0 else 0, axis=1
)
    n_overlaps_df["per_cluster_overlap_fraction"] = n_overlaps_df.apply(
    lambda row: max(
        row["n_cluster1_in2"] / row["n_cluster1"] if row["n_cluster1"] > 0 else 0,
        row["n_cluster2_in1"] / row["n_cluster2"] if row["n_cluster2"] > 0 else 0
    ),
    axis=1
)
    return n_overlaps_df

def _should_merge(overlap_stats_df: pd.DataFrame, 
                  per_cluster_overlap_threshold: float, # former per_cluster_overlap_threshold
                  combined_per_cluster_overlap_threshold: float,  # former combined_overlap_threshold
                  min_n_overlap: int) -> pd.DataFrame:
    """Using user-defined thresholds, assess whether each cluster overlap is sufficient for merging clusters"""
    
    overlap_stats_df["should_merge"] = (
    (overlap_stats_df["per_cluster_overlap_fraction"] >= per_cluster_overlap_threshold) &
    (overlap_stats_df["total_overlap_fraction"] >= combined_per_cluster_overlap_threshold) &
    (overlap_stats_df["n_in_overlap"]) >= min_n_overlap
)
    return overlap_stats_df


def _merge_clusters_union_find(should_merge_df: pd.DataFrame) -> UnionFind:
    """Use union-find algorithm to merge clusters"""

    union_find = UnionFind()

    for row in should_merge_df[should_merge_df["should_merge"] == True][["cluster1", "cluster2"]].itertuples(index=False):
        union_find.union(row.cluster1, row.cluster2)

    return union_find

def _assign_new_clusters(union_find: UnionFind, clustered: sql.DataFrame) -> sql.DataFrame:
    """Assign new cluster IDs using union-find object"""
    replacement_dict = {str(x): str(union_find.find(x)) for x in union_find.parent.keys()}
    return clustered.replace(to_replace=replacement_dict, subset=['cluster'])


def cluster_merge(clustered: sql.DataFrame, cluster_columns: list[str], per_cluster_overlap_threshold: float = 0.1, combined_overlap_threshold: float = 0.5, min_n_overlap:int = 10) -> sql.DataFrame:
    """
    Merges overlapping clusters based on defined overlap thresholds.

    Args:
        clustered (sql.DataFrame): Input DataFrame containing data points with assigned cluster labels.
        cluster_columns (list[str]): Column names used to define the clustering dimensions.
        per_cluster_per_cluster_overlap_threshold (float): 
            Minimum threshold for per-cluster overlap. A merge is allowed only if:
                max(n_overlap / n_cluster1, n_overlap / n_cluster2) > threshold.
            This checks how much one cluster is embedded in another.
            (Formerly called `per_cluster_overlap_threshold`). Default is 0.1.
        combined_per_cluster_overlap_threshold (float): 
            Minimum threshold for total (combined) overlap. A merge is allowed only if:
                (n_overlap / (n_cluster1 + n_cluster2)) > threshold.
            This ensures the overlap is significant in absolute terms.
            (Formerly called `combined_overlap_threshold`). Default is 0.5.
        min_n_overlap (int): Minimum number of overlapping points required to consider a merge. Default is 10.

    Returns:
        sql.DataFrame: DataFrame with clusters merged and reassigned.
    """

    clustered = clustered.withColumn("cluster", F.col("cluster").cast("string"))

    cluster_bounds = _get_cluster_bounds(clustered=clustered, cluster_columns=cluster_columns)
    overlapping_clusters = _find_overlapping_pairs(cluster_bounds=cluster_bounds, cluster_columns=cluster_columns)
    n_overlaps_df = _apply_get_n_overlaps(
        clustered=clustered,
        overlapping_clusters=overlapping_clusters,
        cluster_bounds=cluster_bounds,
    )
    overlap_stats_df = _calculate_overlap_stats(n_overlaps_df=n_overlaps_df)
    should_merge_df = _should_merge(overlap_stats_df=overlap_stats_df, per_cluster_overlap_threshold=per_cluster_overlap_threshold, combined_per_cluster_overlap_threshold=combined_overlap_threshold, min_n_overlap=min_n_overlap)
    union_find_obj = _merge_clusters_union_find(should_merge_df=should_merge_df)
    clustered_merged = _assign_new_clusters(union_find=union_find_obj, clustered=clustered)
    return clustered_merged