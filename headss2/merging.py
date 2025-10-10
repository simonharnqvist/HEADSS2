from pyspark.sql import SparkSession, Row, functions as F
from pyspark import sql
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterable, Iterator
import numpy as np
from headss2.union_find import UnionFind
from itertools import chain
from dataclasses import dataclass

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

    join_cond = F.col("bounds_1.cluster") > F.col("bounds_2.cluster")

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

def _bound_region_point_overlap(clustered_df, cluster_ids, split_regions, cluster_cols, minimum_members):
    # Isolate clusters
    cluster1, cluster2 = cluster_ids
    cluster1_df = clustered_df[clustered_df["cluster"] == cluster1]
    cluster2_df = clustered_df[clustered_df["cluster"] == cluster2]

    # Get regions
    cluster1_region = int(cluster1_df["region"].iloc[0])
    cluster2_region = int(cluster2_df["region"].iloc[0])

    # Assume stitch_regions is indexed by 'region'
    split_regions_indexed = split_regions.set_index("region")

    limits1 = [[
        split_regions_indexed.loc[cluster1_region, f"{col}_min"],
        split_regions_indexed.loc[cluster1_region, f"{col}_max"]
    ] for col in cluster_cols]

    limits2 = [[
        split_regions_indexed.loc[cluster2_region, f"{col}_min"],
        split_regions_indexed.loc[cluster2_region, f"{col}_max"]
    ] for col in cluster_cols]

    # Apply bounds only on differing limits
    for i, col in enumerate(cluster_cols):
        if limits1[i] != limits2[i]:
            cluster2_df = cluster2_df[cluster2_df[col] > float(limits1[i][0])]
            cluster1_df = cluster1_df[cluster1_df[col] < float(limits2[i][1])]

    if len(cluster1_df) <= minimum_members or len(cluster2_df) <= minimum_members:
        return 0

    merged = cluster1_df.merge(cluster2_df, how="inner", on=cluster_cols)

    overlap_count = merged.shape[0]
    frac1 = overlap_count / cluster1_df.shape[0]
    frac2 = overlap_count / cluster2_df.shape[0]

    return max(frac1, frac2)


def _calculate_total_point_overlap(n_merged: int, n_cluster1: int, n_cluster2: int):
    return max(n_merged/n_cluster1 if n_cluster1 else 0, n_merged/n_cluster2 if n_cluster2 else 0)


@dataclass
class OverlapStats:
    cluster1: str
    cluster2: str
    n_overlap: int
    n1: int
    n2: int
    bound_region_point_overlap: float
    total_point_overlap: float

def _compute_overlap_stats(data: pd.DataFrame, split_regions: pd.DataFrame, cluster_cols: list[str], pairs_df: pd.DataFrame, minimum_members: int) -> pd.DataFrame:
    stats = []

    for row in pairs_df.itertuples():
        c1, c2 = row.cluster1, row.cluster2

        cluster1_df = data[data["cluster"] == c1]
        cluster2_df = data[data["cluster"] == c2]
        data_subset = pd.concat([cluster1_df, cluster2_df])

        merged = pd.merge(
            cluster1_df,
            cluster2_df,
            on=cluster_cols,
            how="inner"
        )
        n_overlap = len(merged)
        n1 = len(cluster1_df)
        n2 = len(cluster2_df)

        # brpo = overlap
        brpo = _bound_region_point_overlap(clustered_df=data_subset, 
                                           cluster_ids=[c1, c2],
                                           cluster_cols=cluster_cols,
                                           split_regions=split_regions,
                                           minimum_members=minimum_members)

        tpo = _calculate_total_point_overlap(n_overlap, n1, n2)

        print(c1, c2, brpo, tpo)

        stats.append(OverlapStats(c1, c2, n_overlap, n1, n2, brpo, tpo))

    return pd.DataFrame([s.__dict__ for s in stats])


def _should_merge(overlap_stats_df: pd.DataFrame, per_cluster_overlap_threshold: float, combined_overlap_threshold: float, min_n_overlap: int) -> pd.DataFrame:
    df = overlap_stats_df.copy()
    df["should_merge"] = (
        (df["total_point_overlap"] >= combined_overlap_threshold) &
        (df["bound_region_point_overlap"] >= per_cluster_overlap_threshold) &
        (df["n_overlap"] >= min_n_overlap)
    )
    return df


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


def cluster_merge(clustered: sql.DataFrame, cluster_columns: list[str], split_regions: pd.DataFrame,
                  per_cluster_overlap_threshold: float = 0.1, combined_overlap_threshold: float = 0.5, min_n_overlap:int = 10, min_members = 10) -> sql.DataFrame:
    """
    Merges overlapping clusters based on defined overlap thresholds.

    Args:
        clustered (sql.DataFrame): Input DataFrame containing data points with assigned cluster labels.
        cluster_columns (list[str]): Column names used to define the clustering dimensions.
        per_cluster_overlap_threshold (float): 
            Minimum threshold for per-cluster overlap. A merge is allowed only if:
                max(n_overlap / n_cluster1, n_overlap / n_cluster2) > threshold.
            This checks how much one cluster is embedded in another.
            (Formerly called `total_threshold`). Default is 0.1.
        combined_overlap_threshold (float): 
            Minimum threshold for total (combined) overlap. A merge is allowed only if:
                (n_overlap / (n_cluster1 + n_cluster2)) > threshold.
            This ensures the overlap is significant in absolute terms.
            (Formerly called `overlap_threshold`). Default is 0.5.
        min_n_overlap (int): Minimum number of overlapping points required to consider a merge. Default is 10.

    Returns:
        sql.DataFrame: DataFrame with clusters merged and reassigned.
    """
    clustered = clustered.withColumn("cluster", F.col("cluster").cast("string"))

    # Step 1: Compute cluster bounds
    cluster_bounds = _get_cluster_bounds(clustered, cluster_columns)

    # Step 2: Identify candidate overlapping clusters
    overlap_pairs = _find_overlapping_pairs(cluster_bounds, cluster_columns)

    # Step 3: Convert data to Pandas for pairwise comparison
    data = clustered.toPandas()

    # Step 4: Compute overlap stats for each pair
    stats_df = _compute_overlap_stats(data=data, split_regions=split_regions, cluster_cols=cluster_columns, pairs_df=overlap_pairs, minimum_members=min_members)

    # Step 5: Filter which overlaps should result in merges
    should_merge_df = _should_merge(stats_df, per_cluster_overlap_threshold, combined_overlap_threshold, min_n_overlap)

    # Step 6: Merge clusters using union-find
    uf = _merge_clusters_union_find(should_merge_df)

    # Step 7: Reassign clusters in the original Spark DataFrame
    return _assign_new_clusters(uf, clustered)