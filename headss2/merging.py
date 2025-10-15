from pyspark.sql import SparkSession, Row, functions as F
from pyspark import sql
from pyspark.sql.functions import pandas_udf
import pandas as pd
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
    cluster1, cluster2 = cluster_ids
    cluster1_df = clustered_df[clustered_df["cluster"] == cluster1]
    cluster2_df = clustered_df[clustered_df["cluster"] == cluster2]

    cluster1_region = int(cluster1_df["region"].iloc[0])
    cluster2_region = int(cluster2_df["region"].iloc[0])

    split_regions_indexed = split_regions.set_index("region")

    limits1 = [[
        split_regions_indexed.loc[cluster1_region, f"{col}_min"],
        split_regions_indexed.loc[cluster1_region, f"{col}_max"]
    ] for col in cluster_cols]

    limits2 = [[
        split_regions_indexed.loc[cluster2_region, f"{col}_min"],
        split_regions_indexed.loc[cluster2_region, f"{col}_max"]
    ] for col in cluster_cols]

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


def _total_point_overlap(n_merged: int, n_cluster1: int, n_cluster2: int):
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

def _compute_overlap_stats(data: sql.DataFrame, split_regions: pd.DataFrame, cluster_cols: list[str], pairs_df: pd.DataFrame, minimum_members: int) -> pd.DataFrame:
    stats = []

    for row in pairs_df.itertuples():
        c1, c2 = row.cluster1, row.cluster2

        cluster1_df = data.filter(data.cluster == c1).toPandas()
        cluster2_df = data.filter(data.cluster == c2).toPandas()
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

        tpo = _total_point_overlap(n_overlap, n1, n2)
        stats.append(OverlapStats(c1, c2, n_overlap, n1, n2, brpo, tpo))

    return pd.DataFrame([s.__dict__ for s in stats])


def _should_merge(overlap_stats_df: pd.DataFrame, bound_region_point_overlap_threshold: float, total_point_overlap_threshold: float, min_n_overlap: int) -> pd.DataFrame:
    df = overlap_stats_df.copy()

    if len(df) == 0:
        return pd.DataFrame([], columns=["total_point_overlap", "bound_region_point_overlap", "n_overlap", "should_merge"])
    
    df["should_merge"] = (
        (df["total_point_overlap"] >= total_point_overlap_threshold) &
        (df["bound_region_point_overlap"] >= bound_region_point_overlap_threshold) &
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
                  bound_region_point_overlap_threshold: float = 0.1, total_point_overlap_threshold: float = 0.5, min_n_overlap:int = 10, min_members = 10) -> sql.DataFrame:
    """Merge clusters based on overlaps.

    Args:
        clustered (sql.DataFrame): Clustered data.
        cluster_columns (list[str]): Columns that we clustered on.
        split_regions (pd.DataFrame): Split regions data.
        bound_region_point_overlap_threshold (float | None, optional): Minimum threshold for merging: fraction of joint data points lying within the bound overlap region divided by the smallest of the two clusters. Previously known as 'total threshold'. Defaults to 0.5.
        total_point_overlap_threshold (float | None, optional): Minimum threshold for merging: fraction of all joint data points divided by the smallest of the two clusters. Previously known as 'overlap threshold'. Defaults to 0.1.
        min_n_overlap (int | None, optional): Minimum number of overlapping points to allow merging. Defaults to 10.        
        min_members (int, optional): Minimum number of members per cluster. Defaults to 10.

    Returns:
        sql.DataFrame: Clustered data with merged clusters.
    """

    clustered = clustered.withColumn("cluster", F.col("cluster").cast("string"))
    cluster_bounds = _get_cluster_bounds(clustered, cluster_columns)
    overlap_pairs = _find_overlapping_pairs(cluster_bounds, cluster_columns)

    if len(overlap_pairs) == 0:
        return clustered

    stats_df = _compute_overlap_stats(data=clustered, split_regions=split_regions, cluster_cols=cluster_columns, pairs_df=overlap_pairs, minimum_members=min_members)
    should_merge_df = _should_merge(stats_df, bound_region_point_overlap_threshold, total_point_overlap_threshold, min_n_overlap)
    uf = _merge_clusters_union_find(should_merge_df)

    return _assign_new_clusters(uf, clustered)