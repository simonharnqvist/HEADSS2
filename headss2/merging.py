from pyspark.sql import SparkSession, Row, functions as F
from pyspark import sql
from pyspark.sql.functions import col, array_distinct, flatten, collect_list, explode
from pyspark.sql.types import StructType, StructField, IntegerType
import pandas as pd
from graphframes import GraphFrame
from typing import Iterable, Iterator


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
) -> sql.DataFrame:
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
    return overlapping_pairs


def get_overlap_bounds(
    cluster_bounds: sql.DataFrame, group1: int, group2: int
) -> tuple[float]:
    # Get bounding box for both groups
    boxes = cluster_bounds.filter(
        (F.col("group") == group1) | (F.col("group") == group2)
    )

    # Compute scalar bounds for the overlap region
    x1 = boxes.agg(F.max("min(x)")).collect()[0][0]
    x2 = boxes.agg(F.min("max(x)")).collect()[0][0]
    y1 = boxes.agg(F.max("min(y)")).collect()[0][0]
    y2 = boxes.agg(F.min("max(y)")).collect()[0][0]

    return x1, x2, y1, y2


def get_overlap_fractions(
    clustered_subset: sql.DataFrame,
    g1: int,
    g2: int,
    cluster_bounds: sql.DataFrame,
    n_points_per_cluster: dict,
) -> tuple:
    # Get bounding boxes for both groups
    box1 = cluster_bounds.filter(cluster_bounds.group == g1).toPandas()
    box2 = cluster_bounds.filter(cluster_bounds.group == g2).toPandas()

    if box1.empty or box2.empty:
        return g1, g2, 0.0, 0.0

    # Compute overlap bounding box
    x1 = max(box1["min(x)"].iloc[0], box2["min(x)"].iloc[0])
    x2 = min(box1["max(x)"].iloc[0], box2["max(x)"].iloc[0])
    y1 = max(box1["min(y)"].iloc[0], box2["min(y)"].iloc[0])
    y2 = min(box1["max(y)"].iloc[0], box2["max(y)"].iloc[0])

    if x1 > x2 or y1 > y2:
        # No overlap in bounding box
        return g1, g2, 0.0, 0.0

    # Subset points in both groups within overlap region
    overlap_points = clustered_subset.filter(
        (clustered_subset["x"] >= x1)
        & (clustered_subset["x"] <= x2)
        & (clustered_subset["y"] >= y1)
        & (clustered_subset["y"] <= y2)
    )

    # Count overlap points per group
    counts_df = overlap_points.groupBy("group").count()
    counts = {row["group"]: row["count"] for row in counts_df.collect()}
    n_overlap_g1 = counts.get(g1, 0)
    n_overlap_g2 = counts.get(g2, 0)
    members_in_overlap = n_overlap_g1 + n_overlap_g2

    # Total members in each group
    n_total_g1 = n_points_per_cluster.get(g1, 0)
    n_total_g2 = n_points_per_cluster.get(g2, 0)
    total_members = n_total_g1 + n_total_g2

    overlap_fraction = members_in_overlap / total_members if total_members else 0.0
    total_fraction = max(
        n_overlap_g1 / n_total_g1 if n_total_g1 else 0.0,
        n_overlap_g2 / n_total_g2 if n_total_g2 else 0.0,
    )

    return g1, g2, overlap_fraction, total_fraction


def chain_merge_clusters(
    merge_pairs: sql.DataFrame, max_iter: int = 10
) -> sql.DataFrame:
    """
    Chain merge clusters by finding connected components using iterative Spark joins.
    """
    edges = merge_pairs.select(
        F.col("group1").alias("src"), F.col("group2").alias("dst")
    )

    vertices = (
        edges.select("src")
        .union(edges.select("dst"))
        .distinct()
        .withColumnRenamed("src", "id")
    )

    components = vertices.withColumn("component", F.col("id"))

    for _ in range(max_iter):
        # Propagate component IDs through edges
        joined = edges.join(
            components.withColumnRenamed("id", "src_id"),
            edges.src == F.col("src_id"),
            "inner",
        ).select(edges.dst.alias("id"), F.col("component"))

        new_components = (
            components.union(joined)
            .groupBy("id")
            .agg(F.min("component").alias("component"))
        )

        changed = (
            new_components.alias("new")
            .join(components.alias("old"), on="id")
            .filter(F.col("new.component") != F.col("old.component"))
            .count()
        )

        if changed == 0:
            break

        components = new_components

    return components.select(
        F.col("id").alias("group"), F.col("component").alias("merged_group")
    )


def merge_clusters(
    spark: SparkSession,
    clustered: sql.DataFrame,
    split_columns: list,
    min_merge_members: int = 10,
    overlap_merge_threshold: float = 0.5,
    total_merge_threshold: float = 0.1,
) -> sql.DataFrame:
    """Merge clusters to remove artificial effects of splitting.

    Args:
        spark (SparkSession): Spark session.
        clustered (DataFrame): Dataframe of clustered data, with each row assigned to a cluster in column 'group'.
        split_columns (list): List of columns to cluster/split by.
        min_merge_members (int, optional): Minimum members in overlap to allow merge. Defaults to 10.
        overlap_merge_threshold (float, optional): Fraction of mutual members within the overlap region required to allow merge.. Defaults to 0.5.
        total_merge_threshold (float, optional): Fraction of mutual members within the whole cluster to allow merge. Defaults to 0.1.

    Returns:
        DataFrame: Data with merged clusters.
    """
    cluster_bounds = get_cluster_bounds(clustered, split_columns)
    df_overlap_pairs = find_overlapping_pairs(cluster_bounds, split_columns).collect()

    n_points_per_cluster = {
        row["group"]: row["count"]
        for row in clustered.groupBy("group").count().collect()
    }

    overlaps_per_pair = []
    for pair in df_overlap_pairs:
        g1, g2 = pair["group1"], pair["group2"]
        subset = clustered.filter(clustered.group.isin(g1, g2))
        res = get_overlap_fractions(
            subset,
            g1=g1,
            g2=g2,
            cluster_bounds=cluster_bounds,
            n_points_per_cluster=n_points_per_cluster,
        )
        overlaps_per_pair.append(res)

    overlaps_per_pair = pd.DataFrame(
        overlaps_per_pair,
        columns=["group1", "group2", "overlap_fraction", "total_fraction"],
    )

    merge_pairs = overlaps_per_pair[
        (overlaps_per_pair["overlap_fraction"] > overlap_merge_threshold)
        & (overlaps_per_pair["total_fraction"] > total_merge_threshold)
    ]

    merged_clusters = chain_merge_clusters(spark.createDataFrame(merge_pairs))

    df_with_merged = clustered.join(merged_clusters, on="group", how="left").withColumn(
        "merged_group", F.coalesce(F.col("merged_group"), F.col("group"))
    )

    df_with_merged = df_with_merged.withColumn(
        "group", F.col("merged_group").cast("int")
    ).drop("merged_group")

    return df_with_merged
