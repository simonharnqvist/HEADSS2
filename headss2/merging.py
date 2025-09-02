from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.functions import col, array_distinct, flatten, collect_list, explode
from pyspark.sql.types import StructType, StructField, IntegerType
import pandas as pd
from graphframes import GraphFrame
from typing import Iterable


def get_cluster_bounds(
    df_clustered: DataFrame, split_columns: list[str], group_col="group"
) -> DataFrame:
    """Compute bounding boxes (min/max) for each cluster."""
    agg_exprs = []
    for c in split_columns:
        agg_exprs.append(F.min(c).alias(f"{c}_min"))
        agg_exprs.append(F.max(c).alias(f"{c}_max"))

    cluster_bounds = df_clustered.groupBy(group_col).agg(*agg_exprs)
    return cluster_bounds


def find_overlapping_pairs(
    cluster_bounds: DataFrame, split_columns: list[str]
) -> DataFrame:
    """Find pairs of clusters with overlapping bounding boxes."""
    bounds_1 = cluster_bounds.alias("bounds_1")
    bounds_2 = cluster_bounds.alias("bounds_2")

    join_cond = F.col("bounds_1.group") < F.col(
        "bounds_2.group"
    )  # avoid self and duplicate pairs

    overlap_conditions = []
    for c in split_columns:
        cond = (F.col(f"bounds_1.{c}_min") <= F.col(f"bounds_2.{c}_max")) & (
            F.col(f"bounds_1.{c}_max") >= F.col(f"bounds_2.{c}_min")
        )
        overlap_conditions.append(cond)

    full_overlap_cond = overlap_conditions[0]
    for cond in overlap_conditions[1:]:
        full_overlap_cond = full_overlap_cond & cond

    overlapping_pairs = bounds_1.join(bounds_2, join_cond & full_overlap_cond).select(
        F.col("bounds_1.group").alias("group1"), F.col("bounds_2.group").alias("group2")
    )
    return overlapping_pairs


def get_cluster_oob_matches(
    spark: SparkSession,
    df_clustered: DataFrame,
    df_overlap_pairs: DataFrame,
    split_columns: list,
    minimum_members: int,
    overlap_threshold: float,
    total_threshold: float,
) -> DataFrame:
    """
    Evaluate overlapping cluster pairs based on actual members in overlap regions,
    filtering with thresholds.
    """
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    df_a = df_clustered.select(
        [
            F.col(c).alias(f"{c}_a") if c in split_columns + ["group"] else F.col(c)
            for c in df_clustered.columns
        ]
    )
    df_b = df_clustered.select(
        [
            F.col(c).alias(f"{c}_b") if c in split_columns + ["group"] else F.col(c)
            for c in df_clustered.columns
        ]
    )

    joined = (
        df_overlap_pairs.join(df_a, F.col("group1") == F.col("group_a"))
        .join(df_b, F.col("group2") == F.col("group_b"))
        .select(
            "group1",
            "group2",
            *[f"{c}_a" for c in split_columns],
            *[f"{c}_b" for c in split_columns],
        )
    )

    schema = StructType(
        [StructField("group1", IntegerType()), StructField("group2", IntegerType())]
    )

    def pandas_oob_merge(pdf_iter):
        results = []
        for pdf in pdf_iter:
            print("Got type:", type(pdf))
            if not isinstance(pdf, pd.DataFrame):
                print("Data:", pdf)
                continue

            group1 = pdf["group1"].iloc[0]
            group2 = pdf["group2"].iloc[0]

            cluster1 = pdf[[f"{c}_a" for c in split_columns]].copy()
            cluster1.columns = split_columns

            cluster2 = pdf[[f"{c}_b" for c in split_columns]].copy()
            cluster2.columns = split_columns

            if len(cluster1) < minimum_members or len(cluster2) < minimum_members:
                continue

            overlap_bounds = {}
            for col_name in split_columns:
                min1, max1 = cluster1[col_name].min(), cluster1[col_name].max()
                min2, max2 = cluster2[col_name].min(), cluster2[col_name].max()
                overlap_min = max(min1, min2)
                overlap_max = min(max1, max2)

                if overlap_min > overlap_max:
                    break  # no overlap

                overlap_bounds[col_name] = (overlap_min, overlap_max)

            else:
                for col_name in split_columns:
                    low, high = overlap_bounds[col_name]
                    cluster1 = cluster1[
                        (cluster1[col_name] >= low) & (cluster1[col_name] <= high)
                    ]
                    cluster2 = cluster2[
                        (cluster2[col_name] >= low) & (cluster2[col_name] <= high)
                    ]

                if len(cluster1) < minimum_members or len(cluster2) < minimum_members:
                    continue

                merged = pd.merge(cluster1, cluster2, on=split_columns)
                N_merged = len(merged)
                if N_merged == 0:
                    continue

                perc_merged = [N_merged / len(cluster1), N_merged / len(cluster2)]
                perc_overlap = [N_merged / len(pdf), N_merged / len(pdf)]

                if (
                    max(perc_merged) > overlap_threshold
                    and max(perc_overlap) > total_threshold
                ):
                    results.append(
                        {"group1": min(group1, group2), "group2": max(group1, group2)}
                    )

                print(
                    f"Groups: {group1}, {group2}, merged: {N_merged}, cluster1: {len(cluster1)}, cluster2: {len(cluster2)}"
                )

        return pd.DataFrame(results)

    merged_df = joined.groupBy("group1", "group2").applyInPandas(
        pandas_oob_merge, schema=schema
    )
    return merged_df


def chain_merge_clusters(merge_pairs: DataFrame) -> DataFrame:
    """
    Chain merge clusters by finding connected components of cluster merge graph.
    """
    groups1 = merge_pairs.select(F.col("group1").alias("id"))
    groups2 = merge_pairs.select(F.col("group2").alias("id"))
    vertices = groups1.union(groups2).distinct()

    edges = merge_pairs.select(
        F.col("group1").alias("src"), F.col("group2").alias("dst")
    )

    g = GraphFrame(vertices, edges)
    components = g.connectedComponents()

    return components.select(
        F.col("id").alias("group"), F.col("component").alias("merged_group")
    )


def merge_clusters(
    spark: SparkSession,
    df_clustered: DataFrame,
    split_columns: list,
    minimum_members: int = 10,
    overlap_threshold: float = 0.5,
    total_threshold: float = 0.1,
) -> DataFrame:
    """Merge clusters to remove artificial effects of splitting.

    Args:
        spark (SparkSession): Spark session.
        df_clustered (DataFrame): Dataframe of clustered data, with each row assigned to a cluster in column 'group'.
        split_columns (list): List of columns to cluster/split by.
        minimum_members (int, optional): Minimum members in overlap to allow merge. Defaults to 10.
        overlap_threshold (float, optional): Fraction of mutual members within the overlap region required to allow merge.. Defaults to 0.5.
        total_threshold (float, optional): Fraction of mutual members within the whole cluster to allow merge. Defaults to 0.1.

    Returns:
        DataFrame: Data with merged clusters.
    """
    cluster_bounds = get_cluster_bounds(df_clustered, split_columns)
    df_overlap_pairs = find_overlapping_pairs(cluster_bounds, split_columns)

    merge_pairs = get_cluster_oob_matches(
        spark,
        df_clustered,
        df_overlap_pairs,
        split_columns,
        minimum_members,
        overlap_threshold,
        total_threshold,
    )

    merged_clusters = chain_merge_clusters(merge_pairs)

    df_with_merged = df_clustered.join(
        merged_clusters, on="group", how="left"
    ).withColumn("merged_group", F.coalesce(F.col("merged_group"), F.col("group")))

    df_with_merged = df_with_merged.withColumn(
        "group", F.col("merged_group").cast("int")
    ).drop("merged_group")

    return df_with_merged
