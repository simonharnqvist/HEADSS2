from pyspark import sql
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType
from functools import reduce


def compute_cluster_bounds(clustered_sdf, split_columns):
    """Compute per-cluster summary: count, min, max for each splitting column."""
    agg_exprs = (
        [F.count("*").alias("count")]
        + [F.min(col).alias(f"{col}_min") for col in split_columns]
        + [F.max(col).alias(f"{col}_max") for col in split_columns]
    )

    return clustered_sdf.groupBy("group").agg(*agg_exprs)


def find_overlapping_groups(bounds_df, split_columns):
    """Identify cluster overlaps via cross join bounding boxes."""
    a = bounds_df.alias("a")
    b = bounds_df.alias("b")

    # Build overlap condition: for any dimension, bounding ranges overlap
    conditions = [
        (F.col(f"b.{col}_min") < F.col(f"a.{col}_min"))
        & (F.col(f"a.{col}_min") < F.col(f"b.{col}_max"))
        for col in split_columns
    ]
    cond_expr = reduce(lambda x, y: x | y, conditions)

    return (
        a.crossJoin(b)
        .filter(F.col("a.group") != F.col("b.group"))
        .filter(cond_expr)
        .select(F.col("a.group").alias("group1"), F.col("b.group").alias("group2"))
    )


def compute_oob_overlap(
    spark_session,
    clustered_sdf,
    bounds_df,
    split_regions_df,
    split_columns,
    min_members,
):
    """
    Evaluate out-of-bounds overlap for candidate cluster pairs.
    Efficiently computes overlaps entirely in Spark.
    """

    # Step 1: Get overlapping group pairs (group1, group2) using your existing logic
    overlaps = find_overlapping_groups(bounds_df, split_columns)  # returns Spark DF with group1, group2

    # Step 2: Join cluster data to group1 and group2
    df1 = clustered_sdf.alias("df1")
    df2 = clustered_sdf.alias("df2")
    
    joined = (
        overlaps
        .join(df1, F.col("group1") == F.col("df1.group"))
        .join(df2, F.col("group2") == F.col("df2.group"))
    )

    # Step 3: Filter on matching split_columns (equivalent to inner join on those)
    for col in split_columns:
        joined = joined.filter(F.col(f"df1.{col}") == F.col(f"df2.{col}"))

    # Step 4: Count number of overlapping rows per (group1, group2)
    overlap_counts = (
        joined
        .groupBy("group1", "group2")
        .agg(F.count("*").alias("n_overlap"))
    )

    # Step 5: Compute group sizes (needed for fractions)
    group_sizes = (
        clustered_sdf
        .groupBy("group")
        .agg(F.count("*").alias("group_size"))
    )

    group_sizes_1 = group_sizes.withColumnRenamed("group", "group1").withColumnRenamed("group_size", "size1")
    group_sizes_2 = group_sizes.withColumnRenamed("group", "group2").withColumnRenamed("group_size", "size2")

    # Step 6: Join group sizes with overlap counts
    result = (
        overlap_counts
        .join(group_sizes_1, on="group1")
        .join(group_sizes_2, on="group2")
        .filter((F.col("size1") >= min_members) & (F.col("size2") >= min_members))
        .withColumn("frac1", F.col("n_overlap") / F.col("size1"))
        .withColumn("frac2", F.col("n_overlap") / F.col("size2"))
    )

    # Step 7: Select final columns
    return result.select("group1", "group2", "n_overlap", "frac1", "frac2")

def union_find_merge(overlap_df, overlap_thresh, total_thresh):
    """
    Gather all cluster merges where thresholds are exceeded.
    Use a simple union-find in Python for grouping.
    Returns a mapping of old_group -> new_root_group.
    """
    pairs = (
        overlap_df.filter(
            (F.col("frac1") >= overlap_thresh) & (F.col("frac2") >= total_thresh)
        )
        .select("group1", "group2")
        .collect()
    )

    parent = {}

    def find(u):
        parent.setdefault(u, u)
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru

    for r in pairs:
        union(r.group1, r.group2)

    return {g: find(g) for g in parent.keys()}


def apply_cluster_merge(spark_session: sql.SparkSession, clustered_sdf, group_map):
    """Apply cluster root mapping onto DataFrame."""
    mapping_df = spark_session.createDataFrame(
        [(k, v) for k, v in group_map.items()],
        schema=StructType(
            [
                StructField("group", IntegerType()),
                StructField("root_group", IntegerType()),
            ]
        ),
    )
    return clustered_sdf.join(mapping_df, on="group", how="left").withColumn(
        "cluster_merged", F.coalesce("root_group", "group")
    )


def merge_clusters(
    spark_session,
    clustered_sdf,
    split_regions_df,
    split_columns,
    minimum_members=10,
    overlap_threshold=0.5,
    total_threshold=0.1,
):
    """
    End-to-end Spark-based merge pipeline:
    1. Compute cluster bounds
    2. Detect overlapping cluster pairs
    3. Estimate OOB overlaps (simplified logic)
    4. Use union-find to collapse clusters
    5. Apply mapping to get final grouped clusters
    """
    bounds_df = compute_cluster_bounds(clustered_sdf, split_columns)
    overlaps_oob = compute_oob_overlap(
        spark_session=spark_session,
        clustered_sdf=clustered_sdf,
        bounds_df=bounds_df,
        split_regions_df=split_regions_df,
        split_columns=split_columns,
        min_members=minimum_members,
    )
    group_map = union_find_merge(overlaps_oob, overlap_threshold, total_threshold)
    return apply_cluster_merge(
        spark_session=spark_session, clustered_sdf=clustered_sdf, group_map=group_map
    )
