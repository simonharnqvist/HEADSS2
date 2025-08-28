from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, expr, lit
from typing import List
import pandas as pd
from functools import reduce
import operator


def calculate_centers_spark(
    clustered_data: DataFrame, split_columns: List[str]
) -> DataFrame:
    """
    Calculate the median center and size for each cluster in PySpark.
    """
    if "group" not in clustered_data.columns:
        raise KeyError("Column 'group' not in dataframe")

    if clustered_data.rdd.isEmpty():
        raise ValueError("Empty dataframe")

    # Approximate median using percentile_approx for each group
    exprs = [
        expr(f"percentile_approx({col_name}, 0.5) as {col_name}")
        for col_name in split_columns
    ]
    exprs.append(count("*").alias("N"))

    centers_df = clustered_data.groupBy("group").agg(*exprs)
    return centers_df


def get_centers_spark(
    clustered_data: DataFrame, split_columns: List[str]
) -> List[DataFrame]:
    """
    Compute the center of each cluster per region.
    Returns a list of Spark DataFrames (one per region).
    """
    if "region" not in clustered_data.columns:
        raise KeyError("Missing 'region' column for region-based grouping")

    regions = (
        clustered_data.select("region")
        .distinct()
        .rdd.map(lambda r: r["region"])
        .collect()
    )
    centers_list = []

    for region in regions:
        region_df = clustered_data.filter(col("region") == region)
        centers_df = calculate_centers_spark(region_df, split_columns)
        centers_df = centers_df.withColumn("region", lit(region))  # Add region back
        centers_list.append(centers_df)

    return centers_list


def cut_misplaced_clusters_spark(
    centers: List[DataFrame], stitch_regions: pd.DataFrame, split_columns: List[str]
) -> pd.DataFrame:
    """
    Drop clusters whose centers lie outside the valid bounding box from stitch_regions.
    Returns a Pandas DataFrame of valid clusters.
    """
    valid_centers = []

    for idx, center_df in enumerate(centers):
        bounds = stitch_regions.loc[idx]

        conditions = [
            (col(col_name) >= bounds[f"{col_name}_mins"])
            & (col(col_name) <= bounds[f"{col_name}_max"])
            for col_name in split_columns
        ]

        # Combine all conditions with AND
        if not conditions:
            continue
        combined_condition = reduce(operator.and_, conditions)
        filtered_df = center_df.filter(combined_condition)

        if not filtered_df.rdd.isEmpty():
            valid_centers.append(filtered_df.toPandas())

    if not valid_centers:
        return pd.DataFrame(columns=["group"] + split_columns + ["N", "region"])

    return pd.concat(valid_centers, ignore_index=True)


def stitch_clusters_spark(
    regions: DataFrame,
    centers: List[DataFrame],
    stitch_regions: pd.DataFrame,
    split_columns: List[str],
) -> DataFrame:
    """
    Filter regions to include only valid clusters based on their center positions.
    """
    valid_clusters_df = cut_misplaced_clusters_spark(
        centers, stitch_regions, split_columns
    )
    valid_group_ids = valid_clusters_df["group"].dropna().unique().tolist()

    return regions.filter(col("group").isin(valid_group_ids))


def stitch(
    clustered_data: DataFrame, split_columns: List[str], stitch_regions: pd.DataFrame
) -> DataFrame:
    """
    Stitch regions by removing misplaced clusters using PySpark.
    """
    centers = get_centers_spark(clustered_data, split_columns)
    return stitch_clusters_spark(clustered_data, centers, stitch_regions, split_columns)
