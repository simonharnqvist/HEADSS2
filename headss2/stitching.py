from pyspark import sql
from pyspark.sql.functions import col, count, expr, lit
from typing import List
import pandas as pd
from functools import reduce
import operator
import numpy as np


def calculate_centers(
    clustered_data: pd.DataFrame, split_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate the median center and size for each cluster.

    Robust against empty partitions, missing groups, or metadata inference.

    :param data: A pandas DataFrame with a 'group' column and coordinate columns.
    :param split_columns: Names of columns to include in median calculation.
    :return: DataFrame with median center coordinates, cluster size (N), and group ID.
    """
    if clustered_data.empty:
        raise ValueError("Empty dataframe")
    if "group" not in clustered_data.columns:
        raise KeyError("Column 'group' not in dataframe")

    # Drop NA values in group column to avoid sort issues
    groups = clustered_data["group"].dropna().unique()
    records = []

    for group in sorted(groups):
        group_data = clustered_data[clustered_data["group"] == group]
        if group_data.empty:
            continue

        center = group_data[split_columns].median(numeric_only=True)
        center_df = pd.DataFrame([center])
        center_df["N"] = len(group_data)
        center_df["group"] = group
        records.append(center_df)

    if not records:
        return pd.DataFrame(columns=split_columns + ["N", "group"])

    res = pd.concat(records)

    if "N" not in res.columns:
        raise KeyError("Column 'N' not in dataframe")

    return res


def get_centers(
    clustered: sql.DataFrame, split_columns: List[str]
) -> List[pd.DataFrame]:
    """
    Compute the median center and size of each cluster, per region.
    Returns a list of Spark DataFrames (one per region).
    Uses the Pandas-based `calculate_centers()` function internally.
    """
    if "region" not in clustered.columns:
        raise KeyError("Missing 'region' column for region-based grouping")

    if not isinstance(clustered, sql.DataFrame):
        raise ValueError(
            f"'clustered' expected to be a Spark DataFrame, found {type(clustered)}"
        )

    regions = (
        clustered.select("region").distinct().rdd.map(lambda r: r["region"]).collect()
    )

    centers_list = []

    for region in regions:
        region_df = clustered.filter(col("region") == region)
        region_pd = region_df.toPandas()
        centers_pd = calculate_centers(region_pd, split_columns=split_columns)
        centers_pd["region"] = region
        centers_list.append(centers_pd)

    return centers_list


def cut_misplaced_clusters(
    centers: list[pd.DataFrame], stitch_regions: pd.DataFrame, split_columns: List[str]
) -> pd.DataFrame:
    """
    Drop clusters whose centers occupy the incorrect region defined by
        stitching_regions.
    """

    centers_transformed: list[pd.DataFrame] = []

    for center_df in centers:
        region = int(center_df["region"].unique().item())
        stitch_region = stitch_regions[stitch_regions["region"] == region]

        center_transformed: pd.DataFrame = pd.DataFrame(
            center_df[
                np.all(
                    [
                        (
                            center_df.loc[:, col_name].between(
                                float(stitch_region[f"{col_name}_min"].item()),
                                float(stitch_region[f"{col_name}_max"].item()),
                            )
                        )
                        for i, col_name in enumerate(split_columns)
                    ],
                    axis=0,
                )
            ]
        )
        centers_transformed.append(center_transformed)

    return pd.concat(centers_transformed)


def stitch_clusters(
    regions: sql.DataFrame,
    centers: List[pd.DataFrame],
    stitch_regions: pd.DataFrame,
    split_columns: List[str],
) -> sql.DataFrame:
    """
    Filter regions to include only valid clusters based on their center positions.
    """
    valid_clusters_df = cut_misplaced_clusters(centers, stitch_regions, split_columns)
    valid_group_ids = valid_clusters_df["group"].dropna().unique().tolist()

    return regions.filter(col("group").isin(valid_group_ids))


def stitch(
    clustered: sql.DataFrame, split_columns: List[str], stitch_regions: pd.DataFrame
) -> sql.DataFrame:
    """
    Stitch regions by removing misplaced clusters using PySpark.
    """
    if not isinstance(clustered, sql.DataFrame):
        raise ValueError(
            f"'clustered' must be a Spark dataframe, found {type(clustered)}"
        )
    centers = get_centers(clustered, split_columns)
    return stitch_clusters(clustered, centers, stitch_regions, split_columns)
