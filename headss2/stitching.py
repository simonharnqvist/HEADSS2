from pyspark import sql
from pyspark.sql.functions import col, count, expr, lit
from typing import List
import pandas as pd
from functools import reduce
import operator
import numpy as np


def calculate_centers(
    clustered_data: pd.DataFrame, cluster_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate the median center and size for each cluster.

    Robust against empty partitions, missing clusters, or metadata inference.

    :param data: A pandas DataFrame with a 'cluster' column and coordinate columns.
    :param cluster_columns: Names of columns to include in median calculation.
    :return: DataFrame with median center coordinates, cluster size (N), and cluster ID.
    """
    if clustered_data.empty:
        raise ValueError("Empty dataframe")
    if "cluster" not in clustered_data.columns:
        raise KeyError("Column 'cluster' not in dataframe")

    # Drop NA values in cluster column to avoid sort issues
    clusters = clustered_data["cluster"].dropna().unique()
    records = []

    for cluster in sorted(clusters):
        cluster_data = clustered_data[clustered_data["cluster"] == cluster]
        if cluster_data.empty:
            continue

        center = cluster_data[cluster_columns].median(numeric_only=True)
        center_df = pd.DataFrame([center])
        center_df["N"] = len(cluster_data)
        center_df["cluster"] = cluster
        records.append(center_df)

    if not records:
        return pd.DataFrame(columns=cluster_columns + ["N", "cluster"])

    res = pd.concat(records)

    if "N" not in res.columns:
        raise KeyError("Column 'N' not in dataframe")

    return res


def get_centers(
    clustered: sql.DataFrame, cluster_columns: List[str]
) -> List[pd.DataFrame]:
    """
    Compute the median center and size of each cluster, per region.
    Returns a list of Spark DataFrames (one per region).
    Uses the Pandas-based `calculate_centers()` function internally.
    """
    if "region" not in clustered.columns:
        raise KeyError("Missing 'region' column for region-based clustering")

    if not isinstance(clustered, sql.DataFrame):
        raise ValueError(
            f"'clustered' expected to be a Spark DataFrame, found {type(clustered)}"
        )

    
    regions = (
        clustered.select("region").distinct().toPandas()["region"].to_list()
    )

    centers_list = []

    for region in regions:
        region_df = clustered.filter(col("region") == region)
        region_pd = region_df.toPandas()
        centers_pd = calculate_centers(region_pd, cluster_columns=cluster_columns)
        centers_pd["region"] = region
        centers_list.append(centers_pd)

    return centers_list


def cut_misplaced_clusters(
    centers: list[pd.DataFrame], stitch_regions: pd.DataFrame, cluster_columns: List[str]
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
                        for i, col_name in enumerate(cluster_columns)
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
    cluster_columns: List[str],
) -> sql.DataFrame:
    """
    Filter regions to include only valid clusters based on their center positions.
    """
    valid_clusters_df = cut_misplaced_clusters(centers, stitch_regions, cluster_columns)
    valid_cluster_ids = valid_clusters_df["cluster"].dropna().unique().tolist()

    return regions.filter(col("cluster").isin(valid_cluster_ids))


def stitch(
    clustered: sql.DataFrame, cluster_columns: List[str], stitch_regions: pd.DataFrame
) -> sql.DataFrame:
    """
    Stitch regions by removing misplaced clusters using PySpark.
    """
    if not isinstance(clustered, sql.DataFrame):
        raise ValueError(
            f"'clustered' must be a Spark dataframe, found {type(clustered)}"
        )
    
    if not cluster_columns or len(cluster_columns) < 1:
        raise ValueError("Cluster columns not provided")
    
    for col in ["region", "cluster"] + cluster_columns:
        if col not in clustered.columns:
            raise KeyError(f"Missing column '{col}'. Found {clustered.columns}")



    centers = get_centers(clustered, cluster_columns)
    return stitch_clusters(clustered, centers, stitch_regions, cluster_columns)
