import pandas as pd
import numpy as np
from typing import List, Union, Any
import dask.dataframe as dd

def calculate_centers(data: pd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
    """
    Calculate the median center and size for each cluster.

    Robust against empty partitions, missing groups, or metadata inference.

    :param data: A pandas DataFrame with a 'group' column and coordinate columns.
    :param split_columns: Names of columns to include in median calculation.
    :return: DataFrame with median center coordinates, cluster size (N), and group ID.
    """
    if data.empty or 'group' not in data.columns:
        # Return a well-formed but empty DataFrame to support Dask's emulation
        return pd.DataFrame(columns=split_columns + ['N', 'group'])

    # Drop NA values in group column to avoid sort issues
    groups = data['group'].dropna().unique()
    records = []

    for group in sorted(groups):
        group_data = data[data['group'] == group]
        if group_data.empty:
            continue

        center = group_data[split_columns].median(numeric_only=True)
        center['N'] = len(group_data)
        center['group'] = group
        records.append(center)

    if not records:
        return pd.DataFrame(columns=split_columns + ['N', 'group'])

    return pd.DataFrame(records)


def get_all_centers(regions: dd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
    """
    Compute the center of each cluster using group-wise median and attach group labels.

    :param regions: Dask DataFrame with a 'group' column.
    :param split_columns: Columns over which to compute median centers.
    :return: pandas DataFrame with one row per group, including group ID and center coords.
    """
    def calculate_centers_partition(df: pd.DataFrame) -> pd.DataFrame:
        return calculate_centers(df, split_columns)

    centers_dd = regions.map_partitions(calculate_centers_partition)
    return centers_dd.compute()


def cut_misplaced_clusters(
    centers: pd.DataFrame,
    stitch_regions: pd.DataFrame,
    split_columns: List[str]
) -> pd.DataFrame:
    """
    Filter cluster centers whose coordinates fall outside all stitch regions.

    Each center (row in `centers`) is retained only if it lies within at least one
    bounding box defined in `stitch_regions`.

    :param centers: DataFrame of cluster centers with coordinate columns.
    :param stitch_regions: DataFrame with min/max bounds per dimension.
    :param split_columns: List of dimensions (e.g. ['x', 'y', 'z']).
    :return: DataFrame of valid centers.
    """
    res = pd.DataFrame()
    for index, center in enumerate(centers):
        # Iterate over all centers to check it lies within the stitching map.
        center = center[np.all([(center[col].between(
                            stitch_regions.loc[index][f'{col}_mins'], 
                                stitch_regions.loc[index][f'{col}_max']))
                                    for i, col in enumerate(split_columns)], 
                                        axis = 0)]
        res = pd.concat([res,center], ignore_index = True)
        return res

def stitch_clusters(
    regions: dd.DataFrame,
    centers: pd.DataFrame,
    stitch_regions: pd.DataFrame,
    split_columns: List[str]
) -> dd.DataFrame:
    """Filter regions to include only valid clusters based on their center positions."""
    valid_clusters = cut_misplaced_clusters(centers, stitch_regions, split_columns)
    valid_group_ids = valid_clusters["group"].dropna().unique().tolist()

    # Filter the Dask DataFrame based on valid group labels
    return regions[regions["group"].isin(valid_group_ids)]

def stitch(
    regions: dd.DataFrame,
    split_columns: List[str],
    stitch_regions: pd.DataFrame
) -> dd.DataFrame:
    """Main stitching pipeline with Dask support."""
    centers = get_all_centers(regions, split_columns)
    return stitch_clusters(regions, centers, stitch_regions, split_columns)