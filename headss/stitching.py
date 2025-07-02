import pandas as pd
import numpy as np
from typing import List, Union, Any

def calculate_centers(clustered_data: pd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
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
        raise IndexError("Column 'group' not in dataframe")    

    # Drop NA values in group column to avoid sort issues
    groups = clustered_data['group'].dropna().unique()
    records = []

    for group in sorted(groups):
        group_data = clustered_data[clustered_data['group'] == group]
        if group_data.empty:
            continue

        center = group_data[split_columns].median(numeric_only=True)
        center_df = pd.DataFrame([center])
        center_df['N'] = len(group_data)
        center_df['group'] = group
        records.append(center_df)

    if not records:
        return pd.DataFrame(columns=split_columns + ['N', 'group'])

    return pd.concat(records)


def get_centers(clustered_data: pd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
    """
    Compute the center of each cluster using group-wise median and attach group labels.

    :param clustered_data: Dask DataFrame with a 'group' column.
    :param split_columns: Columns over which to compute median centers.
    :return: dd DataFrame with one row per group, including group ID and center coords.
    """

    return [
        calculate_centers(
            clustered_data = group.copy(),
            split_columns=split_columns
        )
        for i, (_, group) in enumerate(clustered_data.groupby('region'))
    ]

def cut_misplaced_clusters(
    centers: List[pd.DataFrame],
    stitch_regions: pd.DataFrame,
    split_columns: List[str]
) -> pd.DataFrame:
    """
    Drop clusters whose centers occupy the incorrect region defined by 
        stitching_regions.
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
    regions: pd.DataFrame,
    centers: pd.DataFrame,
    stitch_regions: pd.DataFrame,
    split_columns: List[str]
) -> pd.DataFrame:
    """Filter regions to include only valid clusters based on their center positions."""
    valid_clusters = cut_misplaced_clusters(centers, stitch_regions, split_columns)
    valid_group_ids = valid_clusters["group"].dropna().unique().tolist()

    # Filter the Dask DataFrame based on valid group labels
    return regions[regions["group"].isin(valid_group_ids)]

def stitch(
    clustered_data: pd.DataFrame,
    split_columns: List[str],
    stitch_regions: pd.DataFrame
) -> pd.DataFrame:
    """Main stitching pipeline with Dask support."""
    centers = get_centers(clustered_data, split_columns)
    return stitch_clusters(clustered_data, centers, stitch_regions, split_columns)