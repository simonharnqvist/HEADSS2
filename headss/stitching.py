import pandas as pd
import numpy as np
from typing import List, Union, Any
import dask.dataframe as dd

def calculate_centers(data: pd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
    """Calculate the median center and size for each cluster."""
    groups = data['group'].unique()
    records = []
    for group in sorted(groups):
        tmp = data[data.group == group]
        center = tmp[split_columns].median()
        center['N'] = tmp.shape[0]
        center['group'] = group
        records.append(center)
    return pd.DataFrame(records)

def get_all_centers(regions: dd.DataFrame, split_columns: List[str], n_regions: int) -> np.ndarray:
    centers = np.empty(n_regions, dtype=object)

    # Pull grouped region data into Pandas
    regions_pd = regions.groupby('region').apply(lambda df: df, meta=regions)._meta_nonempty
    grouped = regions.groupby('region').apply(lambda x: x, meta=regions_pd).compute()

    for i, (_, region_df) in enumerate(grouped.groupby('region')):
        centers[i] = calculate_centers(region_df, split_columns)

    return centers

def cut_misplaced_clusters(centers: np.ndarray, stitch_regions: pd.DataFrame, split_columns: List[str]) -> pd.DataFrame:
    """Filter clusters whose centers lie outside their expected stitch region."""
    valid_centers = []
    for i, center_df in enumerate(centers):
        bounds = [
            center_df[col].between(
                stitch_regions.loc[i, f"{col}_mins"],
                stitch_regions.loc[i, f"{col}_max"]
            ) for col in split_columns
        ]
        mask = np.logical_and.reduce(bounds)
        valid_centers.append(center_df[mask])
    return pd.concat(valid_centers, ignore_index=True)

def stitch_clusters(
    regions: dd.DataFrame,
    centers: np.ndarray,
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
    n_regions = regions.npartitions
    centers = get_all_centers(regions, split_columns, n_regions)
    return stitch_clusters(regions, centers, stitch_regions, split_columns)