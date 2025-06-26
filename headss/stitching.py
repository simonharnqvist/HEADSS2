import pandas as pd
import numpy as np
from typing import List, Union, Any

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

def get_all_centers(regions: List[pd.DataFrame], split_columns: List[str], n_regions: int) -> np.ndarray:
    """Aggregate cluster centers for all regions."""
    centers = np.empty(n_regions, dtype=object)
    for i, region_df in enumerate(regions):
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
    regions: List[pd.DataFrame],
    centers: np.ndarray,
    stitch_regions: pd.DataFrame,
    split_columns: List[str]
) -> pd.DataFrame:
    """Combine all regions and remove clusters outside stitching bounds."""
    valid_clusters = cut_misplaced_clusters(centers, stitch_regions, split_columns)
    combined = pd.concat(regions, ignore_index=True)
    return combined[combined['group'].isin(valid_clusters['group'])]

def stitch(
    regions: List[pd.DataFrame],
    split_columns: List[str],
    stitch_regions: pd.DataFrame
) -> pd.DataFrame:
    """Main stitching pipeline."""
    n_regions = len(regions)
    centers = get_all_centers(regions, split_columns, n_regions)
    return stitch_clusters(regions, centers, stitch_regions, split_columns)
