import pandas as pd
from hdbscan import HDBSCAN
import numpy as np
from typing import List

def run_hdbscan(df: pd.DataFrame,
                min_cluster_size: int, min_samples: int, allow_single_cluster: bool, 
                cluster_method: str, cluster_columns: List[str], drop_ungrouped: bool = True,
                group_offset = 0, random_seed:int = 11) -> pd.DataFrame:
    """Cluster objects and format the results into a single dataframe.

    Args:
        df (pd.DataFrame): Dataset to cluster.
        min_cluster_size (int): Minimum permitted cluster size.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        allow_single_cluster (bool): Whether to allow HDBSCAN to return a single cluster.
        cluster_method (str): Method to select clusters; either "eom" (Excess of mass) or "leaf".
        cluster_columns (List[str]): Columns to use in clustering.
        drop_ungrouped (bool, optional): Whether to drop points that have not been included in a cluster. Defaults to True.
        group_offset (int, optional): Offset number of starting cluster. Defaults to 0.
        random_seed (int, optional): Random seed. Defaults to 11.

    Returns:
        pd.DataFrame: Clustered dataset.
    """
    #"""Cluster objects and format the results into a single dataframe."""
    np.random.seed(11)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=cluster_method,
        gen_min_span_tree=False
    ).fit(df[cluster_columns])
    
    labels = clusterer.labels_
    unique_labels = sorted(set(labels) - {-1})
    label_map = {label: group_offset + i for i, label in enumerate(unique_labels)}

    df.loc[:, 'group'] = [label_map[label] if label != -1 else -1 for label in labels]

    if drop_ungrouped:
        df = df[df.group != -1]

    return df, len(unique_labels)

def cluster(split_data: pd.DataFrame, 
            min_cluster_size: int, min_samples: int, allow_single_cluster: bool, 
            cluster_method: str, cluster_columns: List[str], drop_ungrouped: bool = True) -> pd.DataFrame:
    """Perform clustering with HDBSCAN per region, assigning globally unique group IDs.

    Args:
        split_data (pd.DataFrame): Dataset to cluster (split_data from Regions).
        min_cluster_size (int): Minimum permitted cluster size.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        allow_single_cluster (bool): Whether to allow HDBSCAN to return a single cluster.
        cluster_method (str): Method to select clusters; either "eom" (Excess of mass) or "leaf".
        cluster_columns (List[str]): Columns to use in clustering.
        drop_ungrouped (bool, optional): Whether to drop points that have not been included in a cluster. Defaults to True.

    Returns:
        pd.DataFrame: Clustered data.
    """    
    group_offset = 0
    clustered_frames = []

    for _, group in split_data.groupby('region'):
        clustered_df, num_clusters = run_hdbscan(
            df=group.copy(),
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=allow_single_cluster,
            cluster_method=cluster_method,
            cluster_columns=cluster_columns,
            drop_ungrouped=drop_ungrouped,
            group_offset=group_offset
        )
        clustered_frames.append(clustered_df)
        group_offset += num_clusters

    result_df = pd.concat(clustered_frames, ignore_index=True)

    return result_df