import pandas as pd
import dask.dataframe as dd
from hdbscan import HDBSCAN
import numpy as np
from typing import List

def run_hdbscan(region_idx: int, df: pd.DataFrame,
                min_cluster_size: int, min_samples: int, allow_single_cluster: bool, 
                cluster_method: str, cluster_columns: List[str], drop_ungrouped: bool = True) -> pd.DataFrame:
    """Cluster objects and format the results into a single dataframe."""
    np.random.seed(11)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=cluster_method,
        gen_min_span_tree=False
    ).fit(df[cluster_columns])
    
    df.loc[:, 'group'] = [
        f"{region_idx}_{label}" if label != -1 else -1
        for label in clusterer.labels_
    ]

    if drop_ungrouped:
        df = df[df.group != -1]

    return df

def cluster(split_data: dd.DataFrame, 
            min_cluster_size: int, min_samples: int, allow_single_cluster: bool, 
            cluster_method: str, cluster_columns: List[str], drop_ungrouped: bool = True) -> dd.DataFrame:
    """Perform clustering with HDBSCAN per partition (=region), prefixing group with partition index."""

    result_df = pd.concat([
        run_hdbscan(
            region_idx=i,
            df=group.copy(),
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=allow_single_cluster,
            cluster_method=cluster_method,
            cluster_columns=cluster_columns,
            drop_ungrouped=drop_ungrouped
        )
        for i, (_, group) in enumerate(split_data.groupby('region'))
    ], ignore_index=True)

    return result_df