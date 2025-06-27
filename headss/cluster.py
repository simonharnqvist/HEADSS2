import pandas as pd
import dask.dataframe as dd
from hdbscan import HDBSCAN

def run_hdbscan(df: pd.DataFrame,
            min_cluster_size:int, min_samples:int, allow_single_cluster:bool, 
            cluster_method: str, cluster_columns:"list[str]", drop_ungrouped:bool = True) -> pd.DataFrame:
    """ Cluster objects and format the results into a single dataframe."""

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                prediction_data=False,
                                allow_single_cluster=allow_single_cluster,
                                cluster_selection_method=cluster_method,
                                gen_min_span_tree=False).fit(df[cluster_columns])
    
    df.loc[:, 'group'] = clusterer.labels_ + df.index.astype("Int64")

    if drop_ungrouped:
        df = df.loc[df.group!=-1, :]

    return df

def cluster(region_partitioned_split_data: dd.DataFrame, min_cluster_size:int, min_samples:int, allow_single_cluster:bool, 
            cluster_method: str, cluster_columns:"list[str]", drop_ungrouped:bool = True):
    """Perform clustering with HDBSCAN per partition (=region)

    Args:
        region_partitioned_split_data (dd.DataFrame): _description_
        min_cluster_size (int): _description_
        min_samples (int): _description_
        allow_single_cluster (bool): _description_
        cluster_method (str): _description_
        cluster_columns (list[str]): _description_
        drop_ungrouped (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    
    return region_partitioned_split_data.map_partitions(run_hdbscan, 
                                                 min_cluster_size = min_cluster_size, 
                                                 min_samples = min_samples, 
                                                 allow_single_cluster = allow_single_cluster,
                                                 cluster_method = cluster_method, 
                                                 cluster_columns = cluster_columns, 
                                                 drop_ungrouped = drop_ungrouped)


