import pandas as pd
from typing import List, Tuple, Set
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time
import os
import itertools


# Define globals for parallelised work
_clustered = None
_split_regions = None
_split_columns = None
_minimum_members = None
_overlap_threshold = None
_total_threshold = None

def init_worker(clustered, split_regions, split_columns, minimum_members, overlap_threshold, total_threshold):
    """Initialisation function for parallelised work"""
    global _clustered, _split_regions, _split_columns
    global _minimum_members, _overlap_threshold, _total_threshold

    _clustered = clustered
    _split_regions = split_regions
    _split_columns = split_columns
    _minimum_members = minimum_members
    _overlap_threshold = overlap_threshold
    _total_threshold = total_threshold


def describe_clusters(clustered: pd.DataFrame, split_columns: List[str], group_col:str = 'group') -> List[int | pd.DataFrame]:
    """Summary statistics for each cluster."""
    groups = clustered[group_col].unique()
    cluster_info = np.zeros(int(max(groups)+1), dtype = object)
    for index, group in enumerate(groups):
        tmp = clustered[clustered[group_col] == group]
        cluster_info[group] = tmp.describe()[split_columns]\
                                        .loc[['count', 'min', 'max']]
    return cluster_info

def find_overlapping_clusters(cluster_descriptions: List[int | pd.DataFrame], split_columns: List[str]):
    """Compare bounds to find overlapping clusters."""
    bounds = []
    valid_indices = []

    for idx, desc in enumerate(cluster_descriptions):
        if isinstance(desc, pd.DataFrame):
            min_vals = desc.loc['min', split_columns].values
            max_vals = desc.loc['max', split_columns].values
            bounds.append((min_vals, max_vals))
            valid_indices.append(idx)

    matches = []
    for i, (min1, max1) in zip(valid_indices, bounds):
        for j, (min2, max2) in zip(valid_indices, bounds):
            if i == j:
                continue
            overlap = (min2 < min1) & (min1 < max2)
            if np.any(overlap):
                matches.append((i, j))

    return pd.DataFrame(matches, columns=['group1', 'group2'])

def get_cluster_oob_info(clustered: pd.DataFrame, split_regions: pd.DataFrame, 
                         split_columns: List[str], cluster_index: int) -> Tuple[pd.DataFrame, List[float]]:
        '''Returns info about the out of bounds (oob) area of the clustering region 
        i.e. the area not included in the final stitching'''
        
        cluster = clustered[clustered.group == cluster_index]
        region = cluster.region.values[0]
        limits = []
        for col in split_columns:
            limits.append([split_regions.loc[region,f'{col}_mins'], 
                      split_regions.loc[region,f'{col}_max']])
        return cluster, limits

def get_cluster_oob_matches(clustered: pd.DataFrame, split_regions: pd.DataFrame, 
                         split_columns: List[str], cluster_indices: Tuple[int, int],
                         minimum_members:int) -> Tuple[int, List[float]]:
    '''Returns points of a cluster points out of bounds (oob) of the clustering region
    i.e. the area not included in the final stitching'''

    # Isolate target clusters
    cluster1, limits1 = get_cluster_oob_info(clustered = clustered, split_regions = split_regions, split_columns=split_columns, cluster_index=cluster_indices[0])
    cluster2, limits2 = get_cluster_oob_info(clustered = clustered, split_regions = split_regions, split_columns=split_columns, cluster_index=cluster_indices[1])
    for i, value in enumerate(limits1):
        if value != limits2[i]:
            cluster2 = cluster2[cluster2[split_columns[i]]>float(limits1[i][0])]
            cluster1 = cluster1[cluster1[split_columns[i]]<float(limits2[i][1])]
    if len(cluster1) <= minimum_members or len(cluster2) <= minimum_members:
        return 0, [0,0]
    
    # merge overlapping region to check fractional matches
    merged = cluster1.merge(cluster2, how = 'inner', on = split_columns)
    return merged.shape[0],[merged.shape[0]/cluster1.shape[0], \
               merged.shape[0]/cluster2.shape[0]]


def cluster_merges_per_cluster(args: Tuple[int, List[Tuple[int]]]) -> Tuple[int]:
    """Find clusters to merge with"""
    cluster, cluster_merges = args
    subset = [pair for pair in cluster_merges if pair[1] == cluster]
    unique_clusters = list(set(val for pair in subset for val in pair))
    pairs = [(a,b) for a,b in itertools.combinations(unique_clusters, 2)]
    return pairs

def check_merge_branches(cluster_merges: List[Tuple[int]], n_cores: int) -> Tuple[int]:
    """Ensures all clusters merge to the final cluster in a chain.
    Without this, chains can have multiple terminal nodes.
    """

    assert not isinstance(cluster_merges, pd.DataFrame)

    args_list = [(cluster, cluster_merges) for cluster in cluster_merges]

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(cluster_merges_per_cluster, args_list))
        flattened_results = [item for sublist in results for item in sublist]
        return tuple(set(cluster_merges) | set(flattened_results))

def should_merge(cluster_indices: Tuple[int]) -> Tuple[int]:
    """Assess whether two clusters should merge"""
    i, j = cluster_indices
    tmp1 = _clustered[_clustered.group == i]
    tmp2 = _clustered[_clustered.group == j]
    if len(tmp1) == 0 or len(tmp2) == 0:
        return None

    N_merged, perc_merged = get_cluster_oob_matches(
        clustered=_clustered,
        split_regions=_split_regions,
        split_columns=_split_columns,
        cluster_indices=[i, j],
        minimum_members=_minimum_members
    )
    perc_overlap = [N_merged / len(tmp1), N_merged / len(tmp2)]
    if max(perc_merged) > _overlap_threshold and max(perc_overlap) > _total_threshold:
        return (max(i, j), min(i, j))
    return None

def check_cluster_merge(clustered: pd.DataFrame, 
                        matches: pd.DataFrame, 
                        split_regions: pd.DataFrame, 
                        split_columns: pd.DataFrame, 
                        minimum_members: float, 
                        overlap_threshold: float, 
                        total_threshold: float,
                        n_cores: int) -> pd.DataFrame:
    """Merge clusters in parallel."""

    pairs = matches.values.tolist()

    with ProcessPoolExecutor(
        max_workers=n_cores,
        initializer=init_worker,
        initargs=(clustered, split_regions, split_columns,
                  minimum_members, overlap_threshold, total_threshold)
    ) as executor:
        results = list(executor.map(should_merge, pairs))

    cluster_merges: List[Tuple[int]] = [res for res in results if res]

    merged = check_merge_branches(cluster_merges, n_cores=n_cores)
    return pd.DataFrame(merged, columns=["group1", "group2"])


def merge_overlapping_clusters(clustered: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    '''Iterate over merge list until cluster numbers stabilise'''
    N_clusters = 0
    N_clusters_pre = 1

    res = clustered.copy()
    
    # Iterate until all chains are followed through
    while N_clusters != N_clusters_pre:
        N_clusters_pre = len(res.group.unique())
        for k in merges.values:
            i,j = min(k), max(k)
            # Move all clusters to max label to directionise the merges.
            res.loc[res["group"] == j, 'group'] = i
        N_clusters = len(res.group.unique())
    return res

def merge_clusters(clustered: pd.DataFrame, group_col: str,
                   split_regions: pd.DataFrame, split_columns: List[str],
                   minimum_members: int = 10, overlap_threshold:float = 0.5,
                   total_threshold: float = 0.1,
                   n_cores: int = 4) -> pd.DataFrame:
    """Find clusters to merge and merge them.

    Args:
        clustered (pd.DataFrame): Clustered (and stitched, if required) data.
        group_col (str): Column name for clustering.
        split_regions (pd.DataFrame): Split regions from `regions` module.
        split_columns (List[str]): Split columns from `regions` module.
        minimum_members (int, optional): Minimum number of cluster members. Defaults to 10.
        overlap_threshold (float, optional): Threshold for overlap. Defaults to 0.5.
        total_threshold (float, optional): Total threshold. Defaults to 0.1.
        n_cores (int, optional): Number of cores; -1 = all available. Defaults to 4.

    Returns:
        pd.DataFrame: Data with merged clusters.
    """
    
    if n_cores == -1 or n_cores > os.cpu_count():
        n_cores = os.cpu_count()
    elif n_cores < 1 or not isinstance(n_cores, int):
        n_cores = 1
    
    cluster_info = describe_clusters(clustered=clustered, group_col=group_col, split_columns=split_columns)
    matches = find_overlapping_clusters(cluster_descriptions = cluster_info[:], split_columns=split_columns)
    cluster_merges = check_cluster_merge(clustered = clustered, matches = matches, 
                                            split_regions = split_regions, 
                                            split_columns = split_columns, 
                                            minimum_members=minimum_members, 
                                            overlap_threshold=overlap_threshold, 
                                            total_threshold=total_threshold,
                                            n_cores=n_cores)
        
    return merge_overlapping_clusters(clustered=clustered, merges = cluster_merges).drop_duplicates()



