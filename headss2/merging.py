import pandas as pd
from typing import List, Tuple
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def describe_clusters(clustered: pd.DataFrame, split_columns: List[str], group_col:str = 'group') -> List[int | pd.DataFrame]:
    groups = clustered[group_col].unique()
    cluster_info = np.zeros(int(max(groups)+1), dtype = object)
    for index, group in enumerate(groups):
        tmp = clustered[clustered[group_col] == group]
        cluster_info[group] = tmp.describe()[split_columns]\
                                        .loc[['count', 'min', 'max']]
    return cluster_info

def find_overlapping_clusters(cluster_descriptions: List[int | pd.DataFrame], split_columns: List[str]):
    bounds = []
    valid_indices = []

    # Preprocess: Extract bounds into arrays
    for idx, desc in enumerate(cluster_descriptions):
        if isinstance(desc, pd.DataFrame):
            min_vals = desc.loc['min', split_columns].values
            max_vals = desc.loc['max', split_columns].values
            bounds.append((min_vals, max_vals))
            valid_indices.append(idx)

    # Compare bounds using vectorized logic
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
    for i, value in enumerate(limits1): # Iterate over all axis
        if value != limits2[i]: # Only check overlapping axis
    #             Limit members to overlapping region
            cluster2 = cluster2[cluster2[split_columns[i]]>float(limits1[i][0])]
            cluster1 = cluster1[cluster1[split_columns[i]]<float(limits2[i][1])]
    if len(cluster1) <= minimum_members or len(cluster2) <= minimum_members:
        return 0, [0,0]
    # merge overlapping region to check fractional matches
    merged = cluster1.merge(cluster2, how = 'inner', on = split_columns)
    return merged.shape[0],[merged.shape[0]/cluster1.shape[0], \
               merged.shape[0]/cluster2.shape[0]]

def should_merge(cluster_indices: Tuple[int], clustered: pd.DataFrame, split_regions: pd.DataFrame, 
                 split_columns: List[str], minimum_members: int, overlap_threshold: float, total_threshold: float):
    """Assess whether two clusters should merge"""
    i, j = cluster_indices
    tmp1 = clustered[clustered.group == i]
    tmp2 = clustered[clustered.group == j]
    if len(tmp1) == 0 or len(tmp2) == 0:
        return None

    N_merged, perc_merged = get_cluster_oob_matches(
        clustered=clustered,
        split_regions=split_regions,
        split_columns=split_columns,
        cluster_indices=[i, j],
        minimum_members=minimum_members
    )
    perc_overlap = [N_merged / len(tmp1), N_merged / len(tmp2)]
    if max(perc_merged) > overlap_threshold and max(perc_overlap) > total_threshold:
        return [max(i, j), min(i, j)]
    return None


def check_merge_branches(cluster_merge):
    '''Ensures all clusters merge to the final cluster in a chain. Without this branched 
    chains can have two final nodes which do not merge'''
    record = []; add = []
    for i,j in cluster_merge.values:
        # Avoids repeating checks
        if j not in record:
            record.append(j) # Update checked list
            tmp = cluster_merge.loc[cluster_merge.group2 == j] # Get list of joined groups
            if tmp.shape[0]>1: # If potential chain is identified
                # Get unique gorups
                uni = np.unique(np.hstack((tmp['group1']\
                                                .unique(),(tmp['group2'].unique()))))
                for i in uni:
                    # Avoid redunency only check larger cluster numbers.
                    for j in uni[uni>i]: 
                        if i!=j: # Do not merge cluster to itself
                            add.append([max(i,j), min(i,j)])
    add = pd.DataFrame(add, columns = ['group1', 'group2']).drop_duplicates()
    return cluster_merge.merge(add, how = 'outer')

def check_cluster_merge(clustered: pd.DataFrame, 
                        matches: pd.DataFrame, 
                        split_regions: pd.DataFrame, 
                        split_columns: pd.DataFrame, 
                        minimum_members: float, 
                        overlap_threshold: float, 
                        total_threshold: float) -> pd.DataFrame:

    pairs = matches.values.tolist()
    cluster_merge = []

    partial_should_merge = partial(
        should_merge,
        clustered=clustered,
        split_regions=split_regions,
        split_columns=split_columns,
        minimum_members=minimum_members,
        overlap_threshold=overlap_threshold,
        total_threshold=total_threshold
)

    pairs = matches.values.tolist()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial_should_merge, pairs))

    results = [res for res in results if res]

    cluster_merge_df = pd.DataFrame(results, columns=['group1', 'group2']).drop_duplicates()
    return check_merge_branches(cluster_merge_df)


def merge_overlapping_clusters(clustered: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    '''iterate over merge list until cluster numbers stabalise'''
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
                   total_threshold: float = 0.1) -> pd.DataFrame:
    
    cluster_info = describe_clusters(clustered=clustered, group_col=group_col, split_columns=split_columns)
    matches = find_overlapping_clusters(cluster_descriptions = cluster_info[:], split_columns=split_columns)
    cluster_merges = check_cluster_merge(clustered = clustered, matches = matches, 
                                            split_regions = split_regions, 
                                            split_columns = split_columns, 
                                            minimum_members=minimum_members, 
                                            overlap_threshold=overlap_threshold, 
                                            total_threshold=total_threshold)
        
    return merge_overlapping_clusters(clustered=clustered, merges = cluster_merges).drop_duplicates()



