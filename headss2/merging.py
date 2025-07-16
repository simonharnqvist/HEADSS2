import pandas as pd
from typing import List, Tuple
import numpy as np


def describe_clusters(clustered: pd.DataFrame, cluster_col:str = 'group') -> pd.DataFrame:
    """Generate summary statistics (min, max, size) per cluster"""

    return (
        clustered.groupby(cluster_col)[["x", "y"]]
        .agg(["min", "max", "count"])
        .set_axis([f"{col}_{stat}" for col, stat in clustered.groupby(cluster_col)[["x", "y"]]
                   .agg(["min", "max", "count"]).columns], axis=1)
        .reset_index()
        .melt(id_vars=cluster_col, var_name="column_stat", value_name="value")
        .assign(
            column=lambda df: df["column_stat"].str.split("_").str[0],
            stat=lambda df: df["column_stat"].str.split("_").str[1]
        )
        .pivot_table(index=[cluster_col, "column"], columns="stat", values="value")
        .reset_index()[[cluster_col, "column", "count", "min", "max"]]
        .rename_axis(None, axis=1)
    )

def find_overlapping_clusters(cluster_descriptions: pd.DataFrame, split_columns: list) -> pd.DataFrame:
    """Detects one-directional overlaps between cluster pairs across specified axes."""

    all_matches = []

    for col in split_columns:
        axis_df = cluster_descriptions[cluster_descriptions["column"] == col]

        # Cartesian product: merge clusters on current axis
        pairs = axis_df.merge(axis_df, on="column", suffixes=("_1", "_2"))

        # Keep only directional (non-self) pairs
        pairs = pairs[pairs["group_1"] != pairs["group_2"]]

        # Bounding box overlap condition
        overlap = (
            (pairs["min_2"] < pairs["min_1"]) &
            (pairs["max_2"] > pairs["min_1"])
        )

        # Extract directional overlaps only (left → right)
        matched = pairs.loc[overlap, ["group_1", "group_2"]]
        matched.columns = ["group1", "group2"]
        all_matches.append(matched)

    # Combine all matches and remove exact duplicates (not reversed ones)
    result = pd.concat(all_matches, ignore_index=True).drop_duplicates()

    # Sort for consistent output style
    result = result.sort_values(by=["group1", "group2"]).reset_index(drop=True)
    return result

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

def check_cluster_merge(clustered: pd.DataFrame, matches: pd.DataFrame, split_regions: pd.DataFrame, split_columns: pd.DataFrame, minimum_members: float, overlap_threshold: float, total_threshold: float) -> pd.DataFrame:
    '''Checks if two clusters should merge'''

    cluster_merge = []
    for (i,j) in matches[:].values:
        tmp1 = clustered[clustered.group == i]
        tmp2 = clustered[clustered.group == j]
        N_merged, perc_merged = get_cluster_oob_matches(clustered=clustered, split_regions=split_regions, split_columns=split_columns, cluster_indices = [i,j], minimum_members=minimum_members)
        if len(tmp1) == 0 or len(tmp2) == 0:
            continue
        else:
            perc_overlap = [N_merged/len(tmp1),N_merged/len(tmp2)]
            if max(perc_merged) > overlap_threshold and \
                max(perc_overlap) > total_threshold:
                    cluster_merge.append([max(i,j), min(i,j)])
    cluster_merge = pd.DataFrame(cluster_merge, columns = ['group1', 'group2'])\
                                    .drop_duplicates()
    return check_merge_branches(cluster_merge = cluster_merge)

def merge_overlapping_clusters(clustered: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    '''iterate over merge list until cluster numbers stabalise'''
    N_clusters = 0
    N_clusters_pre = 1

    res = clustered.copy()
    
    # Iterate until all chains are followed through
    while N_clusters != N_clusters_pre:
        N_clusters_pre = len(clustered.group.unique())
        for k in merges.values:
            i,j = min(k), max(k)
            # Move all clusters to max label to directionise the merges.
            res.loc[res["group"] == j, 'group'] = i
        N_clusters = len(res.group.unique())
    return res

def merge_clusters(clustered: pd.DataFrame, cluster_col: str,
                   split_regions: pd.DataFrame, split_columns: List[str],
                   minimum_members: int = 10, overlap_threshold:float = 0.5,
                   total_threshold: float = 0.1) -> pd.DataFrame:
    cluster_info = describe_clusters(clustered=clustered, cluster_col=cluster_col)
    matches = find_overlapping_clusters(cluster_info = cluster_info[:], split_columns=split_columns)
    cluster_merges = check_cluster_merge(clustered = clustered, matches = matches, 
                                         split_regions = split_regions, 
                                         split_columns = split_columns, 
                                         minimum_members=minimum_members, 
                                         overlap_threshold=overlap_threshold, 
                                         total_threshold=total_threshold)
    return merge_overlapping_clusters(clustered=clustered, merge_list = cluster_merges).drop_duplicates()



