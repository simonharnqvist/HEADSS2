import pytest
import pandas as pd
from headss2 import merging


@pytest.fixture
def t48k_clustered():
    return pd.read_csv("tests/ground_truth/t48k_clustered.csv").drop(columns=["Unnamed: 0"])

@pytest.fixture
def t48k_split_columns():
    return ['x', 'y']

@pytest.fixture
def t48k_split_regions():
    return pd.read_csv("tests/ground_truth/t48k_split_regions.csv").drop(columns=["Unnamed: 0"])

@pytest.fixture
def t48k_cluster_descriptions():
    # Sample summary data
    summary_data = pd.DataFrame({
        "cluster": [2, 2, 3, 3, 8, 8, 17, 17, 20, 20, 28, 28, 29, 29, 30, 30, 35, 35, 39, 39],
        "column": ["x", "y"] * 10,
        "count": [35.0, 35.0, 2434.0, 2434.0, 3125.0, 3125.0, 956.0, 956.0, 705.0, 705.0,
                621.0, 621.0, 43.0, 43.0, 2113.0, 2113.0, 32.0, 32.0, 652.0, 652.0],
        "min": [33.784000, 59.237999, 32.777000, 42.812000, 64.861000, 106.042000,
                261.179993, 38.352001, 341.269012, 96.347000, 324.967987, 232.867004,
                586.687988, 61.516998, 460.355988, 37.678001, 463.545990, 159.522003,
                324.967987, 232.867004],
        "max": [45.984001, 114.748001, 240.164001, 223.873001, 318.402008, 303.213989,
                440.354004, 160.807007, 432.454010, 245.923004, 479.851990, 287.210999,
                619.125000, 164.901001, 581.617004, 245.738007, 478.403015, 220.684006,
                497.927002, 290.294006]
    })

    # Group by cluster and pivot into the desired format
    cluster_tuples = []
    for cluster_id, group_df in summary_data.groupby("cluster"):
        df = pd.DataFrame({
            col: {
                "count": group_df.loc[group_df["column"] == col, "count"].values[0],
                "min": group_df.loc[group_df["column"] == col, "min"].values[0],
                "max": group_df.loc[group_df["column"] == col, "max"].values[0]
            }
            for col in group_df["column"]
        })
        cluster_tuples.append((cluster_id, df))
    
    return cluster_tuples

@pytest.fixture
def t48k_overlapping_clusters():
    return pd.read_csv("tests/ground_truth/t48k_matches.csv").drop(columns=["Unnamed: 0"])

@pytest.fixture
def t48k_cluster_oob_info():
    return pd.read_csv("tests/ground_truth/t48k_cluster_oob_info.csv").drop(columns=["Unnamed: 0"])

@pytest.fixture
def t48k_merged_clusters():
    return pd.read_csv("tests/ground_truth/t48k_merged_clusters.csv").drop(columns=["Unnamed: 0"])


def test_describe_clusters(t48k_clustered, t48k_cluster_descriptions):
    actual = merging.describe_clusters(t48k_clustered, split_columns=["x", "y"])
    expected = t48k_cluster_descriptions

    assert len(expected) == len(actual), "Mismatch in number of clusters"

    # Sort both lists by cluster ID to ensure consistent order
    expected_sorted = sorted(expected, key=lambda x: x[0])
    generated_sorted = sorted(actual, key=lambda x: x[0])

    for (expected_id, expected_df), (generated_id, generated_df) in zip(expected_sorted, generated_sorted):
        assert expected_id == generated_id, f"Cluster ID mismatch: {expected_id} != {generated_id}"
        pd.testing.assert_frame_equal(expected_df, generated_df, check_dtype=False, check_exact=False)

def test_find_overlapping_clusters(t48k_overlapping_clusters, 
                                   t48k_cluster_descriptions,
                                   t48k_split_columns):
    actual = merging.find_overlapping_clusters(cluster_descriptions = t48k_cluster_descriptions, split_columns=t48k_split_columns)
    expected = t48k_overlapping_clusters

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_get_cluster_oob_info(t48k_clustered, t48k_cluster_oob_info, t48k_overlapping_clusters, t48k_split_columns, t48k_split_regions):
    oob_info = merging.get_cluster_oob_info(clustered = t48k_clustered, split_regions = t48k_split_regions, 
                         split_columns = t48k_split_columns, cluster_index = t48k_overlapping_clusters.values[0][1])
    
    actual = oob_info[0].reset_index(drop = True)
    expected = t48k_cluster_oob_info

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

    assert oob_info[1] == [[14.642, 324.79948499999995], [21.381001, 171.12749699999998]]



def test_get_cluster_oob_matches(t48k_clustered, t48k_split_regions, t48k_split_columns, t48k_overlapping_clusters):


    actual = merging.get_cluster_oob_matches(clustered = t48k_clustered, split_regions=t48k_split_regions, split_columns=t48k_split_columns, 
                                     cluster_indices = [val for val in t48k_overlapping_clusters.values[0]], 
                                     minimum_members = 10)

    expected = (10, [0.2857142857142857, 0.004108463434675432])

    assert actual == expected

def test_check_merge_branches():
    dummy_cluster_merge = pd.DataFrame(
    {
    "group1": [1, 2, 3],
    "group2": [3, 1, 2]
    }
)
    actual = merging.check_merge_branches(dummy_cluster_merge)
    expected = dummy_cluster_merge # should remain unchanged

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_check_cluster_merge(t48k_clustered, t48k_overlapping_clusters, t48k_split_columns, t48k_split_regions):
    actual = merging.check_cluster_merge(clustered = t48k_clustered, 
                                         matches = t48k_overlapping_clusters, 
                                         split_regions=t48k_split_regions,
                                         split_columns=t48k_split_columns,
                                         minimum_members=10,
                                         overlap_threshold=0.5, total_threshold=0.1)
    assert len(actual) == 0

def test_merge_overlapping_clusters(t48k_overlapping_clusters, t48k_clustered, t48k_merged_clusters, t48k_split_columns, t48k_split_regions):
    merges = merging.check_cluster_merge(clustered = t48k_clustered, 
                                         matches = t48k_overlapping_clusters, 
                                         split_regions=t48k_split_regions,
                                         split_columns=t48k_split_columns,
                                         minimum_members=10,
                                         overlap_threshold=0.5, total_threshold=0.1)
    actual = merging.merge_overlapping_clusters(t48k_clustered, merges)
    expected = t48k_merged_clusters


    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )