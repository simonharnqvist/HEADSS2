import pytest
import pandas as pd
from headss2.merging import describe_clusters, find_overlapping_clusters, get_cluster_oob_info, get_cluster_oob_matches


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
    return pd.read_csv("tests/ground_truth/t48k_cluster_descriptions.csv").drop(columns=["Unnamed: 0"])

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
    actual = describe_clusters(t48k_clustered, cluster_col = 'group')
    expected = t48k_cluster_descriptions

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_find_overlapping_clusters(t48k_overlapping_clusters, 
                                   t48k_cluster_descriptions,
                                   t48k_split_columns):
    actual = find_overlapping_clusters(cluster_descriptions = t48k_cluster_descriptions, split_columns=t48k_split_columns)
    expected = t48k_overlapping_clusters

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_get_cluster_oob_info(t48k_clustered, t48k_cluster_oob_info, t48k_overlapping_clusters, t48k_split_columns, t48k_split_regions):
    oob_info = get_cluster_oob_info(clustered = t48k_clustered, split_regions = t48k_split_regions, 
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


    actual = get_cluster_oob_matches(clustered = t48k_clustered, split_regions=t48k_split_regions, split_columns=t48k_split_columns, 
                                     cluster_indices = [val for val in t48k_overlapping_clusters.values[0]], 
                                     minimum_members = 10)

    expected = (10, [0.2857142857142857, 0.004108463434675432])

    assert actual == expected

def test_check_merge_branches():
    dummy_cluster_merge = pd.DataFrame(
    {
    "group1": [3, 2, 1],
    "group2": [2, 1, 3]
    }
)
    actual = check_merge_branches(dummy_cluster_merge)
    expected = dummy_cluster_merge # should remain unchanged

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_check_cluster_merge(t48k_clustered, t48k_overlapping_clusters):
    actual = check_cluster_merge(t48k_clustered, t48k_overlapping_clusters)
    assert len(actual) == 0

def test_merge_overlapping_clusters(t48k_overlapping_clusters, t48k_clustered, t48k_merged_clusters):
    merges = test_check_cluster_merge(t48k_clustered, t48k_overlapping_clusters)
    actual = merge_overlapping_clusters(t48k_clustered, merges)
    expected = t48k_merged_clusters


    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )