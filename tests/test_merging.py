import pytest
import pandas as pd

@pytest.fixture
def t48k_clustered():
    return pd.read_csv("tests/ground_truth/t48k_clustered.csv")

def t48k_split_columns():
    return ['x', 'y']

def t48k_split_regions():
    return pd.read_csv("tests/ground_truth/t48k_split_regions.csv")

@pytest.fixture
def t48k_cluster_descriptions():
    return pd.read_csv("tests/ground_truth/t48k_cluster_descriptions.csv")

@pytest.fixture
def t48k_overlapping_clusters():
    return pd.read_csv("tests/ground_truth/t48k_matches.csv")

@pytest.fixture
def t48k_cluster_oob_info():
    return pd.read_csv("tests/ground_truth/t48k_cluster_oob_info.csv")

@pytest.fixture
def t48k_merged_clusters():
    return pd.read_csv("tests/ground_truth/t48k_merged_clusters.csv")


def test_describe_clusters(t48k_clustered, t48k_cluster_descriptions):
    actual = describe_clusters(t48k_clustered, group_col = 'group')
    expected = t48k_cluster_descriptions()

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_find_overlapping_clusters(t48k_overlapping_clusters, 
                                   t48k_cluster_descriptions, 
                                   t48k_split_columns):
    actual = find_overlapping_clusters(t48k_split_columns, t48k_cluster_descriptions)
    expected = t48k_overlapping_clusters()

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_get_cluster_oob_info(t48k_clustered, t48k_cluster_oob_info, t48k_matches, t48k_split_columns):
    actual = get_cluster_oob_info(t48k_split_columns, t48k_clustered, index = t48k_matches.values[0][1], split_regions = t48k_split_regions)
    expected = t48k_cluster_oob_info()

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_get_cluster_oob_matches(t48k_clustered, t48k_split_regions, t48k_split_columns, t48k_matches):
    actual = get_cluster_oob_matches(t48k_clustered, t48k_split_regions, t48k_split_columns, 
                                     cluster_index = [val for val in t48k_matches.values[0]], 
                                     evaluate = "best")
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

def test_check_cluster_merge(t48k_clustered, t48k_matches):
    actual = check_cluster_merge(t48k_clustered, t48k_matches)
    assert len(actual) == 0

def test_merge_overlapping_clusters(t48k_matches, t48k_clustered, t48k_merged_clusters):
    merges = test_check_cluster_merge(t48k_clustered, t48k_matches)
    actual = merge_overlapping_clusters(t48k_clustered, merges)
    expected = t48k_merged_clusters()


    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )