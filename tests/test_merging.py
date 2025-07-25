import pytest
import pandas as pd
from headss2 import merging
import numpy as np

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

    def make_desc_df(count, xmin, xmax, ymin, ymax):
        df = pd.DataFrame({
            'x': [count, xmin, xmax],
            'y': [count, ymin, ymax]
        }, index=['count', 'min', 'max'])
        return df

    arr = np.array([
        0, 0,
        make_desc_df(35.000000, 33.784000, 45.984001, 59.237999, 114.748001),
        make_desc_df(2434.000000, 32.777000, 240.164001, 42.812000, 223.873001),
        0, 0, 0, 0,
        make_desc_df(3125.000000, 64.861000, 318.402008, 106.042000, 303.213989),
        0, 0, 0, 0, 0, 0, 0, 0,
        make_desc_df(956.000000, 261.179993, 440.354004, 38.352001, 160.807007),
        0, 0,
        make_desc_df(705.000000, 341.269012, 432.454010, 96.347000, 245.923004),
        0, 0, 0, 0, 0, 0, 0,
        make_desc_df(621.000000, 324.967987, 479.851990, 232.867004, 287.210999),
        make_desc_df(43.000000, 586.687988, 619.125000, 61.516998, 164.901001),
        make_desc_df(2113.000000, 460.355988, 581.617004, 37.678001, 245.738007),
        0,
        0, 0, 0,
        make_desc_df(32.000000, 463.545990, 478.403015, 159.522003, 220.684006),
        0, 0, 0,
        make_desc_df(652.000000, 324.967987, 497.927002, 232.867004, 290.294006)
    ], dtype=object)

    return arr


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

    for idx, elem in enumerate(expected):
        if isinstance(elem, pd.DataFrame):
            pd.testing.assert_frame_equal(elem, actual[idx])
        elif elem == 0: 
            assert actual[idx] == 0, f"Expected 0 but found {actual[idx]} at idx {idx}"
        else:
            raise ValueError(f"Invalid element at index {idx}")


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
    dummy_cluster_merge = ((3, 1), (3, 2), (2, 1))
    actual = merging.check_merge_branches(dummy_cluster_merge, n_cores=4)
    expected = dummy_cluster_merge # should remain unchanged

    assert actual == expected

def test_check_cluster_merge(t48k_clustered, t48k_overlapping_clusters, t48k_split_columns, t48k_split_regions):
    actual = merging.check_cluster_merge(clustered = t48k_clustered, 
                                         matches = t48k_overlapping_clusters, 
                                         split_regions=t48k_split_regions,
                                         split_columns=t48k_split_columns,
                                         minimum_members=10,
                                         overlap_threshold=0.5, total_threshold=0.1, n_cores=4)
    assert len(actual) == 0

def test_merge_overlapping_clusters(t48k_overlapping_clusters, t48k_clustered, t48k_merged_clusters, t48k_split_columns, t48k_split_regions):
    merges = merging.check_cluster_merge(clustered = t48k_clustered, 
                                         matches = t48k_overlapping_clusters, 
                                         split_regions=t48k_split_regions,
                                         split_columns=t48k_split_columns,
                                         minimum_members=10,
                                         overlap_threshold=0.5, total_threshold=0.1, n_cores=4)
    actual = merging.merge_overlapping_clusters(t48k_clustered, merges)
    expected = t48k_merged_clusters


    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )