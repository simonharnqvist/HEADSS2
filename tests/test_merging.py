import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark import sql
import pytest
from headss2.merging import (
    _get_cluster_bounds,
    _find_overlapping_pairs,
    _total_point_overlap,
    _should_merge,
    _assign_new_clusters,
    _merge_clusters_union_find
)
from headss2.union_find import UnionFind

@pytest.fixture
def unified_cluster_data(spark):
    """
    Creates a unified dataset with:
    - 3 clusters: cluster 1 and 2 partially overlap in bounds, 3 does not.
    - Region is constant (0) for simplicity.
    """
    data = [
        # Cluster 1
        [1.0, 1.0, 0, "1"],
        [2.0, 2.0, 0, "1"],
        [3.0, 3.0, 0, "1"],

        # Cluster 2 — partial overlap with Cluster 1
        [2.1, 2.1, 0, "2"],
        [2.5, 2.5, 0, "2"],
        [4.0, 4.0, 0, "2"],

        # Cluster 3 — no overlap
        [10.0, 10.0, 0, "3"],
        [11.0, 11.0, 0, "3"],
        [12.0, 12.0, 0, "3"],
    ]

    columns = ["x", "y", "region", "cluster"]
    df = spark.createDataFrame(pd.DataFrame(data, columns=columns))
    return df

@pytest.fixture
def cluster_bounds(unified_cluster_data):
    return _get_cluster_bounds(unified_cluster_data, ["x", "y"])

@pytest.fixture
def overlapping_pairs(cluster_bounds):
    return _find_overlapping_pairs(cluster_bounds, ["x", "y"])

@pytest.fixture
def dummy_overlap_stats():
    return pd.DataFrame([
        ["0_1", "0_2", 0.5, 1.0, 3],
        ["0_1", "0_3", 0.6, 0.01, 4],
        ["2_1", "2_2", 0.1, 0.9, 5],
        ["2_2", "2_3", 0.4, 0.4, 6]],
        columns=["cluster1", "cluster2", "total_point_overlap", "bound_region_point_overlap", "n_overlap"]
    )

def test_get_cluster_bounds(unified_cluster_data):
    result = _get_cluster_bounds(unified_cluster_data, cluster_columns=["x", "y"]).toPandas()

    expected = pd.DataFrame({
        "cluster": ["1", "2", "3"],
        "x_min": [1.0, 2.1, 10.0],
        "y_min": [1.0, 2.1, 10.0],
        "x_max": [3.0, 4.0, 12.0],
        "y_max": [3.0, 4.0, 12.0],
    })

    pd.testing.assert_frame_equal(
        result.sort_values("cluster").reset_index(drop=True),
        expected.sort_values("cluster").reset_index(drop=True),
        check_dtype=True,
    )

def test_find_overlapping_pairs(overlapping_pairs):
    expected = pd.DataFrame({
        "cluster1": ["2"],
        "cluster2": ["1"],
    })

    pd.testing.assert_frame_equal(
        overlapping_pairs.sort_values(by=["cluster1", "cluster2"]).reset_index(drop=True),
        expected.sort_values(by=["cluster1", "cluster2"]).reset_index(drop=True),
    )

def test_total_point_overlap():
    n_merged = 20
    n_cluster1 = 20
    n_cluster2 = 40
    tpo_calculated = _total_point_overlap(n_merged = n_merged, n_cluster1 = n_cluster1, n_cluster2 = n_cluster2)
    assert tpo_calculated == 1

def test_should_merge(dummy_overlap_stats):
    brpo_thresh = 0.5
    tpo_thresh = 0.1
    n_overlap_thresh = 4

    should_merge_df = _should_merge(overlap_stats_df=dummy_overlap_stats, bound_region_point_overlap_threshold=brpo_thresh, total_point_overlap_threshold=tpo_thresh, min_n_overlap=n_overlap_thresh)

    expected = [
        False, # n_overlap too low
        False, # brpo too low
        True, # all above thresh
        False, # tpo too low
    ]

    assert list(should_merge_df["should_merge"]) == expected

def test_assign_new_clusters(spark, dummy_overlap_stats):
    should_merge_df = _should_merge(overlap_stats_df=dummy_overlap_stats, bound_region_point_overlap_threshold=0.5, total_point_overlap_threshold=0.1, min_n_overlap=4)
    uf = _merge_clusters_union_find(should_merge_df)

    dummy_clustered = spark.createDataFrame(pd.DataFrame(
        [
            [1, 2, "0_1"], # x/y vals are irrelevant for this test
            [2, 3, "0_2"],
            [2, 4, "0_3"],
            [9, 2, "2_1"],
            [9, 4, "2_2"],
            [11, 6, "2_3"]
        ],
        columns=["x", "y", "cluster"]
    ))

    clustered_new = _assign_new_clusters(union_find = uf, clustered = dummy_clustered)
    clusters = list(clustered_new.toPandas()["cluster"].unique())

    assert len(clusters) == 5 # 2_1 and 2_2 should merge, all else the same

    if "2_1" in clusters:
        assert "2_2" not in clusters
    elif "2_2" in clusters:
        assert "2_1" not in clusters

    for cluster in ["0_1", "0_2", "0_3", "2_3"]:
        assert cluster in clusters
