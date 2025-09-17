import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark import sql
import pytest
from headss2.merging import (
    _get_cluster_bounds,
    _find_overlapping_pairs,
    _get_n_overlaps,
    _apply_get_n_overlaps,
    _assign_new_clusters,
    _calculate_overlap_stats,
    _merge_clusters_union_find,
    _should_merge,
    merge_clusters
)
from headss2.union_find import UnionFind

@pytest.fixture
def unified_cluster_data(spark):
    """
    Creates a unified dataset with:
    - 3 clusters: cluster 1 and 2 partially overlap, 3 does not.
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
def overlap_stats_df(unified_cluster_data, cluster_bounds, overlapping_pairs):
    return _apply_get_n_overlaps(
        clustered=unified_cluster_data,
        cluster_bounds=cluster_bounds,
        overlapping_clusters=overlapping_pairs,
    )

@pytest.fixture
def overlap_stats(overlap_stats_df):
    return _calculate_overlap_stats(n_overlaps_df=overlap_stats_df)


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
        "cluster1": ["1"],
        "cluster2": ["2"],
    })

    pd.testing.assert_frame_equal(
        overlapping_pairs.sort_values(by=["cluster1", "cluster2"]).reset_index(drop=True),
        expected.sort_values(by=["cluster1", "cluster2"]).reset_index(drop=True),
    )

def test_apply_get_n_overlaps(overlap_stats_df):
    # points [2.0, 2.0], [2.1,2.1] and [2.5, 2.5] overlap
    assert int(overlap_stats_df.loc[0, "n_cluster1_in2"]) == 1 
    assert int(overlap_stats_df.loc[0, "n_cluster2_in1"]) == 2  

def test_calculate_overlap_stats(overlap_stats):
    expected = pd.DataFrame([
        {
            "cluster1": "1",
            "cluster2": "2",
            "n_cluster1": 3,
            "n_cluster2": 3,
            "n_cluster1_in2": 1,
            "n_cluster2_in1": 2,
            "n_in_overlap": 3,
            "n_total": 6,
            "total_overlap_fraction": 0.5,
            "per_cluster_overlap_fraction": 2 / 3,
        }
    ])
    pd.testing.assert_frame_equal(overlap_stats, expected)

def test_should_merge_default_params(overlap_stats):
    should_merge_df = _should_merge(
        overlap_stats_df=overlap_stats,
        per_cluster_overlap_threshold=0.1,
        combined_per_cluster_overlap_threshold=0.5,
        min_n_overlap=10
    )
    assert should_merge_df[should_merge_df["cluster1"] == "1"]["should_merge"].values[0] == False

def test_should_merge_modified_params(overlap_stats):
    should_merge_df = _should_merge(
        overlap_stats_df=overlap_stats,
        per_cluster_overlap_threshold=0.1,
        combined_per_cluster_overlap_threshold=0.5,
        min_n_overlap=1
    )
    assert should_merge_df[should_merge_df["cluster1"] == "1"]["should_merge"].values[0] == True    

def test_assign_new_clusters(unified_cluster_data, overlap_stats):
    should_merge_df = _should_merge(
        overlap_stats_df=overlap_stats,
        per_cluster_overlap_threshold=0.1,
        combined_per_cluster_overlap_threshold=0.5,
        min_n_overlap=1
    )

    union_find = _merge_clusters_union_find(should_merge_df=should_merge_df)
    clustered_reassigned = _assign_new_clusters(union_find=union_find, clustered=unified_cluster_data)
    
    assert set(row['cluster'] for row in clustered_reassigned.select("cluster").distinct().collect()) == {"2", "3"}
