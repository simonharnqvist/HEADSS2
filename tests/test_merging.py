import pandas as pd
from pyspark.sql import SparkSession, Row
import pytest
from headss2.merging import (
    get_cluster_bounds,
    find_overlapping_pairs,
    chain_merge_clusters,
)


# @pytest.fixture(scope="session")
# def spark():
#     return SparkSession.builder.master("local[*]").appName("test-regions").getOrCreate()


@pytest.fixture
def example_data(spark):
    data = [
        # Group 1
        [10.0, 20.0, 0, 1, 0],
        [15.0, 25.0, 0, 1, 0],
        [12.0, 18.0, 0, 1, 0],
        # Group 2
        [50.0, 60.0, 0, 2, 0],
        [55.0, 58.0, 0, 2, 0],
        [52.0, 62.0, 0, 2, 0],
        # Group 3
        [100.0, 200.0, 0, 3, 0],
        [110.0, 210.0, 0, 3, 0],
        [105.0, 190.0, 0, 3, 0],
    ]

    columns = ["x", "y", "region", "group", "index"]

    return spark.createDataFrame(pd.DataFrame(data, columns=columns))


@pytest.fixture
def cluster_bounds_df(spark):
    data = [
        Row(group=1, x_min=0, x_max=10, y_min=0, y_max=10),
        Row(group=2, x_min=5, x_max=15, y_min=5, y_max=15),
        Row(group=3, x_min=20, x_max=30, y_min=0, y_max=10),
        Row(group=4, x_min=8, x_max=25, y_min=8, y_max=25),
    ]
    return spark.createDataFrame(data)


@pytest.fixture
def clustered_data(spark):
    # Create small 2D clusters with partial overlap
    data = [
        # Cluster 1 (group = 1)
        Row(x=1, y=1, group=1),
        Row(x=2, y=2, group=1),
        Row(x=3, y=3, group=1),
        # Cluster 2 (group = 2) — partially overlaps with cluster 1
        Row(x=2, y=2, group=2),
        Row(x=3, y=3, group=2),
        Row(x=4, y=4, group=2),
        # Cluster 3 (group = 3) — no overlap
        Row(x=10, y=10, group=3),
        Row(x=11, y=11, group=3),
        Row(x=12, y=12, group=3),
    ]
    return spark.createDataFrame(data)


@pytest.fixture
def overlap_pairs(spark):
    # Only compare group 1 with 2, and 1 with 3
    data = [
        Row(group1=1, group2=2),
        Row(group1=1, group2=3),
    ]
    return spark.createDataFrame(data)


def test_get_cluster_bounds(example_data):
    # Convert the example_data (pandas df) to Spark DataFrame

    # Call the function with split columns x and y
    result = get_cluster_bounds(
        example_data, split_columns=["x", "y"], group_col="group"
    )

    # Collect result into Pandas for easy comparison
    result_df = result.toPandas()

    expected = pd.DataFrame(
        {
            "group": [1, 2, 3],
            "x_min": [10.0, 50.0, 100.0],
            "x_max": [15.0, 55.0, 110.0],
            "y_min": [18.0, 58.0, 190.0],
            "y_max": [25.0, 62.0, 210.0],
        }
    )

    pd.testing.assert_frame_equal(
        expected,
        result_df,
        check_dtype=False,
        check_index_type=False,
    )


def test_find_overlapping_pairs(cluster_bounds_df):
    result = find_overlapping_pairs(cluster_bounds_df, split_columns=["x", "y"])
    result_df = (
        result.toPandas().sort_values(by=["group1", "group2"]).reset_index(drop=True)
    )

    # Define expected overlapping group pairs
    expected = pd.DataFrame({"group1": [1, 1, 2, 3], "group2": [2, 4, 4, 4]})

    expected = expected.sort_values(by=["group1", "group2"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, expected)


def test_chain_merge_clusters(spark):
    # Create test input: group merge pairs
    data = [
        Row(group1=1, group2=2),
        Row(group1=2, group2=3),
        Row(group1=4, group2=5),
    ]
    spark.sparkContext.setCheckpointDir("/tmp/graphframe-checkpoints")
    merge_pairs_df = spark.createDataFrame(data)

    # Call function
    result = chain_merge_clusters(merge_pairs_df)

    # Collect and process output
    result_dict = {row["group"]: row["merged_group"] for row in result.collect()}

    # Check connected groups have the same component
    assert result_dict[1] == result_dict[2] == result_dict[3]
    assert result_dict[4] == result_dict[5]
    assert result_dict[1] != result_dict[4]
