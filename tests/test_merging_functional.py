import pytest
import pandas as pd
from headss2.merging import merge_clusters
from pyspark.sql import SparkSession


# @pytest.fixture(scope="session")
# def spark():
#     return SparkSession.builder..appName("test-regions").getOrCreate()


@pytest.fixture
def t4_8k_clustered(spark):
    return spark.createDataFrame(
        pd.read_csv("tests/ground_truth/t48k_clustered.csv").iloc[:, 1:]
    )  # get around schema


@pytest.fixture
def t4_8k_split_regions():
    return pd.read_csv("tests/ground_truth/t48k_split_regions.csv").iloc[:, 1:]


@pytest.fixture
def t4_8k_merged(spark):
    return pd.read_csv("tests/ground_truth/t48k_merged_clusters.csv").iloc[:, 1:]


def test_merging_t4_8k(spark, t4_8k_clustered, t4_8k_merged):

    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    merged = merge_clusters(
        spark=spark,
        clustered=t4_8k_clustered,
        split_columns=["x", "y"],
        min_merge_members=10,
        overlap_merge_threshold=0.5,
        total_merge_threshold=0.1,
    )

    column_order = ["x", "y", "region", "group", "index"]

    pd.testing.assert_frame_equal(
        merged.toPandas()[column_order],
        t4_8k_merged[column_order],
        check_dtype=False,
        check_index_type=False,
    )
