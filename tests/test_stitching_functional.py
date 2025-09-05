import pytest
import pandas as pd
from headss2 import clustering, datasets, stitching
from pyspark.sql.functions import col
from pyspark.sql.types import StringType
from pyspark import sql


# @pytest.fixture(scope="session")
# def spark():
#     return (
#         sql.SparkSession.builder.master("local[*]")
#         .appName("test-regions")
#         .getOrCreate()
#     )


@pytest.fixture
def a3_clustered(spark):
    return (
        spark.createDataFrame(
            pd.read_csv("tests/ground_truth/a3_clustered.csv", index_col=0)
        )  # .drop_duplicates(subset=["x", "y"])
        # .set_index("region", drop=True)
        # .drop(columns=["index"])
    )


@pytest.fixture
def a3_stitch_regions():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)


@pytest.fixture
def a3_stitched(spark):
    return spark.createDataFrame(
        pd.read_csv("tests/ground_truth/a3_stitched.csv", index_col=0)
    )


def test_stitching_a3(a3_clustered, a3_stitch_regions, a3_stitched):
    actual = stitching.stitch(
        clustered=a3_clustered,
        split_columns=["x", "y"],
        stitch_regions=a3_stitch_regions,
    )

    expected = a3_stitched

    # Cast all columns to string in both DataFrames
    for col_name in actual.columns:
        actual = actual.withColumn(col_name, col(col_name).cast(StringType()))
    for col_name in expected.columns:
        expected = expected.withColumn(col_name, col(col_name).cast(StringType()))

    # Define columns to check
    columns_to_check = ["x", "y", "region", "group"]

    # Select relevant columns and sort for comparison
    actual_subset = actual.select(columns_to_check).orderBy(columns_to_check)
    expected_subset = expected.select(columns_to_check).orderBy(columns_to_check)

    # Convert to pandas for comparison
    actual_pd = actual_subset.toPandas().reset_index(drop=True)
    expected_pd = expected_subset.toPandas().reset_index(drop=True)

    # Perform pandas dataframe equality check
    pd.testing.assert_frame_equal(
        expected_pd,
        actual_pd,
        check_dtype=False,
    )
