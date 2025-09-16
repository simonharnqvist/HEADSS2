import pytest
import pandas as pd
from headss2 import regions, dataset
from pyspark.sql import SparkSession


# @pytest.fixture(scope="session")
# def spark():
#     return SparkSession.builder.master("local[*]").appName("test-regions").getOrCreate()


@pytest.fixture
def flame_regions_ground_truth():
    return pd.read_csv(
        "tests/ground_truth/flame_regions.csv", index_col=0
    ).drop_duplicates(subset=["x", "y"])


@pytest.fixture
def flame_split_ground_truth():
    return pd.read_csv("tests/ground_truth/flame_split_regions.csv", index_col=0)


@pytest.fixture
def flame_stitch_ground_truth():
    return pd.read_csv("tests/ground_truth/flame_stitch_regions.csv", index_col=0)


@pytest.fixture
def spiral_regions_ground_truth():
    return (
        pd.read_csv(
            "tests/ground_truth/spiral_regions.csv", dtype={"2": "Int64"}, index_col=0
        )
        .drop_duplicates(subset=["x", "y"])
        .reset_index()
        .sort_values(by=["x", "y"])
    )


@pytest.fixture
def spiral_split_ground_truth():
    return pd.read_csv("tests/ground_truth/spiral_split_regions.csv", index_col=0)


@pytest.fixture
def spiral_stitch_ground_truth():
    return pd.read_csv("tests/ground_truth/spiral_stitch_regions.csv", index_col=0)


@pytest.fixture
def flame_data(spark):
    return dataset("flame", format="spark", spark_session=spark)


@pytest.fixture
def spiral_data(spark):
    return dataset("spiral", format="spark", spark_session=spark)


@pytest.fixture
def a3_data(spark):
    return dataset("a3", format="spark", spark_session=spark)


@pytest.fixture
def flame_regions(spark, flame_data):
    return regions.make_regions(
        spark_session=spark, n=2, df=flame_data, cluster_columns=["x", "y"]
    )


@pytest.fixture
def spiral_regions(spark, spiral_data):
    return regions.make_regions(
        spark_session=spark, n=2, df=spiral_data, cluster_columns=["x", "y"]
    )


@pytest.fixture
def a3_regions(spark, a3_data):
    return regions.make_regions(
        spark_session=spark, n=2, df=a3_data, cluster_columns=["x", "y"]
    )


@pytest.fixture
def a3_stitch_ground_truth():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)


## Test cases


def test_flame_regions_match_ground_truth(flame_regions_ground_truth, flame_regions):
    expected = flame_regions_ground_truth.copy()
    actual = flame_regions.split_data.toPandas().copy()

    expected.columns = expected.columns.map(str)
    actual.columns = actual.columns.map(str)

    columns_to_check = ["x", "y", "region"]
    actual = actual.reset_index(drop=True)

    expected_subset = (
        expected[columns_to_check]
        .sort_values(by=columns_to_check)
        .reset_index(drop=True)
    )
    actual_subset = (
        actual[columns_to_check].sort_values(by=columns_to_check).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        expected_subset,
        actual_subset,
        check_dtype=False,
    )


def test_flame_regions_split_match_ground_truth(
    flame_split_ground_truth, flame_regions
):
    column_order = ["region", "x_min", "x_max", "y_min", "y_max"]
    pd.testing.assert_frame_equal(
        flame_split_ground_truth[column_order],
        flame_regions.split_regions[column_order],
        check_dtype=False,
        check_index_type=False,
    )


def test_flame_regions_stitch_match_ground_truth(
    flame_stitch_ground_truth, flame_regions
):
    column_order = ["region", "x_min", "x_max", "y_min", "y_max"]
    pd.testing.assert_frame_equal(
        flame_stitch_ground_truth[column_order],
        flame_regions.stitch_regions[column_order],
        check_dtype=False,
        check_index_type=False,
    )


def test_spiral_regions_match_ground_truth(spiral_regions_ground_truth, spiral_regions):
    expected = spiral_regions_ground_truth.copy()
    actual = spiral_regions.split_data.toPandas().sort_values(by=["x", "y"])

    print(expected)
    print(actual)

    expected.columns = expected.columns.map(str)
    actual.columns = actual.columns.map(str)

    columns_to_check = ["x", "y", "region"]
    actual = actual.reset_index(drop=True)

    expected_subset = (
        expected[columns_to_check]
        .sort_values(by=columns_to_check)
        .reset_index(drop=True)
    )
    actual_subset = (
        actual[columns_to_check].sort_values(by=columns_to_check).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        expected_subset, actual_subset, check_dtype=False, check_index_type=False
    )


def test_spiral_regions_split_match_ground_truth(
    spiral_split_ground_truth, spiral_regions
):

    print(spiral_regions.split_regions)
    column_order = ["region", "x_min", "x_max", "y_min", "y_max"]
    pd.testing.assert_frame_equal(
        spiral_split_ground_truth[column_order],
        spiral_regions.split_regions[column_order],
        check_dtype=False,
        check_index_type=False,
    )


def test_spiral_regions_stitch_match_ground_truth(
    spiral_stitch_ground_truth, spiral_regions
):
    column_order = ["region", "x_min", "x_max", "y_min", "y_max"]
    pd.testing.assert_frame_equal(
        spiral_stitch_ground_truth[column_order],
        spiral_regions.stitch_regions[column_order],
        check_dtype=False,
        check_index_type=False,
    )


def test_a3_stitch_regions_match_ground_truth(a3_stitch_ground_truth, a3_regions):

    column_order = ["region", "x_min", "x_max", "y_min", "y_max"]

    pd.testing.assert_frame_equal(
        a3_stitch_ground_truth[column_order],
        a3_regions.stitch_regions[column_order],
        check_dtype=False,
        check_index_type=False,
    )
