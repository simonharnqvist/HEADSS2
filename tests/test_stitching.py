import pandas as pd
from headss2 import stitching
import pytest
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
    return spark.createDataFrame(
        pd.read_csv("tests/ground_truth/a3_clustered.csv", index_col=0)
    )


@pytest.fixture
def a3_centers():
    return pd.read_csv("tests/ground_truth/a3_centers.csv", index_col=0)


@pytest.fixture
def a3_stitch_regions():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)


@pytest.fixture
def a3_misplaced_centers():
    return pd.read_csv(
        "tests/ground_truth/a3_cut_misplaced_centers.csv", index_col=0
    ).set_index("cluster")


@pytest.fixture
def make_a3_centers(a3_clustered):
    centers = stitching.get_centers(a3_clustered, cluster_columns=["x", "y"])
    return centers


def test_calculate_centers():
    data = pd.DataFrame(
        {"cluster": [0, 0, 1, 1, 1], "x": [1, 3, 5, 7, 9], "y": [2, 4, 6, 8, 10]}
    )
    cluster_columns = ["x", "y"]

    result = stitching.calculate_centers(data, cluster_columns)

    assert len(result) == 2
    assert set(result["cluster"]) == {0, 1}

    # Check median approx for cluster 0
    cluster_0 = result[result["cluster"] == 0].iloc[0]
    assert cluster_0["N"] == 2
    assert cluster_0["x"] == 2  # median of [1,3]
    assert cluster_0["y"] == 3  # median of [2,4]

    # Check median approx for cluster 1
    cluster_1 = result[result["cluster"] == 1].iloc[0]
    assert cluster_1["N"] == 3
    assert cluster_1["x"] == 7  # median of [5,7,9]
    assert cluster_1["y"] == 8  # median of [6,8,10]


def test_get_centers(make_a3_centers, a3_centers):

    actual = (
        pd.concat(make_a3_centers)
        .loc[:, ["cluster", "x", "y", "N"]]
        .set_index("cluster", drop=True)
    ).sort_values("cluster")
    expected = a3_centers.set_index("cluster", drop=True)

    pd.testing.assert_frame_equal(
        expected, actual, check_dtype=False, check_index_type=False
    )


def test_cut_misplaced_clusters(
    make_a3_centers, a3_stitch_regions, a3_misplaced_centers
):

    actual = (
        stitching.cut_misplaced_clusters(
            make_a3_centers, stitch_regions=a3_stitch_regions, cluster_columns=["x", "y"]
        )
        .loc[:, ["cluster", "x", "y", "N"]]
        .set_index("cluster", drop=True)
        .sort_values("cluster")
    )
    expected = a3_misplaced_centers

    print(actual.head(), "\n", expected.head())

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
    )
