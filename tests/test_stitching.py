import pandas as pd
from headss2 import stitching
import pytest

@pytest.fixture
def a3_clustered():
    return pd.read_csv("tests/ground_truth/a3_clustered.csv", index_col=0)

@pytest.fixture
def a3_centers():
    return pd.read_csv("tests/ground_truth/a3_centers.csv", index_col=0)

@pytest.fixture
def a3_stitch_regions():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)

@pytest.fixture
def a3_misplaced_centers():
    return pd.read_csv("tests/ground_truth/a3_cut_misplaced_centers.csv", index_col=0).set_index("group")

@pytest.fixture
def make_a3_centers(a3_clustered):
    centers = stitching.get_centers(
        a3_clustered, 
        split_columns=["x", "y"])
    return centers


def test_get_centers(make_a3_centers, a3_centers):

    actual = make_a3_centers
    actual = pd.concat(actual).loc[:, ["group", "x", "y", "N"]].set_index("group", drop=True)
    expected = a3_centers.set_index("group", drop=True)

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_cut_misplaced_clusters(make_a3_centers, a3_stitch_regions, a3_misplaced_centers):

    actual = stitching.cut_misplaced_clusters(make_a3_centers, 
                                              stitch_regions=a3_stitch_regions, 
                                              split_columns=["x", "y"]).loc[:, ["group", "x", "y", "N"]].set_index("group", drop=True)
    expected = a3_misplaced_centers
    
    print(actual.head(), "\n", expected.head())

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
    )
