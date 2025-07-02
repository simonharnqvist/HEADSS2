import pandas as pd
from headss import stitching
import pytest
import dask.dataframe as dd

@pytest.fixture
def a3_clustered():
    return pd.read_csv("tests/ground_truth/a3_clustered.csv", index_col=0)

@pytest.fixture
def a3_centers():
    return pd.read_csv("tests/ground_truth/a3_centers.csv", index_col=0)

@pytest.fixture
def a3_stitch_regions():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)

def test_centers(a3_clustered, a3_centers):
    actual = stitching.get_centers(
        dd.from_pandas(a3_clustered, npartitions=1), 
        split_columns=["x", "y"])[["group", "x", "y", "N"]].set_index("group", drop=True).compute()
    expected = a3_centers.set_index("group", drop=True)


    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_index_type=False
    )

def test_cut_misplaced_clusters(a3_centers, a3_stitch_regions):

    a3_centers = dd.from_pandas(a3_centers, npartitions=1)

    actual = stitching.cut_misplaced_clusters(a3_centers, 
                                              stitch_regions=a3_stitch_regions, 
                                              split_columns=["x", "y"])
    expected = pd.read_csv("tests/ground_truth/a3_cut_misplaced_centers.csv", index_col=0)

    print(actual.head(), "\n", expected.head())

    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
    )
