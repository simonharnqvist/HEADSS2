import pytest
import pandas as pd
from headss2 import clustering, datasets, stitching

@pytest.fixture
def a3_clustered():
    return pd.read_csv("tests/ground_truth/a3_clustered.csv", index_col=0).set_index("region", drop=True).drop(columns=["index"])

@pytest.fixture
def a3_stitch_regions():
    return pd.read_csv("tests/ground_truth/a3_stitch_regions.csv", index_col=0)

@pytest.fixture
def a3_stitched():
    return pd.read_csv("tests/ground_truth/a3_stitched.csv", index_col=0)

def test_stitching_a3(a3_clustered, a3_stitch_regions, a3_stitched):

    actual = stitching.stitch(clustered_data=a3_clustered, 
        split_columns=["x", "y"], stitch_regions=a3_stitch_regions)
    
    expected = a3_stitched.copy()

    expected.columns = expected.columns.map(str)
    actual.columns = actual.columns.map(str)

    columns_to_check = ['x', 'y', 'region', 'group']
    actual["region"] = actual.index
    actual = actual.reset_index(drop=True)

    expected_subset = expected[columns_to_check].sort_values(by=columns_to_check).reset_index(drop=True)
    actual_subset = actual[columns_to_check].sort_values(by=columns_to_check).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        expected_subset,
        actual_subset,
        check_dtype=False,
    )
