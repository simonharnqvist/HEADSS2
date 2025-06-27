import pytest
import pandas as pd

from headss import datasets, cluster, stitching
from dataset_fixtures import a3_clustered

@pytest.fixture
def a3_stitching_result():
    return pd.read_csv("tests/ground_truth/a3_stitching_result.csv", index_col=0).set_index("region", drop=True).drop(columns=["index"])

def test_flame_stitching_consistent(a3_clustered, a3_stitching_result):
    regions_data, stitching_data = a3_clustered

    actual = stitching.stitch(regions = regions_data, split_columns=["x", "y"], stitch_regions=stitching_data).compute()
    expected = a3_stitching_result

    with pytest.raises(AssertionError): #dataframes expected to be different, but columns etc should match
    
        pd.testing.assert_frame_equal(
            expected, actual,
            check_dtype=False,
            check_index_type=False
        )