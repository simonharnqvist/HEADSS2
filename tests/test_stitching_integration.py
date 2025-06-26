import pytest
import pandas as pd

from headss import datasets, cluster, stitching
from dataset_fixtures import flame_clustered, spiral_clustered

@pytest.fixture
def flame_stitching_result():
    return pd.read_csv("tests/ground_truth/flame_stitching_result.csv")

@pytest.fixture
def spiral_stitching_result():
    return pd.read_csv("tests/ground_truth/spiral_stitching_result.csv")

def test_flame_stitching_consistent(flame_clustered, flame_stitching_result):

    regions_data, stitching_data = flame_clustered

    actual = stitching.stitch(regions = regions_data, split_columns=["x", "y"], stitch_regions=stitching_data) 
    expected = flame_stitching_result()
    
    pd.testing.assert_frame_equal(
        expected, actual,
        check_dtype=False,
        check_index_type=False
    )
    ## this will most likely fail since clustering is not deterministic  - compare dimensions? Or check that the error is what is expected?

def test_spiral_stitching_consistent(spiral_clustered, spiral_stitching_result):

    regions_data, stitching_data = spiral_clustered

    actual = stitching.stitch(regions = regions_data, split_columns=["x", "y"], stitch_regions=stitching_data) 
    expected = spiral_stitching_result()
    
    pd.testing.assert_frame_equal(
        expected, actual,
        check_dtype=False,
        check_index_type=False
    )
    ## this will most likely fail since clustering is not deterministic  - compare dimensions?

