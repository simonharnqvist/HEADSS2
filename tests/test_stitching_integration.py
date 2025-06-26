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
    actual = stitching.stitch(df = flame_clustered, split_columns=["x", "y"]) 
    expected = flame_stitching_result()
    
    pd.testing.assert_frame_equal(
        expected, actual,
        check_dtype=False,
        check_index_type=False
    )
    ## this will most likely fail since clustering is not deterministic  - compare dimensions? Or check that the error is what is expected?

def test_spiral_stitching_consistent(spiral_clustered, spiral_stitching_result):
    actual = stitching.stitch(df = spiral_clustered, split_columns=["x", "y"]) 
    expected = spiral_stitching_result()
    
    pd.testing.assert_frame_equal(
        expected, actual,
        check_dtype=False,
        check_index_type=False
    )
    ## this will most likely fail since clustering is not deterministic  - compare dimensions?

