import pytest
import pandas as pd
from headss import regions

@pytest.fixture
def flame_regions_ground_truth():
    return pd.read_csv("tests/ground_truth/flame_regions.csv", index_col=0)

@pytest.fixture
def flame_split_ground_truth():
    return pd.read_csv("tests/ground_truth/flame_split_regions.csv", index_col=0)

@pytest.fixture
def flame_stitch_ground_truth():
    return pd.read_csv("tests/ground_truth/flame_stitch_regions.csv", index_col=0)

@pytest.fixture
def spiral_regions_ground_truth():
    return pd.read_csv("tests/ground_truth/spiral_regions.csv", index_col=0)

@pytest.fixture
def spiral_split_ground_truth():
    return pd.read_csv("tests/ground_truth/spiral_split_regions.csv", index_col=0)

@pytest.fixture
def spiral_stitch_ground_truth():
    return pd.read_csv("tests/ground_truth/spiral_stitch_regions.csv", index_col=0)

@pytest.fixture
def flame_data():
    return pd.read_csv("example_data/flame.csv", header=None).rename(columns = {0:'x', 1:'y'})

@pytest.fixture
def spiral_data():
    return pd.read_csv("example_data/spiral.csv", header=None).rename(columns = {0:'x', 1:'y'})

@pytest.fixture
def flame_regions(flame_data):
    return regions.make_regions(n_cubes = 2, df = flame_data, split_columns=['x','y'])

@pytest.fixture
def spiral_regions(spiral_data):
    return regions.make_regions(n_cubes = 2, df = spiral_data, split_columns=['x','y'])

## Test cases

def test_flame_regions_match_ground_truth(flame_regions_ground_truth, flame_regions):
    print(flame_regions_ground_truth.columns, flame_regions.split_data.columns)
    pd.testing.assert_frame_equal(flame_regions_ground_truth, flame_regions.split_data)

def test_flame_regions_split_match_ground_truth(flame_split_ground_truth, flame_regions):
    pd.testing.assert_frame_equal(flame_split_ground_truth, flame_regions.split_regions)    

def test_flame_regions_stitch_match_ground_truth(flame_split_ground_truth, flame_regions):
    pd.testing.assert_frame_equal(flame_split_ground_truth, flame_regions.stitch_regions)


def test_spiral_regions_match_ground_truth(spiral_regions_ground_truth, spiral_regions):
    pd.testing.assert_frame_equal(spiral_regions_ground_truth, spiral_regions.split_data)

def test_spiral_regions_split_match_ground_truth(spiral_split_ground_truth, spiral_regions):
    pd.testing.assert_frame_equal(spiral_split_ground_truth, spiral_regions.split_regions)    

def test_spiral_regions_stitch_match_ground_truth(spiral_split_ground_truth, spiral_regions):
    pd.testing.assert_frame_equal(spiral_split_ground_truth, spiral_regions.stitch_regions)