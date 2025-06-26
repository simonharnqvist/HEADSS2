import numpy as np
import pandas as pd
import pytest
from headss import regions

@pytest.fixture
def flame():
    return pd.read_csv("example_data/flame.csv", header=None, usecols=[0,1,2]).rename(columns = {0:'x', 1:'y'})

def test_get_step(flame):
    step = regions.get_step(flame, ["x", "y"], n_cubes=2)
    expected_step = np.array([6.85, 6.675])
    np.testing.assert_allclose(step, expected_step)

def test_get_limits(flame):
    step = np.array([6.85, 6.675])
    limits = regions.get_limits(flame, step=step, split_columns=["x", "y"])
    expected_limits = np.array([
        [0.5, 14.45],
        [0.5, 17.7875],
        [0.5, 21.125],
        [3.925, 14.45],
        [3.925, 17.7875],
        [3.925, 21.125],
        [7.35, 14.45],
        [7.35, 17.7875],
        [7.35, 21.125]
    ])
    np.testing.assert_allclose(limits, expected_limits)

def test_get_minima():
    step = np.array([6.85, 6.675])
    limits = np.array([
        [0.5, 14.45],
        [0.5, 17.7875],
        [0.5, 21.125],
        [3.925, 14.45],
        [3.925, 17.7875],
        [3.925, 21.125],
        [7.35, 14.45],
        [7.35, 17.7875],
        [7.35, 21.125]
    ])
    minima = regions.get_minima(limits, step)
    expected_minima = np.array([
        [0.5, 14.45],
        [0.5, 19.45625],
        [0.5, 22.79375],
        [5.6375, 14.45],
        [5.6375, 19.45625],
        [5.6375, 22.79375],
        [9.0625, 14.45],
        [9.0625, 19.45625],
        [9.0625, 22.79375]
    ])
    np.testing.assert_allclose(minima, expected_minima)
