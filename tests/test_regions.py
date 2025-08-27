import numpy as np
import pandas as pd
import pytest
from headss2 import regions, dataset
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[*]").appName("test-regions").getOrCreate()


@pytest.fixture
def flame():
    return dataset("flame")  # still returns a Pandas DataFrame


@pytest.fixture
def flame_sdf(spark, flame):
    return spark.createDataFrame(flame)


def test_get_step(flame_sdf):
    step = regions.get_step(flame_sdf, ["x", "y"], n=2)
    expected_step = np.array([6.85, 6.675])
    np.testing.assert_allclose(step, expected_step, rtol=1e-4)


def test_get_limits(flame_sdf):
    step = np.array([6.85, 6.675])
    limits = regions.get_limits(flame_sdf, step=step, split_columns=["x", "y"])
    expected_limits = np.array(
        [
            [0.5, 14.45],
            [0.5, 17.7875],
            [0.5, 21.125],
            [3.925, 14.45],
            [3.925, 17.7875],
            [3.925, 21.125],
            [7.35, 14.45],
            [7.35, 17.7875],
            [7.35, 21.125],
        ]
    )
    np.testing.assert_allclose(limits, expected_limits, rtol=1e-4)


def test_get_minima():
    step = np.array([6.85, 6.675])
    limits = np.array(
        [
            [0.5, 14.45],
            [0.5, 17.7875],
            [0.5, 21.125],
            [3.925, 14.45],
            [3.925, 17.7875],
            [3.925, 21.125],
            [7.35, 14.45],
            [7.35, 17.7875],
            [7.35, 21.125],
        ]
    )
    minima = regions.get_minima(limits, step)
    expected_minima = np.array(
        [
            [0.5, 14.45],
            [0.5, 19.45625],
            [0.5, 22.79375],
            [5.6375, 14.45],
            [5.6375, 19.45625],
            [5.6375, 22.79375],
            [9.0625, 14.45],
            [9.0625, 19.45625],
            [9.0625, 22.79375],
        ]
    )
    np.testing.assert_allclose(minima, expected_minima, rtol=1e-4)
