import numpy as np
import pandas as pd
import pytest
from headss2 import regions, dataset
from pyspark.sql import SparkSession
from pyspark import sql


@pytest.fixture
def example_data(spark):
    data = [
        (0.0, 0.0), (0.1, 0.1), (0.2, 0.2),  # Cluster A
        (1.0, 1.0), (1.1, 1.1), (1.2, 1.2),  # Cluster B
        (2.0, 2.0), (2.1, 2.2), (2.2, 2.1),  # Cluster C
        (5.0, 5.0), (5.1, 5.1), (5.2, 5.3),  # Cluster D (distant cluster)
        (10.0, 10.0), (10.5, 10.5),          # Sparse region
        (3.0, 0.0), (3.5, 0.5),              # Diagonal/linear points
        (4.0, 1.0), (4.5, 1.5),              # Another diagonal line
        (6.0, 3.0),                          # Outlier
    ]
    return spark.createDataFrame(data, ["x", "y"])
        
@pytest.fixture
def flame(spark):
    return dataset("flame", format="spark", spark_session=spark)


@pytest.fixture
def flame_regions(flame, spark):
    return regions.make_regions(
        spark_session=spark, df=flame, n=2, cluster_columns=["x", "y"]
    )

def test_get_step_and_limits(spark):
    data = [(0.0, 0.0), (1.0, 1.0)]
    df = spark.createDataFrame(data, ["x", "y"])
    step, limits = regions.get_step_and_limits(df, cluster_columns=["x", "y"], n=2)

    assert isinstance(step, np.ndarray)
    assert step.shape == (2,)
    assert limits.shape[1] == 2
    assert limits.shape[0] == (2 * 2 - 1) ** 2  # (2n - 1)^d = 9


def test_get_minima_maxima():
    step = np.array([1.0, 1.0])
    limits = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    low = regions.get_minima(limits, step)
    high = regions.get_maxima(limits, step)

    assert low.shape == limits.shape
    assert high.shape == limits.shape
    assert np.all(high > low)


def test_get_n_regions():
    n = 2
    dims = ["x", "y", "z"]
    expected = (2 * n - 1) ** len(dims)  # (2n - 1)^d = 125
    assert regions.get_n_regions(n, dims) == expected


def test_make_regions_end_to_end(spark, example_data):

    regs = regions.make_regions(spark, example_data, n=2, cluster_columns=["x", "y"])

    assert "region" in regs.split_data.columns

    num_regions = regs.split_data.select("region").distinct().count()
    assert num_regions == 9  # (2*2 - 1)^2 = 3**2=9

    split_region_count = regs.split_regions.shape[0]
    stitch_region_count = regs.stitch_regions.shape[0]
    assert split_region_count == stitch_region_count

    assert regs.split_data.filter("region is null").count() == 0