import numpy as np
import pandas as pd
import pytest
from headss2 import regions, dataset
from pyspark.sql import SparkSession
from pyspark import sql


# @pytest.fixture(scope="session")
# def spark():
#     return SparkSession.builder.master("local[*]").appName("test-regions").getOrCreate()


@pytest.fixture
def flame(spark):
    return dataset("flame", format="spark", spark_session=spark)


@pytest.fixture
def flame_regions(flame, spark):
    return regions.make_regions(
        spark_session=spark, df=flame, n=2, cluster_columns=["x", "y"]
    )


def test_assign_regions_basic(spark):
    data = [(0.1, 0.1), (0.9, 0.9), (0.4, 0.6)]
    df = spark.createDataFrame(data, ["x", "y"])
    df_with_region = regions.assign_regions(df, cluster_columns=["x", "y"], n=2)

    regs = [row["region"] for row in df_with_region.collect()]
    assert all(isinstance(r, int) for r in regs)
    assert len(set(regs)) == len(data), print(regs)


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


def test_make_regions_end_to_end(spark):
    # Simple 2D data
    data = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    df = spark.createDataFrame(data, ["x", "y"])
    result = regions.make_regions(spark, df, n=2, cluster_columns=["x", "y"])

    # Region column exists
    assert "region" in result.split_data.columns

    # Check number of regions
    assert result.split_data.select("region").distinct().count() == 9

    # Region count matches stitch regions
    split_region_count = result.split_regions.shape[0]
    stitch_region_count = result.stitch_regions.shape[0]
    assert split_region_count == stitch_region_count

    # All rows assigned a region
    assert result.split_data.filter("region is null").count() == 0


def test_split_dataframes_preserves_number_of_rows(flame):
    df = regions.assign_regions(df=flame, n=2, cluster_columns=["x", "y"])
    assert df.count() == flame.count()


def test_make_regions_preserves_number_of_rows(flame_regions, flame):
    assert flame_regions.split_data.count() == flame.count()
