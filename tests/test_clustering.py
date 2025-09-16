from headss2 import cluster, dataset, make_regions
import pytest
from pyspark.sql import SparkSession


# @pytest.fixture(scope="session")
# def spark():
#     return SparkSession.builder.master("local[*]").appName("test-regions").getOrCreate()


@pytest.fixture
def a3_regions(spark):
    a3 = dataset("a3")
    regs = make_regions(spark_session=spark, df=a3, n=2, cluster_columns=["x", "y"])
    return regs


def test_clustering_returns_original_number_of_entries_a3(a3_regions):
    clustered = cluster(
        split_data=a3_regions.split_data,
        min_cluster_size=5,
        min_samples=None,
        allow_single_cluster=True,
        clustering_method="eom",
        cluster_columns=["x", "y"],
        drop_unclustered=False,
    )
    assert clustered.count() == a3_regions.split_data.count()


@pytest.fixture
def t4_8k_regions(spark):
    t4_8k = dataset("t4_8k")
    regs = make_regions(spark_session=spark, df=t4_8k, n=2, cluster_columns=["x", "y"])
    return regs


def test_clustering_returns_original_number_of_entries_t4_8k(t4_8k_regions):
    clustered = cluster(
        split_data=t4_8k_regions.split_data,
        min_cluster_size=5,
        min_samples=None,
        allow_single_cluster=True,
        clustering_method="eom",
        cluster_columns=["x", "y"],
        drop_unclustered=False,
    )
    assert clustered.count() == t4_8k_regions.split_data.count()
