from pyspark.sql import SparkSession
import pytest


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.appName("TestSession")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "8g")
        .config("spark.executor.cores", "2")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .getOrCreate()
    )
    yield spark
    spark.stop()
