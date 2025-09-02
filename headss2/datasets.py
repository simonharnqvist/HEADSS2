from pyspark import sql
import pandas as pd
from importlib.resources import files
from headss2 import example_data


def dataset(
    name: str, format: str = "pandas", spark_session: sql.SparkSession | None = None
) -> pd.DataFrame | sql.DataFrame:
    filename = f"{name}.csv"

    file_path = files(example_data).joinpath(filename)

    with file_path.open("rb") as f:
        df = pd.read_csv(f, header=None).iloc[:, 0:2].rename(columns={0: "x", 1: "y"})

    if format.lower() not in ["pandas", "spark"]:
        raise ValueError("'format' must be 'pandas' or 'spark'")

    if format.lower() == "pandas":
        return df
    elif format.lower() == "spark" and spark_session is not None:
        return spark_session.createDataFrame(df)
    elif spark_session is None or not isinstance(spark_session, sql.SparkSession):
        raise ValueError("No valid SparkSession provided")
