import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from typing import List, Tuple

from pyspark import sql
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    StringType,
)
from pyspark.sql.functions import monotonically_increasing_id
import uuid


def run_hdbscan(
    df: pd.DataFrame,
    region: int,
    min_cluster_size: int,
    min_samples: int | None,
    allow_single_cluster: bool,
    clustering_method: str,
    cluster_columns: List[str],
    drop_ungrouped: bool = True,
    random_seed: int = 11,
) -> pd.DataFrame:
    """Cluster objects using HDBSCAN and return the labeled DataFrame and cluster count."""
    np.random.seed(random_seed)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=clustering_method,
        gen_min_span_tree=False,
    ).fit(df[cluster_columns])

    assert isinstance(df, pd.DataFrame)
    df["group"] = clusterer.labels_

    if drop_ungrouped:
        df = df[df["group"] != -1]

    df["group"] = df["group"].apply(lambda x: f"{region}_{x}" if x != -1 else "-1")

    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["group"])

    df.index.rename("index", inplace=True)

    return df


def cluster(
    split_data: sql.DataFrame,
    min_cluster_size: int,
    min_samples: int | None,
    allow_single_cluster: bool,
    clustering_method: str,
    cluster_columns: List[str],
    drop_ungrouped: bool = True,
) -> sql.DataFrame:
    """Perform HDBSCAN clustering on a Spark DataFrame, per region."""

    def run_hdbscan_per_region(pdf: pd.DataFrame) -> pd.DataFrame:
        region_id = pdf["region"].iloc[0]
        clustered_df = run_hdbscan(
            df=pdf,
            region=region_id,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=allow_single_cluster,
            clustering_method=clustering_method,
            cluster_columns=cluster_columns,
            drop_ungrouped=drop_ungrouped,
        )
        assert not clustered_df.empty
        assert "group" in clustered_df.columns
        return clustered_df

    res = run_hdbscan_per_region(pdf=split_data.toPandas())
    assert "group" in res.columns

    schema_list = [
        StructField(col_name, FloatType(), True) for col_name in cluster_columns
    ] + [
        StructField("region", IntegerType(), True),
        StructField("group", StringType(), True),
    ]

    output_schema = StructType(schema_list)

    clustered = split_data.groupBy("region").applyInPandas(
        run_hdbscan_per_region,
        schema=output_schema,
    )

    return clustered
