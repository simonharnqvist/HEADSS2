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
    drop_unclustered: bool = True,
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
    df["cluster"] = clusterer.labels_

    if drop_unclustered:
        df = df[df["cluster"] != -1]

    df["cluster"] = df["cluster"].apply(lambda x: f"{region}_{x}" if x != -1 else "-1")

    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["cluster"])

    df.index.rename("index", inplace=True)

    return df


def cluster(
    split_data: sql.DataFrame,
    min_cluster_size: int,
    min_samples: int | None,
    allow_single_cluster: bool,
    clustering_method: str,
    cluster_columns: List[str],
    drop_unclustered: bool = True,
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
            drop_unclustered=drop_unclustered,
        )
        assert not clustered_df.empty
        assert "cluster" in clustered_df.columns
        return clustered_df

    res = run_hdbscan_per_region(pdf=split_data.toPandas())
    assert "cluster" in res.columns

    schema_list = [
        StructField(col_name, FloatType(), True) for col_name in cluster_columns
    ] + [
        StructField("region", IntegerType(), True),
        StructField("cluster", StringType(), True),
    ]

    output_schema = StructType(schema_list)

    clustered = split_data.groupBy("region").applyInPandas(
        run_hdbscan_per_region,
        schema=output_schema,
    )

    return clustered
