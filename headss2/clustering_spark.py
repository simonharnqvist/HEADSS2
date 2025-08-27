import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from typing import List, Tuple

from pyspark import sql
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import monotonically_increasing_id


def run_hdbscan(
    df: pd.DataFrame,
    min_cluster_size: int,
    min_samples: int | None,
    allow_single_cluster: bool,
    cluster_method: str,
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
        cluster_selection_method=cluster_method,
        gen_min_span_tree=False,
    ).fit(df[cluster_columns])

    df["group"] = clusterer.labels_

    if drop_ungrouped:
        df = df[df["group"] != -1]

    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["group"])

    return df


def cluster(
    split_data: sql.DataFrame,
    min_cluster_size: int,
    min_samples: int | None,
    allow_single_cluster: bool,
    cluster_method: str,
    cluster_columns: List[str],
    drop_ungrouped: bool = True,
) -> sql.DataFrame:
    """Perform HDBSCAN clustering on a Spark DataFrame, per region."""

    def run_hdbscan_per_group(pdf: pd.DataFrame) -> pd.DataFrame:
        clustered_df = run_hdbscan(
            df=pdf,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=allow_single_cluster,
            cluster_method=cluster_method,
            cluster_columns=cluster_columns,
            drop_ungrouped=drop_ungrouped,
        )
        return clustered_df

    output_schema = split_data.schema.add(StructField("group", IntegerType(), True))

    clustered = split_data.groupBy("region")
    print(f"GROUPED BY, {clustered.}")

    # .applyInPandas(
    #     run_hdbscan_per_group,
    #     schema=output_schema,
    # )

    # clustered = clustered.withColumn("global_group", monotonically_increasing_id())

    # return clustered
