import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from hdbscan import HDBSCAN
from typing import List, Optional

from pyspark import sql
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    LongType,
    FloatType,
    StringType,
    IntegerType
)

def run_hdbscan(
    arrow_table: pa.Table,
    region: int,
    min_cluster_size: int,
    min_samples: Optional[int],
    allow_single_cluster: bool,
    clustering_method: str,
    cluster_columns: List[str],
    drop_unclustered: bool = True,
    random_seed: int = 11,
) -> pa.Table:
    """
    Cluster objects using HDBSCAN, given a pyarrow.Table, return a pyarrow.Table
    with a string 'cluster' column (formatted region_cluster or '-1').
    """

    np.random.seed(random_seed)

    schema = pa.schema([
        (col_name, pa.float32()) for col_name in cluster_columns
    ] + [("region", pa.int32()), ("cluster", pa.string())]
    )

    if "cluster" not in arrow_table.column_names:
        empty_column = pa.array([None] * len(arrow_table), type=pa.string())
        arrow_table = arrow_table.append_column('cluster', empty_column)
    
    arrow_table = pa.table(arrow_table, schema=schema)

    if arrow_table.num_rows == 0:
        return pa.Table.from_batches([], schema=schema)

    pdf_for_cluster = arrow_table.select(cluster_columns).to_pandas()
    cluster_data = pdf_for_cluster.to_numpy()

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=False,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_method=clustering_method,
        gen_min_span_tree=False,
    ).fit(cluster_data)

    labels = clusterer.labels_.astype(np.str_)
    labels_arr = pa.array(labels, type=pa.string())

    if "cluster" in arrow_table.schema.names:
        idx = arrow_table.schema.get_field_index("cluster")
        arrow_table = arrow_table.set_column(idx, "cluster", labels_arr)
    else:
        arrow_table = arrow_table.append_column("cluster", labels_arr)

    if drop_unclustered:
        arrow_table_filtered = arrow_table.filter(pc.not_equal(arrow_table["cluster"], pa.scalar("-1", type=pa.string())))
        arrow_table = arrow_table_filtered


    prefix = pa.array([f"{region}"] * len(arrow_table), type=pa.string())

    formatted_cluster = pc.if_else(
        pc.not_equal(arrow_table["cluster"], pa.scalar("-1", type=pa.string())),
        pc.binary_join_element_wise(prefix, arrow_table["cluster"], "_"),
        pa.scalar("-1", type=pa.string())
    )

    cluster_idx = arrow_table.schema.get_field_index("cluster")
    arrow_table = arrow_table.set_column(cluster_idx, "cluster", formatted_cluster)

    return arrow_table


def cluster(
    split_data: sql.DataFrame,
    min_cluster_size: int,
    min_samples: Optional[int],
    allow_single_cluster: bool,
    clustering_method: str,
    cluster_columns: List[str],
    drop_unclustered: bool = True,
) -> sql.DataFrame:
    """
    Perform HDBSCAN clustering on a Spark DataFrame, by region, using applyInArrow.
    """

    split_data = split_data.cache()

    def run_hdbscan_per_region_arrow(table: pa.Table) -> pa.Table:
        region_col = table.column("region")
        unique_regions = set(region_col.to_pylist())

        assert len(unique_regions) == 1, f"Expected only one region, got: {unique_regions}"
        region_id = unique_regions.pop()

        return run_hdbscan(
            arrow_table=table,
            region=region_id,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=allow_single_cluster,
            clustering_method=clustering_method,
            cluster_columns=cluster_columns,
            drop_unclustered=drop_unclustered,
        )

    schema_fields = [
        StructField(col, FloatType(), True)
        for col in cluster_columns
    ] + [
        StructField("region", IntegerType(), True),
        StructField("cluster", StringType(), True),
    ]
    output_schema = StructType(schema_fields)

    clustered = split_data.groupBy("region").applyInArrow(
        run_hdbscan_per_region_arrow,
        schema=output_schema
    )

    return clustered
