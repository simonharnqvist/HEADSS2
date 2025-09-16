from pyspark import sql
import pandas as pd
from headss2 import make_regions, cluster, stitch, merge_clusters


class HEADSS2:

    def __init__(
        self,
        n: int,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        allow_single_cluster: bool = False,
        clustering_method: str = "eom",
        drop_unclustered: bool = True,
        merge_clusters: bool = True,
        per_cluster_overlap_threshold: float | None = 0.5,
        combined_overlap_threshold: float | None = 0.1,
        min_n_overlap: int | None = 10,
        spark_session: sql.SparkSession | None = None,
    ):

        if spark_session is None:
            spark_session = sql.SparkSession.builder.getOrCreate()

        if (
            not per_cluster_overlap_threshold
            or not combined_overlap_threshold
            or not min_n_overlap
            and merge_clusters
        ):
            raise ValueError(
                "merge_clusters is 'True', but values are missing for one or more of 'per_cluster_overlap_threshold', 'combined_overlap_threshold', 'min_n_overlap'"
            )

        self.n = n
        self.spark_session = spark_session
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.allow_single_cluster = allow_single_cluster
        self.clustering_method = clustering_method
        self.drop_unclustered = drop_unclustered
        self.merge_clusters = merge_clusters
        self.per_cluster_overlap_threshold = per_cluster_overlap_threshold
        self.combined_overlap_threshold = combined_overlap_threshold
        self.min_n_overlap = min_n_overlap

        self.data = None
        self.cluster_columns = None
        self.regions = None
        self.clustered = None
        self.stitched = None
        self.merged = None

    def fit(self, data: sql.DataFrame | pd.DataFrame, cluster_on: list[str]):
        regs = make_regions(
            spark_session=self.spark_session,
            df=data,
            n=self.n,
            cluster_columns=cluster_on,
        )

        self.regions = regs
        self.data = data
        self.cluster_columns = cluster_on

        clustered = cluster(
            split_data=regs.split_data,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            allow_single_cluster=self.allow_single_cluster,
            clustering_method=self.clustering_method,
            cluster_columns=cluster_on,
            drop_unclustered=self.drop_unclustered,
        )

        self.clustered = clustered

        stitched = stitch(
            clustered=clustered,
            cluster_columns=cluster_on,
            stitch_regions=regs.stitch_regions,
        )

        self.stitched = stitched

        if self.merge_clusters:
            self.merged = merge_clusters(
                clustered=clustered,
                cluster_columns=cluster_on,
                min_n_overlap=self.min_n_overlap,
                per_cluster_overlap_threshold=self.per_cluster_overlap_threshold,
                combined_overlap_threshold=self.combined_overlap_threshold,
            )
