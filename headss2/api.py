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
        drop_ungrouped: bool = True,
        merge_clusters: bool = True,
        overlap_merge_threshold: float | None = 0.5,
        total_merge_threshold: float | None = 0.1,
        min_merge_members: float | None = 10,
        spark_session: sql.SparkSession | None = None,
    ):

        if spark_session is None:
            spark_session = sql.SparkSession.builder.getOrCreate()

        if (
            not overlap_merge_threshold
            or not total_merge_threshold
            or not min_merge_members
            and merge_clusters
        ):
            raise ValueError(
                "merge_clusters is 'True', but values are missing for one or more of 'overlap_merge_threshold', 'total_merge_treshold', 'min_merge_members'"
            )

        self.n = n
        self.spark_session = spark_session
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.allow_single_cluster = allow_single_cluster
        self.clustering_method = clustering_method
        self.drop_ungrouped = drop_ungrouped
        self.merge_clusters = merge_clusters
        self.overlap_merge_threshold = overlap_merge_threshold
        self.total_merge_threshold = total_merge_threshold
        self.min_merge_members = min_merge_members

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
            split_columns=cluster_on,
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
            drop_ungrouped=self.drop_ungrouped,
        )

        self.clustered = clustered

        stitched = stitch(
            clustered=clustered,
            split_columns=cluster_on,
            stitch_regions=regs.stitch_regions,
        )

        self.stitched = stitched

        if self.merge_clusters:
            self.merged = merge_clusters(
                spark=self.spark_session,
                clustered=clustered,
                split_columns=cluster_on,
                min_merge_members=self.min_merge_members,
                overlap_merge_threshold=self.overlap_merge_threshold,
                total_merge_threshold=self.total_merge_threshold,
            )
