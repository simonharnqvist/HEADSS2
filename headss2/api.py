from pyspark import sql
import pandas as pd
from headss2 import make_regions, cluster, stitch, cluster_merge


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
        bound_region_point_overlap_threshold: float | None = 0.5,
        total_point_overlap_threshold: float | None = 0.1,
        min_n_overlap: int | None = 10,
        spark_session: sql.SparkSession | None = None,
    ):
        """Create HEADSS2 model.

        Args:
            n (int): Number of bins per dimension for splitting.
            min_cluster_size (int, optional): Minimum permitted cluster size. Defaults to 5.
            min_samples (int | None, optional): Minimum number of samples for clustering. Defaults to None.
            allow_single_cluster (bool, optional): Whether to allow HDBSCAN to return a single cluster. Defaults to False.
            clustering_method (str, optional): HDBSCAN clustering method - choose from 'leaf' or 'eom'. Defaults to "eom".
            drop_unclustered (bool, optional): Whether to remove points that have not been assigned to a cluster. Defaults to True.
            merge_clusters (bool, optional): Whether to merge clusters based on overlaps. Defaults to True.
            bound_region_point_overlap_threshold (float | None, optional): Minimum threshold for merging: fraction of joint data points lying within the bound overlap region divided by the smallest of the two clusters. Previously known as 'total threshold'. Defaults to 0.5.
            total_point_overlap_threshold (float | None, optional): Minimum threshold for merging: fraction of all joint data points divided by the smallest of the two clusters. Previously known as 'overlap threshold'. Defaults to 0.1.
            min_n_overlap (int | None, optional): Minimum number of overlapping points to allow merging. Defaults to 10.
            spark_session (sql.SparkSession | None, optional): Spark session. Generated if not provided. Defaults to None.

        Raises:
            ValueError: if merging is True but thresholds are not given
        """

        if spark_session is None:
            spark_session = sql.SparkSession.builder.getOrCreate()

        if (
            not bound_region_point_overlap_threshold
            or not total_point_overlap_threshold
            or not min_n_overlap
            and merge_clusters
        ):
            raise ValueError(
                "merge_clusters is 'True', but values are missing for one or more of 'bound_region_point_overlap_threshold', 'total_point_overlap_threshold', 'min_n_overlap'"
            )

        self.n = n
        self.spark_session = spark_session
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.allow_single_cluster = allow_single_cluster
        self.clustering_method = clustering_method
        self.drop_unclustered = drop_unclustered
        self.merge_clusters = merge_clusters
        self.bound_region_point_overlap_threshold = bound_region_point_overlap_threshold
        self.total_point_overlap_threshold = total_point_overlap_threshold
        self.min_n_overlap = min_n_overlap

        self.data = None
        self.cluster_columns = None
        self.regions = None
        self.clustered = None
        self.stitched = None
        self.merged = None

    def fit(self, data: sql.DataFrame | pd.DataFrame, cluster_on: list[str]):
        """Cluster data by fitting HEADSS2 model.

        Args:
            data (sql.DataFrame | pd.DataFrame): Data to cluster.
            cluster_on (list[str]): Columns to cluster on.

        Returns:
            sql.DataFrame: Clustered/stitched and potentially merged data.
        """

        if isinstance(data, pd.DataFrame):
            data = self.spark_session.createDataFrame(data)
            
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
        clustered = clustered.cache()

        self.clustered = clustered

        stitched = stitch(
            clustered=clustered,
            cluster_columns=cluster_on,
            stitch_regions=regs.stitch_regions,
        )

        self.stitched = stitched

        if self.merge_clusters:
            merged = cluster_merge(
                clustered=stitched,
                cluster_columns=cluster_on,
                min_n_overlap=self.min_n_overlap,
                bound_region_point_overlap_threshold=self.bound_region_point_overlap_threshold,
                total_point_overlap_threshold=self.total_point_overlap_threshold,
                split_regions = self.regions.split_regions
            )
            self.merged = merged
            return merged
        
        else:
            return stitched
