from pyspark import sql
import pandas as pd
from headss2 import make_regions, cluster, stitch, merge_clusters

class HEADSS2:

    def __init__(self,
                 n: int,
                 min_cluster_size: int,
                    min_samples: int,
                    allow_single_cluster: bool,
                    cluster_method: str,
                    drop_ungrouped: bool = True, merge_clusters = True, 
                    overlap_threshold: float|None = None, 
                    total_threshold: float|None = None, 
                    spark_session: sql.SparkSession|None = None)
        
        if spark_session is None:
            spark_session = sql.SparkSession.builder.getOrCreate()
        
        self.spark_session = spark_session
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.allow_single_cluster = allow_single_cluster
        self.cluster_method = cluster_method
        self.drop_ungrouped = drop_ungrouped
        self.merge_clusters = merge_clusters
        self.overlap_threshold = overlap_threshold
        self.total_threshold = total_threshold

        self.data = None
        self.cluster_columns = None
        self.regions = None
        self.clustered = None
        self.stitched = None
        self.merged = None


    def fit(self, data: sql.DataFrame | pd.DataFrame,
                 cluster_on: list[str]):
        regs = make_regions(spark_session=self.spark_session,
                                        df = data,
                                        n = self.n,
                                        split_columns=cluster_on)
        
        self.regions = regs
        self.data = data
        self.cluster_columns = cluster_on
        
        clustered = cluster(split_data=regs.split_data,
                            min_cluster_size=self.min_cluster_size,
                            allow_single_cluster=self.allow_single_cluster,
                            cluster_method=self.cluster_method,
                            cluster_columns=cluster_on,
                            drop_ungrouped=self.drop_ungrouped)
        
        self.clustered = clustered
        
        stitched = stitch(clustered_data=clustered,
                          split_columns=cluster_on,
                          stitch_regions=regs.split_regions)
        
        self.stitched = stitched
        
        if self.merge_clusters:
            self.merged =  merge_clusters(spark_session=self.spark_session,
                                    clustered_sdf=clustered,
                                    split_regions_df=regs.split_regions,
                                    split_columns=cluster_on,
                                    minimum_members=self.min_samples,
                                    overlap_threshold=self.overlap_threshold,
                                    total_threshold=self.total_threshold)

        

        
        

        