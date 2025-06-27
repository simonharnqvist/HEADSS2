from headss import datasets, regions, cluster
import pytest


@pytest.fixture
def flame_clustered():
    df = datasets.dataset("flame")
    regions_obj = regions.make_regions(df=df, n_cubes=2, split_columns=["x", "y"])
    return cluster.cluster(region_partitioned_split_data=regions_obj.split_data,
                           cluster_columns=["x", "y"],
                           min_cluster_size=20, 
                           min_samples=20, 
                           cluster_method="leaf", 
                           allow_single_cluster=False, 
                           drop_ungrouped=True), regions_obj.stitch_regions

@pytest.fixture
def spiral_clustered():
    df = datasets.dataset("spiral")
    regions_obj = regions.make_regions(df=df, n_cubes=2, split_columns=["x", "y"])
    return cluster.cluster(region_partitioned_split_data=regions_obj.split_data, 
                           cluster_columns=["x", "y"],
                           min_cluster_size=20, 
                           min_samples=20, 
                           cluster_method="leaf", 
                           allow_single_cluster=False, 
                           drop_ungrouped=True), regions_obj.stitch_regions

@pytest.fixture
def a3_clustered():
    df = datasets.dataset("a3")
    regions_obj = regions.make_regions(df=df, n_cubes=2, split_columns=["x", "y"])
    return cluster.cluster(region_partitioned_split_data=regions_obj.split_data, 
                           cluster_columns=["x", "y"],
                           min_cluster_size=20, 
                           min_samples=20, 
                           cluster_method="leaf", 
                           allow_single_cluster=False, 
                           drop_ungrouped=True), regions_obj.stitch_regions