from headss import datasets, regions, cluster
import pytest


@pytest.fixture
def flame_clustered():
    df = datasets.dataset("flame")
    regions_data = regions.region_partition(df=df, n_cubes=2, split_columns=["x", "y"])
    return cluster.cluster(region_partitioned_split_data=regions_data,
                           cluster_columns=["x", "y"],
                           min_cluster_size=20, 
                           min_samples=20, 
                           cluster_method="leaf", 
                           allow_single_cluster=False, 
                           drop_ungrouped=True)

@pytest.fixture
def spiral_clustered():
    df = datasets.dataset("spiral")
    regions_data = regions.region_partition(df=df, n_cubes=2, split_columns=["x", "y"])
    return cluster.cluster(region_partitioned_split_data=regions_data, 
                           cluster_columns=["x", "y"],
                           min_cluster_size=20, 
                           min_samples=20, 
                           cluster_method="leaf", 
                           allow_single_cluster=False, 
                           drop_ungrouped=True)