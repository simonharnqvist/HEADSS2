# HEADSS2
This is the new version of HEADSS: HiErArchical Data Splitting and Stitching Software for Non-Distributed Clustering Algorithms. HEADSS2 provides a more predicable API, and enables parallelisation of the most compute-intense steps of the algorithm.

## HEADSS

HEADSS represents a process of splitting big data to avoid the introduction of edge effects and formalise the stitching process to provide a complete feature space. The process is shown (below) with a number of example notebooks for examples of how to use, highlighting best practices and implimentations to avoid.

Example split and stitch boundaries for n = 3 implimentation, where n refers to the number of cuts in each feature in the base layer.
![3x3_split](https://user-images.githubusercontent.com/84581147/170474116-5f718b98-618d-4d61-a95c-c1c7a8012f57.png)
<!-- ![3x3_stitch](https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png) -->
<img src="https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png" width="250" height="250">

The current version supports clustering with HDBSCAN.
> McInnes L, Healy J. Accelerated Hierarchical Density Based Clustering In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

With the ability to split and stitch data while clustering independently if alternative clustering methods are preferred.

## Installation
Currently, HEADSS2 can be installed with `pip install .` from the base directory of this project.

## Usage
Full docs to follow. Example usage below:

#### Create regions for stitching and merging
```python
from headss2 import regions
regs = regions.make_regions(df = data, n = 2, split_columns = ["x", "y"])
```

#### Cluster with HDBSCAN
```python
from headss2 import clustering
clustered = clustering.cluster(split_data = regs.split_data, 
            min_cluster_size = 10, min_samples = 10, allow_single_cluster = False, 
            cluster_method = "eom", cluster_columns = ["x", "y"], drop_ungrouped = True)
```

#### Stitch regions
```python
from headss2 import stitching
stitched = stitching.stitch(
    clustered_data = clustered,
    split_columns: ["x", "y"],
    stitch_regions: regs.stitch_regions
)
```

#### Merge regions
```python
from headss2 import merging
merged = merging.merge_clusters(
  clustered = clustered,
  group_col = "group",
  split_regions = regs.split_regions,
  split_columns = ["x", "y"],
  n_cores = 16)
```

## Contributors
* Simon Harnqvist, Wide-field Astronomy Unit (WFAU), University of Edinburgh. Current maintainer and author of HEADSS2.
* Dennis Crake, formerly of WFAU. Original author of HEADSS.

## Contributing, bug reports, and feature requests

We welcome contributions in any form but particuarly with implementations of additional clustering algorithms. To contribute please fork the project and submit a pull request.

If you have found a (potential) bug, or have ideas for improvements or extensions that you are not able to contribute via a PR, please open a GitHub issue.

## Citation
If using HEADSS2, please cite both this repository and Dennis' <i>Astronomy and Computing</i> paper below for the algorithm and original implementation:
> Crake, DA, Hambly, NC & Mann, RG 2023, 'HEADSS: HiErArchical Data Splitting and Stitching software for
> non-distributed clustering algorithms', Astronomy and Computing, vol. 43, 100709, pp. 1-9.
> https://doi.org/10.1016/j.ascom.2023.100709

## Licensing

The hdbscan package is 3-clause BSD licensed.

>  BSD licence and contact the authors for details on contributing to this code.
