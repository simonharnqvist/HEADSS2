# HEADSS
HiErArchical Data Splitting and Stitching Software for Non-Distributed Clustering Algorithms

HEADSS represents a process of splitting big data to avoid the introduction of edge effects and formalise the stitching process to provide a complete feature space. The process is shown (below) with a number of example notebooks for examples of how to use, highlighting best practices and implimentations to avoid.

Example split and stitch boundaries for N = 3 implimentation, where N refers to the number of cuts in the base layer.
![3x3_split](https://user-images.githubusercontent.com/84581147/170474116-5f718b98-618d-4d61-a95c-c1c7a8012f57.png)
<!-- ![3x3_stitch](https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png) -->
<img src="https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png" width="250" height="250">

The current version supports clustering with:
- HDBSCAN
> McInnes L, Healy J. Accelerated Hierarchical Density Based Clustering In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

With the ability to split and stitch data while clustering independently if alternative clustering methods are preferred.

## Notebooks included are: 
- Train.ipynb
- quick_start.ipynb
- eval.ipynb

### Train.ipynb
Demonstrates in detail the processes involved with HEADSS and how to extract the split/stitching process to use an alternative clustering algorithm to HDBSCAN.

### quick_start.ipynb
Demonstrates a quick use implimentation to explore the algorithm. A number of example datasets are included and can be called using the provided function or the user can provide thier own.

#### Example usage with data as a pandas.DataFrame:
```
merge = headss_merge(df = data, N = 2, split_columns = ['x', 'y'], merge = True,
                      cluster_columns=['x','y'], min_cluster_size = 10, 
                      min_samples = 10, cluster_method = 'leaf', allow_single_cluster = False,
                 total_threshold = 0.1, overlap_threshold = 0.5, minimum_members = 10) 

# clustering result
merged_df = merge.members_df
```

### eval.ipynb
Demonstrate the performance on all example datasets, producing the plots found in the paper.

## Intallation

To be made available using pip or conda. For now, use the provided requirements.txt or environment.yml files.

## Requirements

All workbooks are `.ipynb`, however HEADSS itself is simply `.py` and can be used provided the requirements are correctly installed.

To install requirements:

```setup
pip install -r requirements.txt
```

To set up an environment:

```
conda env create -f environment.yml
```

If hdbscan is not present simply install after entering the environment using:
```pip install hdbscan```

## Contributing

We welcome contributions in any form but particuarly with implementations of additional clustering algorithms. To contribute please fork the project and submit a pull request. Any assistance with formatting, syntax or anything else feel free to contact the authors.

## Licensing

The hdbscan package is 3-clause BSD licensed.

>  BSD licence and contact the authors for details on contributing to this code.
