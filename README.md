# HEADSS
HiErArchical Data Splitting and Stitching Software for Non-Distributed Clustering Algorithms

HEADSS represents a process of splitting big data to avoid the introduction of edge effects and formalise the stitching process to provide a complete feature space. The process is shown (below) with a number of example notebooks for examples of how to use, highlighting best practices and implimentations to avoid.

Example split and stitch boundaries for N = 3 implimentation, where N refers to the number of cuts in the base layer.
![3x3_split](https://user-images.githubusercontent.com/84581147/170474116-5f718b98-618d-4d61-a95c-c1c7a8012f57.png)
<!-- ![3x3_stitch](https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png) -->
<img src="https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png" width="250" height="300">

The current version supports clustering with:
- HDBSCAN

With the ability to split and stitch data while clustering independently if alternative methods are preferred.

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

> Notebooks included are: 
- Train.ipynb
- quick_start.ipynb
- eval.ipynb

> Train.ipynb
Demonstrates in detail the processes involved with HEADSS and how to extract the split/stitching process to use an alternative clustering algorithm to HDBSCAN.

> quick_start.ipynb
Demonstrates a quick use implimentation to explore the algorithm. A number of example datasets are included and can be called using the provided function or the user can provide thier own.

> eval.ipynb
Demonstrate the performance on all example datasets, producing the plots found in the paper.

## Contributing

>  BSD licence and contact the authors for details on contributing to this code.
