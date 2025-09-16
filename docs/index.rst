HEADSS2 documentation
=====================
This is the new version of HEADSS: *HiErArchical Data Splitting and Stitching Software for Non-Distributed Clustering Algorithms*. HEADSS2 provides a more predicable API, and enables parallelisation of the most compute-intense steps of the algorithm.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api



HEADSS
******

HEADSS represents a process of splitting big data to avoid the introduction of edge effects and formalise the stitching process to provide a complete feature space. 

Example split and stitch boundaries for n = 3 implementation, where n refers to the number of cuts in each feature in the base layer.

	

.. image:: https://user-images.githubusercontent.com/84581147/170474116-5f718b98-618d-4d61-a95c-c1c7a8012f57.png

.. image:: https://user-images.githubusercontent.com/84581147/170474111-fe226e70-14d4-4408-b4f0-61451f06b48a.png


The current version supports clustering with HDBSCAN.
   McInnes L, Healy J. Accelerated Hierarchical Density Based Clustering In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42. 2017

With the ability to split and stitch data while clustering independently if alternative clustering methods are preferred.

Installation
************
Currently, HEADSS2 can be installed with ``pip install .`` from the base directory of this project.

Example datasets
****************

The following example datasets are provided with HEADSS2:

.. image:: https://raw.githubusercontent.com/simonharnqvist/HEADSS2/refs/heads/docs/datasets.png

Example usage
*************

**Get example dataset**

.. code-block:: python

   from headss2 import datasets
   data = datasets.dataset("a3")

**Create regions for stitching and merging**

.. code-block:: python

   from headss2 import regions
   regs = regions.make_regions(df=data, n=2, cluster_columns=["x", "y"])


**Cluster with HDBSCAN**

.. code-block:: python

   from headss2 import clustering
   clustered = clustering.cluster(split_data = regs.split_data, 
               min_cluster_size = 10, min_samples = 10, allow_single_cluster = False, 
               cluster_method = "eom", cluster_columns = ["x", "y"], drop_unclustered = True)


**Stitch regions**

.. code-block:: python

   from headss2 import stitching
   stitched = stitching.stitch(
      clustered_data = clustered,
      cluster_columns: ["x", "y"],
      stitch_regions: regs.stitch_regions
   )


**Merge regions**

.. code-block:: python

   from headss2 import merging
   merged = merging.merge_clusters(
   clustered = clustered,
   cluster_col = "cluster",
   split_regions = regs.split_regions,
   cluster_columns = ["x", "y"],
   n_cores = 16)


Contributors
************
* Simon Harnqvist, Wide-field Astronomy Unit (WFAU), University of Edinburgh. Current maintainer and author of HEADSS2.
* Dennis Crake, formerly of WFAU. Original author of HEADSS.

Contributing, bug reports, and feature requests
***********************************************

We welcome contributions in any form but particuarly with implementations of additional clustering algorithms. To contribute please fork the project and submit a pull request.

If you have found a (potential) bug, or have ideas for improvements or extensions that you are not able to contribute via a PR, please open a GitHub issue.

Citation
********
If using HEADSS2, please cite both this repository and Dennis' <i>Astronomy and Computing</i> paper below for the algorithm and original implementation:

   Crake, DA, Hambly, NC & Mann, RG 2023, 'HEADSS: HiErArchical Data Splitting and Stitching software for
   non-distributed clustering algorithms', Astronomy and Computing, vol. 43, 100709, pp. 1-9.
   https://doi.org/10.1016/j.ascom.2023.100709

Licensing
*********

The hdbscan package is 3-clause BSD licensed.
BSD licence and contact the authors for details on contributing to this code.
