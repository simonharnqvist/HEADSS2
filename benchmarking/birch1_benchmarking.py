from headss2 import regions, clustering, datasets, merging, stitching
from HEADSS import headss_merge
import time
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

res = []

data = datasets.dataset("birch1")

SPLIT_COLUMNS = ["x", "y"]

for n in [1500, 3000, 6000, 12_000, 24_000, 48_000, 96_000]:
    data_subset = data[0:n]

    t0 = time.time()
    headss_merge(df = data_subset, N = 2, split_columns = SPLIT_COLUMNS, merge = True,
                      cluster_columns=SPLIT_COLUMNS, min_cluster_size = 10, 
                      min_samples = 10, cluster_method = 'eom', allow_single_cluster = False,
                 total_threshold = 0.1, overlap_threshold = 0.5, minimum_members = 10) 
    t1 = time.time()
    print(f"HEADSS1 n = {n} Time taken: ", t1 - t0)
    res.append(["HEADSS", n, t1-t0])


    t2 = time.time()

    regs = regions.make_regions(data_subset, 2, SPLIT_COLUMNS)
    h2_split_data = regs.split_data
    h2_split_regions = regs.split_regions
    clustered = clustering.cluster(split_data = h2_split_data, min_samples = 10, 
                                   min_cluster_size = 10, allow_single_cluster = False, cluster_method ="eom", cluster_columns=SPLIT_COLUMNS)

    stitched = stitching.stitch(clustered, split_columns=SPLIT_COLUMNS, stitch_regions=regs.stitch_regions)
    merging.merge_clusters(stitched, group_col="group", split_regions=h2_split_regions, split_columns=["x", "y"], n_cores=1)
    t3 = time.time()

    print(f"HEADSS2 (1 core) n = {n} Time taken: ", t3 - t2)
    res.append(["HEADSS2 (1 core)", n, t3-t2])

    t4 = time.time()

    regs = regions.make_regions(data_subset, 2, SPLIT_COLUMNS)
    h2_split_data = regs.split_data
    h2_split_regions = regs.split_regions
    clustered = clustering.cluster(split_data = h2_split_data, min_samples = 10, 
                                   min_cluster_size = 10, allow_single_cluster = False, cluster_method ="eom", cluster_columns=SPLIT_COLUMNS)

    stitched = stitching.stitch(clustered, split_columns=SPLIT_COLUMNS, stitch_regions=regs.stitch_regions)
    merging.merge_clusters(stitched, group_col="group", split_regions=h2_split_regions, split_columns=["x", "y"], n_cores=4)
    t5 = time.time()

    print(f"HEADSS2 (4 cores) n = {n} Time taken: ", t5 - t4)
    res.append(["HEADSS2 (4 cores)", n, t5-t4])

res_df = pd.DataFrame(res, columns=["method", "n", "time"])
res_df.to_csv("birch1_profiling.csv", index=False)
