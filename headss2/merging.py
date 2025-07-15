import pandas as pd
from typing import List, Tuple

def describe_clusters(clustered: pd.DataFrame, cluster_col:str = 'group') -> pd.DataFrame:
    """Generate summary statistics (min, max, size) per cluster"""

    return (
        clustered.groupby(cluster_col)[["x", "y"]]
        .agg(["min", "max", "count"])
        .set_axis([f"{col}_{stat}" for col, stat in clustered.groupby(cluster_col)[["x", "y"]]
                   .agg(["min", "max", "count"]).columns], axis=1)
        .reset_index()
        .melt(id_vars=cluster_col, var_name="column_stat", value_name="value")
        .assign(
            column=lambda df: df["column_stat"].str.split("_").str[0],
            stat=lambda df: df["column_stat"].str.split("_").str[1]
        )
        .pivot_table(index=[cluster_col, "column"], columns="stat", values="value")
        .reset_index()[[cluster_col, "column", "count", "min", "max"]]
        .rename_axis(None, axis=1)
    )

def find_overlapping_clusters(cluster_descriptions: pd.DataFrame, split_columns: list) -> pd.DataFrame:
    """Detects one-directional overlaps between cluster pairs across specified axes."""

    all_matches = []

    for col in split_columns:
        axis_df = cluster_descriptions[cluster_descriptions["column"] == col]

        # Cartesian product: merge clusters on current axis
        pairs = axis_df.merge(axis_df, on="column", suffixes=("_1", "_2"))

        # Keep only directional (non-self) pairs
        pairs = pairs[pairs["group_1"] != pairs["group_2"]]

        # Bounding box overlap condition
        overlap = (
            (pairs["min_2"] < pairs["min_1"]) &
            (pairs["max_2"] > pairs["min_1"])
        )

        # Extract directional overlaps only (left → right)
        matched = pairs.loc[overlap, ["group_1", "group_2"]]
        matched.columns = ["group1", "group2"]
        all_matches.append(matched)

    # Combine all matches and remove exact duplicates (not reversed ones)
    result = pd.concat(all_matches, ignore_index=True).drop_duplicates()

    # Sort for consistent output style
    result = result.sort_values(by=["group1", "group2"]).reset_index(drop=True)
    return result

