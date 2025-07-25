import pandas as pd
import numpy as np
import itertools
from dataclasses import dataclass
from typing import List


def cut_df_by_region(df: pd.DataFrame, minima: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Filter the DataFrame to retain only rows within the specified minima and step bounds for each column.

    :param df: Input pandas DataFrame.
    :param minima: Lower bounds for each split column.
    :param split_columns: Names of columns to apply slicing on.
    :param step: Step size for each column dimension.
    :return: Filtered pandas DataFrame within the specified region.
    """
    for i, value in enumerate(minima):
        df = df[df[split_columns[i]].between(value, value + step[i])]
    return df


def split_dataframes(df: pd.DataFrame, n_regions: int, limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> np.ndarray:
    """
    Create a list of DataFrames by splitting according to value regions.

    :param df: Original input pandas DataFrame.
    :param n_regions: Number of regions to create.
    :param limits: Array of minima per region.
    :param split_columns: Column names to split on.
    :param step: Step size for each dimension.
    :return: Array of region-specific pandas DataFrames.
    """
    split = np.empty(n_regions, dtype=object)
    for i, mins in enumerate(limits):
        split[i] = cut_df_by_region(df, minima=mins, split_columns=split_columns, step=step)
    return split


def concat_dataframes(dfs: List[pd.DataFrame], add_pos: bool = True, pos_col: str = "region") -> pd.DataFrame:
    """
    Concatenate a list of DataFrames and assign region labels if needed.

    :param dfs: List of pandas DataFrames.
    :param add_pos: If True, add a region ID to each chunk.
    :param pos_col: Name of the region ID column.
    :return: A single concatenated pandas DataFrame.
    """
    if add_pos:
        for idx, df in enumerate(dfs):
            df[pos_col] = idx
    return pd.concat(dfs, ignore_index=True)


def get_limits(df: pd.DataFrame, step: np.ndarray, split_columns: List[str]) -> np.ndarray:
    """
    Compute all combinations of minima across split columns.

    :param df: Input DataFrame to analyze.
    :param step: Step size for each column.
    :param split_columns: Columns to compute minima over.
    :return: Array of all minima combinations.
    """
    stats = df.describe()
    mins = [np.arange(stats[col]["min"], stats[col]["max"] - val + stats[col]["max"] / 100, val / 2)
            for col, val in zip(split_columns, step)]
    return np.array(list(itertools.product(*mins)))


def get_step(df: pd.DataFrame, split_columns: List[str], n: int) -> np.ndarray:
    """
    Determine step sizes for each split dimension.

    :param df: Input DataFrame.
    :param split_columns: Columns to calculate range over.
    :param n: Desired number of subdivisions per dimension.
    :return: Step size per dimension.
    """
    df = df[split_columns]
    return (df.max().values - df.min().values) / n


def get_split_data(df: pd.DataFrame, n_regions: int, limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Orchestrate DataFrame splitting and concatenate into a labeled result.

    :param df: Input DataFrame.
    :param n_regions: Total number of regions.
    :param limits: Region lower bounds.
    :param split_columns: Split dimension names.
    :param step: Step size per dimension.
    :return: Final labeled pandas DataFrame across all regions.
    """
    dfs = split_dataframes(df, n_regions=n_regions, limits=limits, split_columns=split_columns, step=step)
    return concat_dataframes(dfs, add_pos=True, pos_col='region')


def get_split_regions(limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Build a DataFrame describing region min/max bounds.

    :param limits: Minima for each region.
    :param split_columns: Column names for splitting.
    :param step: Step size per dimension.
    :return: DataFrame with min and max bounds per region.
    """
    regions = pd.DataFrame(np.hstack((limits, limits + step)),
                           columns=[f"{col}_mins" for col in split_columns] + [f"{col}_max" for col in split_columns])
    return regions.reset_index().rename(columns={'index': 'region'})


def get_stitch_regions(low_cuts: np.ndarray, high_cuts: np.ndarray, split_columns: List[str]) -> pd.DataFrame:
    """
    Create region descriptors for data stitching.

    :param low_cuts: Minimum bounds for each region.
    :param high_cuts: Maximum bounds for each region.
    :param split_columns: Names of dimensions to describe.
    :return: Stitching descriptor DataFrame.
    """
    regions = pd.DataFrame(np.hstack((low_cuts, high_cuts)),
                           columns=[f"{col}_mins" for col in split_columns] + [f"{col}_max" for col in split_columns])
    return regions.reset_index().rename(columns={'index': 'region'})


def get_n_regions(n: int, split_columns: List[str]) -> int:
    """
    Compute the number of distinct regions from dimensional cube counts.

    :param n: Subdivisions per dimension.
    :param split_columns: Names of dimensions.
    :return: Total region count.
    """
    return (2 * n - 1) ** len(split_columns)


def get_minima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Generate adjusted lower bounds for overlapping stitching.

    :param limits: Original minima.
    :param step: Step values per column.
    :return: Offset minima for stitching.
    """
    low = limits + step * 0.25
    for i, minimum in enumerate(np.min(limits, axis=0)):
        low[:, i][low[:, i] == low[:, i].min()] = minimum
    return low


def get_maxima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Generate adjusted upper bounds for overlapping stitching.

    :param limits: Original minima.
    :param step: Step values per column.
    :return: Offset maxima for stitching.
    """
    high = limits + step * 0.75
    for i, maximum in enumerate(np.max(limits, axis=0)):
        high[:, i][high[:, i] == high[:, i].max()] = maximum + step[i]
    return high


@dataclass
class Regions:
    """
    Container for storing split and stitch metadata.

    :param split_data: Region-partitioned Dask DataFrame.
    :param split_regions: Table of region boundaries (min/max).
    :param stitch_regions: Table of stitching overlaps.
    """
    split_data: pd.DataFrame
    split_regions: pd.DataFrame
    stitch_regions: pd.DataFrame


def make_regions(df: pd.DataFrame, n: int, split_columns: List[str]) -> Regions:
    """
    Perform full region decomposition and return partitioned dataset.

    :param df: Input DataFrame to regionally split.
    :param n: Number of cubes per dimension.
    :param split_columns: Columns to use for spatial splitting.
    :return: Structured Regions object with Dask-optimized outputs.
    """
    n_regions = get_n_regions(n=n, split_columns=split_columns)
    step = get_step(df, split_columns=split_columns, n=n)
    limits = get_limits(df, step=step, split_columns=split_columns)
    low_cuts = get_minima(limits, step=step)
    high_cuts = get_maxima(limits, step=step)

    split_data = get_split_data(df, n_regions=n_regions, limits=limits, split_columns=split_columns, step=step)

    split_regions = get_split_regions(limits=limits, split_columns=split_columns, step=step)
    stitch_regions = get_stitch_regions(low_cuts=low_cuts, high_cuts=high_cuts, split_columns=split_columns)

    return Regions(split_data=split_data, split_regions=split_regions, stitch_regions=stitch_regions)