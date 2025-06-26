import pandas as pd
import numpy as np
import itertools
from dataclasses import dataclass
from typing import List
from headss.partition import partition_by_region
import dask.dataframe as dd


def cut_df_by_region(df: pd.DataFrame, minima: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Filters a DataFrame based on value ranges defined by minima and step size for each split column.

    :param df: Input DataFrame.
    :param minima: Array of minimum values per split dimension.
    :param split_columns: List of column names to split on.
    :param step: Step size for each split dimension.
    :return: Filtered DataFrame.
    """
    for i, value in enumerate(minima):
        df = df[df[split_columns[i]].between(value, value + step[i])]
    return df


def split_dataframes(df: pd.DataFrame, n_regions: int, limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> np.ndarray:
    """
    Splits the DataFrame into multiple regions based on limits.

    :param df: Input DataFrame.
    :param n_regions: Number of regions to split into.
    :param limits: Array of lower-bound limit values for each region.
    :param split_columns: Columns to split on.
    :param step: Step size for each split column.
    :return: Array of split DataFrames.
    """
    split = np.empty(n_regions, dtype=object)
    for i, mins in enumerate(limits):
        split[i] = cut_df_by_region(df, minima=mins, split_columns=split_columns, step=step)
    return split


def concat_dataframes(dfs: List[pd.DataFrame], add_pos: bool = True, pos_col: str = "region") -> pd.DataFrame:
    """
    Concatenates a list of DataFrames into a single DataFrame with optional region index.

    :param dfs: List of DataFrames to concatenate.
    :param add_pos: Whether to add region index as a column.
    :param pos_col: Name of the region index column.
    :return: Concatenated DataFrame.
    """
    if add_pos:
        for idx, df in enumerate(dfs):
            df[pos_col] = idx
    return pd.concat(dfs, ignore_index=True)


def get_limits(df: pd.DataFrame, step: np.ndarray, split_columns: List[str]) -> np.ndarray:
    """
    Calculates all combinations of minima across split dimensions.

    :param df: Input DataFrame.
    :param step: Step size for each dimension.
    :param split_columns: Columns to evaluate.
    :return: Array of minima combinations.
    """
    stats = df.describe()
    mins = [np.arange(stats[col]["min"], stats[col]["max"] - val + stats[col]["max"] / 100, val / 2) for col, val in zip(split_columns, step)]
    return np.array(list(itertools.product(*mins)))


def get_step(df: pd.DataFrame, split_columns: List[str], n_cubes: int) -> np.ndarray:
    """
    Calculates step sizes for each split column.

    :param df: Input DataFrame.
    :param split_columns: Columns to split on.
    :param n_cubes: Number of intended cubes per dimension.
    :return: Step sizes.
    """
    df = df[split_columns]
    return (df.max().values - df.min().values) / n_cubes


def get_split_data(df: pd.DataFrame, n_regions: int, limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Orchestrates the full region splitting and combination process.

    :param df: Input DataFrame.
    :param n_regions: Number of regions to produce.
    :param limits: Array of minima defining regions.
    :param split_columns: Columns to split on.
    :param step: Step size for each dimension.
    :return: Final concatenated DataFrame with region labels.
    """
    dfs = split_dataframes(df, n_regions=n_regions, limits=limits, split_columns=split_columns, step=step)
    return concat_dataframes(dfs, add_pos=True, pos_col='region')


def get_split_regions(limits: np.ndarray, split_columns: List[str], step: np.ndarray) -> pd.DataFrame:
    """
    Constructs a DataFrame showing the min and max bounds of each region.

    :param limits: Array of region minima.
    :param split_columns: Names of split columns.
    :param step: Step size per column.
    :return: DataFrame of regions with min/max boundaries.
    """
    regions = pd.DataFrame(np.hstack((limits, limits + step)),
                           columns=[f"{col}_mins" for col in split_columns] + [f"{col}_max" for col in split_columns])
    return regions.reset_index().rename(columns={'index': 'region'})


def get_stitch_regions(low_cuts: np.ndarray, high_cuts: np.ndarray, split_columns: List[str]) -> pd.DataFrame:
    """
    Combines low and high region bounds into a single stitched region DataFrame.

    :param low_cuts: Array of lower bounds.
    :param high_cuts: Array of upper bounds.
    :param split_columns: Names of split columns.
    :return: DataFrame of stitch regions.
    """
    regions = pd.DataFrame(np.hstack((low_cuts, high_cuts)),
                           columns=[f"{col}_mins" for col in split_columns] + [f"{col}_max" for col in split_columns])
    return regions.reset_index().rename(columns={'index': 'region'})


def get_n_regions(n_cubes: int, split_columns: List[str]) -> int:
    """
    Calculates number of total regions to be generated.

    :param n_cubes: Number of subdivisions per dimension.
    :param split_columns: List of columns defining dimensions.
    :return: Number of regions.
    """
    return (2 * n_cubes - 1) ** len(split_columns)


def get_minima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Calculates adjusted lower bounds for stitching regions.

    :param limits: Original limit values.
    :param step: Step size per column.
    :return: Adjusted minima array.
    """
    low = limits + step * 0.25
    for i, minimum in enumerate(np.min(limits, axis=0)):
        low[:, i][low[:, i] == low[:, i].min()] = minimum
    return low


def get_maxima(limits: np.ndarray, step: np.ndarray) -> np.ndarray:
    """
    Calculates adjusted upper bounds for stitching regions.

    :param limits: Original limit values.
    :param step: Step size per column.
    :return: Adjusted maxima array.
    """
    high = limits + step * 0.75
    for i, maximum in enumerate(np.max(limits, axis=0)):
        high[:, i][high[:, i] == high[:, i].max()] = maximum + step[i]
    return high


@dataclass
class Regions:
    """
    Container for split and stitched region metadata and data.
    """
    split_data: pd.DataFrame
    split_regions: pd.DataFrame
    stitch_regions: pd.DataFrame


def make_regions(df: pd.DataFrame, n_cubes: int, split_columns: List[str]) -> Regions:
    """
    Generates region metadata and regioned data from input DataFrame.

    :param df: Input DataFrame to split.
    :param n_cubes: Number of subdivisions per dimension.
    :param split_columns: Columns to split on.
    :return: Regions object with all outputs.
    """
    n_regions = get_n_regions(n_cubes=n_cubes, split_columns=split_columns)
    step = get_step(df, split_columns=split_columns, n_cubes=n_cubes)
    limits = get_limits(df, step=step, split_columns=split_columns)
    low_cuts = get_minima(limits, step=step)
    high_cuts = get_maxima(limits, step=step)

    split_data = get_split_data(df, n_regions=n_regions, limits=limits, split_columns=split_columns, step=step)
    split_regions = get_split_regions(limits=limits, split_columns=split_columns, step=step)
    stitch_regions = get_stitch_regions(low_cuts=low_cuts, high_cuts=high_cuts, split_columns=split_columns)

    return Regions(split_data=split_data, split_regions=split_regions, stitch_regions=stitch_regions)

def region_partition(df: pd.DataFrame, n_cubes: int, split_columns: List[str]) -> dd.DataFrame:
    regions_obj = make_regions(df=df, n_cubes=n_cubes, split_columns=split_columns)
    return partition_by_region(df = regions_obj.split_data)