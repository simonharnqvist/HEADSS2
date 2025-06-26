import pandas as pd
import dask.dataframe as dd

def partition_by_region(df: pd.DataFrame) -> dd.DataFrame:
    """
    Partitions a DataFrame by unique 'region' values so each region becomes its own Dask partition.
    This is useful if you want to use map_partitions and ensure all values from a region are processed together.

    Parameters:
        df (pd.DataFrame): The input DataFrame. Must include a 'region' column.

    Returns:
        dd.DataFrame: A Dask DataFrame where each partition contains data from exactly one region.
    """
    # Ensure region is categorical and sorted to make downstream joins more efficient
    df = df.copy()
    df['region'] = df['region'].astype('category')

    # Create a Dask DataFrame with one partition per region
    region_frames = [df[df['region'] == region].copy() for region in df['region'].cat.categories]
    region_ddfs = [dd.from_pandas(region_df, npartitions=1) for region_df in region_frames]

    return dd.concat(region_ddfs, interleave_partitions=False)
