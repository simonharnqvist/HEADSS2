import pandas as pd
from pathlib import Path
import importlib.resources as pkg_resources

def dataset(name: str) -> pd.DataFrame:
    from headss2 import example_data
    filename = f"{name}.csv"
    with pkg_resources.open_text(example_data, filename) as f:
        df = pd.read_csv(f, header=None)
    return df.iloc[:, 0:2].rename(columns={0: 'x', 1: 'y'})