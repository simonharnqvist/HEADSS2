import pandas as pd
from pathlib import Path

def dataset(name: str) -> pd.DataFrame:
    """Retrieve example dataset by name."""
    path = Path("example_data").joinpath(f"{name}.csv")
    return pd.read_csv(path, header=None)
