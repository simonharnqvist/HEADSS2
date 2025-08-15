import pandas as pd
from pathlib import Path

def dataset(name: str) -> pd.DataFrame:
    """Retrieve example dataset by name. Current datasets are 'a3', 'Aggregation', 'flame', 'pathbased', 'spiral', 'D31', 'birch1', 'jain', t4_8k', 'worms'"""
    script_path = Path(__file__).resolve().parent.parent
    path = script_path.joinpath("example_data").joinpath(f"{name}.csv")
    return pd.read_csv(path, header=None).iloc[:, 0:2].rename(columns = {0:'x', 1:'y'})
