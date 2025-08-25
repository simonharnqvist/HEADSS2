import pandas as pd

def dataset(name: str) -> pd.DataFrame:
    from . import example_data
    import importlib.resources as pkg_resources
    filename = f"{name}.csv"
    with pkg_resources.open_text(example_data, filename) as f:
        df = pd.read_csv(f, header=None)
    return df.iloc[:, 0:2].rename(columns={0: 'x', 1: 'y'})