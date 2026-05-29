from pathlib import Path
from functools import lru_cache

import pandas as pd
from datasets import load_dataset

from app.config import settings

data_path = Path("data")
data_path.mkdir(parents=True, exist_ok=True)
bitext_csv = data_path / "bitext.csv"


@lru_cache(maxsize=1)
def load_customer_service_df() -> pd.DataFrame:
    """
    Load and cache the customer service dataset as a DataFrame.
    """
    if bitext_csv.exists():
        return pd.read_csv(bitext_csv)

    ds = load_dataset(settings.dataset_name, split="train")
    df = ds.to_pandas().copy()

    df["instruction"] = df["instruction"].astype(str)
    df["category"] = df["category"].astype(str).str.upper()
    df["intent"] = df["intent"].astype(str)
    df["response"] = df["response"].astype(str)

    df.to_csv(bitext_csv, index=False)
    return df
