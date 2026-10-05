import numpy as np
import pandas as pd


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) != n_before:
        print(
            f"   [WARN] dropped {n_before - len(df)} NaN row(s) from {csv_path}; "
            "the train index in the filename may no longer align"
        )

    if "Label" not in df.columns:
        raise ValueError(f"{csv_path} has no 'Label' column")

    label = df["Label"].astype(int).to_numpy()
    data = df.drop(columns=["Label"]).to_numpy(dtype=float)
    return data, label


def train_index_from_filename(filename: str) -> int:
    parts = filename.split(".", maxsplit=1)[0].split("_")
    return int(parts[parts.index("tr") + 1])
