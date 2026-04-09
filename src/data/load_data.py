import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean raw dataset
    """
    df = pd.read_csv(file_path, header=1)

    # Rename target column
    df = df.rename(columns={"default payment next month": "target"})

    # Drop ID column if exists
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df