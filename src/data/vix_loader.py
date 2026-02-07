import pandas as pd

def load_vix_csv(path: str) -> pd.Series:
    """
    Loads a VIX CSV and returns a clean pd.Series indexed by Date.
    Tries common VIX column names automatically.
    """
    df = pd.read_csv(path)

    # Date column
    date_col_candidates = [c for c in df.columns if c.lower() in ["date", "datetime", "time"]]
    if not date_col_candidates:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")
    date_col = date_col_candidates[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # Value column (common names)
    value_candidates = ["VIX Close", "CLOSE", "Close", "close", "VIX", "vix"]
    value_col = None
    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        # fallback: first numeric column
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise ValueError(f"No numeric columns found. Columns: {list(df.columns)}")
        value_col = num_cols[0]

    s = df[value_col].astype(float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s
