from __future__ import annotations

import pandas as pd


def _to_series_1d(obj, name: str) -> pd.Series:
    """
    Force yfinance output into a clean 1D float Series.
    Handles Series and 1-column DataFrames.
    """
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(f"{name} must be 1-dimensional, got shape {obj.shape}")
        s = obj.iloc[:, 0].copy()
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame, got {type(obj)}")

    s = s.astype(float)
    s.name = name
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_yahoo_adjclose(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    """
    Loads Adjusted Close prices from Yahoo Finance using yfinance.
    Returns a pd.Series indexed by DatetimeIndex, sorted ascending.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "Missing dependency: yfinance. Install with: pip install yfinance"
        ) from e

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker} between {start} and {end}")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = _to_series_1d(df[col], ticker)
    return s


def load_yahoo_close(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    """
    Loads Close prices from Yahoo Finance using yfinance.
    Useful for indices like ^VIX when you just need the index level as signal.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "Missing dependency: yfinance. Install with: pip install yfinance"
        ) from e

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker} between {start} and {end}")

    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' column returned for ticker={ticker}")

    s = _to_series_1d(df["Close"], ticker)
    return s


def prices_to_wealth(prices: pd.Series) -> pd.Series:
    """
    Converts a price series to a normalized wealth series starting at 1.0.
    """
    prices = _to_series_1d(prices, prices.name if prices.name is not None else "price")
    rets = prices.pct_change().fillna(0.0)
    wealth = (1.0 + rets).cumprod()
    wealth.name = f"{prices.name}_wealth"
    return wealth