"""
Data Providers Module
Unified interface for fetching market data from multiple sources.
- Forex: KCTradings OHLC API
- Equities: Yahoo Finance (yfinance)
"""

import pandas as pd
import numpy as np
import requests
from typing import Union, List, Optional, Literal
from pathlib import Path

# ============================================================================
# FOREX DATA PROVIDER - KCTradings API
# ============================================================================

class ForexDataProvider:
    """
    Fetches forex OHLC data from KCTradings API.
    Supports major currency pairs with various timeframes.
    """

    BASE_URL = "https://ohlc.kctradings.com/ohlc/range/{symbol}/{tf}"

    def __init__(self, api_key: str = "changeme123"):
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key}

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        as_log: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for a forex pair.

        Parameters
        ----------
        symbol : str
            Currency pair (e.g., 'EURUSD', 'GBPUSD', 'AUDUSD')
        timeframe : str
            Timeframe ('m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1')
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        as_log : bool
            If True, return log prices

        Returns
        -------
        pd.DataFrame
            OHLC data with datetime index
        """
        url = self.BASE_URL.format(symbol=symbol, tf=timeframe)
        params = {"start": start, "end": end, "limit": "1000000"}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        df = pd.DataFrame(response.json())
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%dT%H:%M:%S")
        df = df.set_index("datetime")

        if as_log:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = np.log(df[col])

        return df

    def get_close_prices(
        self,
        symbols: Union[str, List[str]],
        timeframe: str,
        start: str,
        end: str,
        as_log: bool = False
    ) -> pd.DataFrame:
        """
        Fetch aligned close prices for multiple forex pairs.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        dfs = {}
        for symbol in symbols:
            df = self.get_ohlc(symbol, timeframe, start, end, as_log=as_log)
            suffix = "_logprice" if as_log else "_price"
            dfs[symbol + suffix] = df["close"]

        result = pd.concat(dfs, axis=1)
        result = result.dropna()
        return result


# ============================================================================
# EQUITIES DATA PROVIDER - Yahoo Finance
# ============================================================================

class EquitiesDataProvider:
    """
    Fetches equity data from Yahoo Finance.
    Supports S&P 500 stocks and ETFs.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def download_prices(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
        price_col: str = "Adj Close"
    ) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance.
        """
        import yfinance as yf

        if isinstance(tickers, str):
            tickers = [tickers]

        data = yf.download(tickers, start=start, end=end, auto_adjust=False)

        if len(tickers) == 1:
            return data[[price_col]].rename(columns={price_col: tickers[0]})
        else:
            return data[price_col]

    def get_prices(
        self,
        tickers: Union[str, List[str]],
        start: str,
        end: str,
        as_log: bool = False,
        from_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get prices with optional caching and log transformation.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Try to load from cache
        if self.cache_dir and from_cache:
            cache_file = self.cache_dir / "sp500_adjclose.parquet"
            if cache_file.exists():
                df_cached = pd.read_parquet(cache_file)
                available = [t for t in tickers if t in df_cached.columns]
                if available:
                    df_prices = df_cached.loc[start:end, available]
                else:
                    df_prices = self.download_prices(tickers, start, end)
            else:
                df_prices = self.download_prices(tickers, start, end)
        else:
            df_prices = self.download_prices(tickers, start, end)

        if as_log:
            suffix = "_logprice"
            df_prices = np.log(df_prices)
        else:
            suffix = "_price"

        df_prices.columns = [str(c) + suffix for c in df_prices.columns]
        return df_prices


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def align_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    how: str = "inner"
) -> tuple:
    """
    Align two DataFrames on their datetime index.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to align
    how : str
        Join method ('inner' or 'outer')

    Returns
    -------
    tuple
        (aligned_df1, aligned_df2)
    """
    idx = df1.index.intersection(df2.index) if how == "inner" else df1.index.union(df2.index)
    return df1.reindex(idx), df2.reindex(idx)


def mask_rollover_bars(
    df: pd.DataFrame,
    rollover_hour: int = 17,
    timezone: str = "America/New_York"
) -> pd.DataFrame:
    """
    Mask forex rollover bars (typically NY 5pm).
    """
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)

    mask = df.index.hour == rollover_hour
    return df[~mask]
