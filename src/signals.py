"""
Trading Signals Module
Z-score based signal generation for pairs trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calc_rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling z-score of spread.

    Parameters
    ----------
    spread : pd.Series
        Cointegration spread
    window : int
        Rolling window for mean and std calculation

    Returns
    -------
    pd.Series
        Z-score series
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=1)
    zscore = (spread - mean) / std
    return zscore.rename("zscore")


def zscore_signal_threshold(
    zscore: pd.Series,
    enter_long: float = -2.0,
    exit_long: float = 0.0,
    enter_short: float = 2.0,
    exit_short: float = 0.0
) -> pd.DataFrame:
    """
    Threshold-based signal generation with hysteresis.

    Enter long when z < enter_long, exit when z > exit_long
    Enter short when z > enter_short, exit when z < exit_short

    Parameters
    ----------
    zscore : pd.Series
        Z-score series
    enter_long, exit_long : float
        Long entry/exit thresholds
    enter_short, exit_short : float
        Short entry/exit thresholds

    Returns
    -------
    pd.DataFrame
        Signals with position and remarks
    """
    if not (enter_long < exit_long) or not (enter_short > exit_short):
        raise ValueError("Invalid entry/exit thresholds")

    df = zscore.to_frame()
    df["signal"] = np.nan

    # Set entry signals
    df["signal"] = np.where(zscore <= enter_long, 1, df["signal"])
    df["signal"] = np.where(zscore >= enter_short, -1, df["signal"])

    # Set exit signals
    last_signal = df["signal"].ffill()
    df["signal"] = np.where((last_signal > 0) & (zscore >= exit_long), 0, df["signal"])
    df["signal"] = np.where((last_signal < 0) & (zscore <= exit_short), 0, df["signal"])

    # Clean start/end
    df.loc[df.index[0], "signal"] = 0
    df.loc[df.index[-1], "signal"] = 0
    df["signal"] = df["signal"].ffill()

    # Add remarks
    before, now = df["signal"].shift(1), df["signal"]
    df["remarks"] = pd.NA
    df.loc[(before == 0) & (now > 0), "remarks"] = "Enter Long"
    df.loc[(before == 0) & (now < 0), "remarks"] = "Enter Short"
    df.loc[(before > 0) & (now == 0), "remarks"] = "Exit Long"
    df.loc[(before < 0) & (now == 0), "remarks"] = "Exit Short"
    df.loc[(before > 0) & (now < 0), "remarks"] = "Long to Short"
    df.loc[(before < 0) & (now > 0), "remarks"] = "Short to Long"
    df.loc[df.index[0], "remarks"] = "Start"
    df.loc[df.index[-1], "remarks"] = "End"

    return df


def zscore_signal_linear(
    zscore: pd.Series,
    threshold_long: float = -3.0,
    threshold_short: float = 3.0
) -> pd.DataFrame:
    """
    Linear scaling signal generation.
    Position size scales linearly with z-score magnitude.

    Parameters
    ----------
    zscore : pd.Series
        Z-score series
    threshold_long, threshold_short : float
        Full position thresholds

    Returns
    -------
    pd.DataFrame
        Signals with fractional position size
    """
    if not (threshold_long < 0) or not (threshold_short > 0):
        raise ValueError("Invalid threshold parameters")

    df = zscore.to_frame()
    df["signal"] = np.nan

    # Linear scaling
    df["signal"] = np.where(
        zscore < 0,
        np.minimum(zscore / threshold_long, 1.0),
        df["signal"]
    )
    df["signal"] = np.where(
        zscore > 0,
        -np.minimum(zscore / threshold_short, 1.0),
        df["signal"]
    )
    df["signal"] = np.where(zscore == 0, 0, df["signal"])

    # Clean start/end
    df.loc[df.index[0], "signal"] = 0
    df.loc[df.index[-1], "signal"] = 0
    df["signal"] = df["signal"].ffill()

    # Add remarks
    before, now = df["signal"].shift(1), df["signal"]
    df["remarks"] = pd.NA
    df.loc[(before <= 0) & (now > 0), "remarks"] = "Enter Long"
    df.loc[(before >= 0) & (now < 0), "remarks"] = "Enter Short"
    df.loc[(before != 0) & (now == 0), "remarks"] = "Exit"
    df.loc[(before < 0) & (now < before), "remarks"] = "Increase Short"
    df.loc[(before > 0) & (now > before), "remarks"] = "Increase Long"
    df.loc[df.index[0], "remarks"] = "Start"
    df.loc[df.index[-1], "remarks"] = "End"

    return df


def generate_signals_debounced(
    zscore: pd.Series,
    entry: float = 3.0,
    exit_: float = 0.5,
    hold_min: int = 24,
    buffer: float = 0.2
) -> pd.Series:
    """
    Debounced signal generation with minimum hold period.
    Prevents rapid flip-flopping in noisy conditions.

    Parameters
    ----------
    zscore : pd.Series
        Z-score series
    entry : float
        Entry threshold (absolute value)
    exit_ : float
        Exit threshold
    hold_min : int
        Minimum bars to hold position
    buffer : float
        Additional buffer for entry (stricter entry)

    Returns
    -------
    pd.Series
        Position signals
    """
    pos = pd.Series(0, index=zscore.index, dtype=float)
    in_pos = 0.0
    prev_z = None
    since_entry = 0
    e_in = entry + buffer

    for t, z in zscore.dropna().items():
        if prev_z is not None:
            # Entry conditions
            enter_long = (in_pos == 0 and prev_z >= -e_in and z < -e_in)
            enter_short = (in_pos == 0 and prev_z <= e_in and z > e_in)

            # Exit conditions (only after hold_min bars)
            exit_long = (in_pos > 0 and since_entry >= hold_min and z > -exit_)
            exit_short = (in_pos < 0 and since_entry >= hold_min and z < exit_)

            if enter_long:
                in_pos = 1.0
                since_entry = 0
            elif enter_short:
                in_pos = -1.0
                since_entry = 0
            elif exit_long or exit_short:
                in_pos = 0.0
                since_entry = 0
            else:
                since_entry += 1

        pos[t] = in_pos
        prev_z = z

    return pos
