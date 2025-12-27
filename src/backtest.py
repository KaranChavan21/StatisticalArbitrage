"""
Backtesting Module
Realistic backtest engine with transaction costs, margin, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal


def normalize_model_weights(
    model: pd.DataFrame,
    initial_margin_long: float = 0.50,
    initial_margin_short: float = 0.50
) -> pd.DataFrame:
    """
    Normalize portfolio weights for proper position sizing.

    Scales weights such that:
    1. Absolute sum of weights = 1 (unit basket)
    2. Leverage is controlled by initial margin requirements

    Parameters
    ----------
    model : pd.DataFrame
        Model with weight columns
    initial_margin_long, initial_margin_short : float
        Initial margin requirements for long/short positions
    """
    model = model.copy()
    weight_cols = model.columns[model.columns.str.contains("_weight")]
    df_weights = model[weight_cols]

    norm_factor = 1 / df_weights.abs().sum(axis=1).values
    nrows = df_weights.shape[0]

    df_norm = df_weights * norm_factor.reshape((nrows, 1))
    long_margin = np.where(df_norm > 0, np.abs(df_norm), 0).sum(axis=1) * initial_margin_long
    short_margin = np.where(df_norm < 0, np.abs(df_norm), 0).sum(axis=1) * initial_margin_short

    leverage = 1 / (long_margin + short_margin)
    scaling = norm_factor * leverage

    model[weight_cols] = df_weights.values * scaling.reshape((nrows, 1))
    model["intercept"] = model["intercept"] * scaling
    model["spread"] = model["spread"] * scaling

    return model


def set_rates(
    index: pd.DatetimeIndex,
    rfr: float = 0.0,
    short_fee: float = 0.0,
    transaction_costs: float = 0.0,
    margin_fee: Optional[float] = None,
    cash_interest: Optional[float] = None,
    collateral_interest: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate rate schedule for backtest.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Backtest index
    rfr : float
        Risk-free rate (per period)
    short_fee : float
        Stock borrow fee (negative)
    transaction_costs : float
        Round-trip transaction costs (negative)
    margin_fee : float
        Margin borrowing fee (negative)
    cash_interest : float
        Interest on idle cash (positive)
    collateral_interest : float
        Interest on collateral (positive)

    Returns
    -------
    pd.DataFrame
        Rate schedule
    """
    df = pd.DataFrame(index=index)
    df["rfr"] = rfr
    df["short_fee"] = short_fee
    df["transaction"] = transaction_costs
    df["margin_fee"] = margin_fee if margin_fee is not None else -rfr
    df["deposit_int"] = cash_interest if cash_interest is not None else rfr
    df["collateral_int"] = collateral_interest if collateral_interest is not None else rfr
    return df


def backtest(
    df_prices: pd.DataFrame,
    df_weights: pd.DataFrame,
    signal: pd.Series,
    df_rates: pd.DataFrame,
    entry_capital_buffer: float = 0.0,
    update_when: Literal["not_open", "always"] = "always"
) -> pd.DataFrame:
    """
    Run pairs trading backtest.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Price series for all assets
    df_weights : pd.DataFrame
        Hedge ratios/weights for each asset
    signal : pd.Series
        Trading signal (-1 to 1)
    df_rates : pd.DataFrame
        Cost/rate schedule
    entry_capital_buffer : float
        Reserve buffer (reduces position size)
    update_when : str
        Weight update frequency ('always' or 'not_open')

    Returns
    -------
    pd.DataFrame
        Backtest results with equity curve
    """
    ticker_names = df_prices.columns.str.replace("_price", "")

    # Calculate trade weights
    signal = signal.to_frame()
    curr_weights = df_weights.values * signal.values * (1 - entry_capital_buffer)

    if update_when == "always":
        trade_weights = curr_weights
    elif update_when == "not_open":
        trade_weights = np.where(signal == 0, curr_weights, np.nan)
    else:
        raise ValueError(f"Unknown update_when: {update_when}")

    # Calculate positions
    df_pos = pd.DataFrame(
        trade_weights / df_prices.values,
        index=df_prices.index,
        columns=ticker_names + "_pos",
        dtype=float
    )
    df_pos = df_pos.ffill().fillna(0)

    # Build results
    btst = pd.concat([df_prices, df_pos], axis="columns")
    btst["TotalLong"] = np.where(trade_weights > 0, trade_weights, 0).sum(axis=1)
    btst["TotalShort"] = np.where(trade_weights < 0, trade_weights, 0).sum(axis=1)

    # Calculate costs
    df_pos_chg = df_pos.diff().fillna(0)
    equity_used = btst["TotalLong"] + btst["TotalShort"]

    btst["TransactionCost"] = (
        df_rates["transaction"].to_frame().values *
        df_prices.values *
        np.abs(df_pos_chg.values)
    ).sum(axis=1)

    btst["MarginCost"] = df_rates["margin_fee"] * np.where(equity_used > 1, equity_used - 1, 0)
    btst["DepositInterest"] = df_rates["deposit_int"] * np.where(equity_used < 1, 1 - equity_used, 0)
    btst["CollateralInterest"] = df_rates["collateral_int"] * btst["TotalShort"]
    btst["ShortFee"] = df_rates["short_fee"] * btst["TotalShort"]

    # Calculate PnL
    btst["CashIO"] = -(df_prices.values * df_pos_chg.values).sum(axis=1)

    btst["EquityCurve"] = np.exp(
        (df_prices.values * df_pos.values).sum(axis=1)
        + btst["CashIO"].cumsum()
        + btst["TransactionCost"].cumsum()
        + btst["MarginCost"].cumsum()
        + btst["DepositInterest"].cumsum()
        + btst["CollateralInterest"].cumsum()
        + btst["ShortFee"].cumsum()
    )

    return btst


def get_backtest_metrics(
    btst: pd.DataFrame,
    risk_free_rate: pd.Series,
    periods_per_year: float = 252.0
) -> Dict[str, float]:
    """
    Calculate comprehensive backtest metrics.

    Parameters
    ----------
    btst : pd.DataFrame
        Backtest results with EquityCurve column
    risk_free_rate : pd.Series
        Risk-free rate per period
    periods_per_year : float
        Annualization factor

    Returns
    -------
    dict
        Performance metrics
    """
    cumsum_returns = np.log(btst["EquityCurve"])
    returns = cumsum_returns.diff().fillna(0)
    sqrt_ann = np.sqrt(periods_per_year)

    # Basic stats
    excess_returns = returns.values - risk_free_rate.values
    mean_return = excess_returns.mean()
    vol = excess_returns.std()
    ann_return = mean_return * periods_per_year
    ann_vol = vol * sqrt_ann

    # Drawdown
    drawdown = cumsum_returns - cumsum_returns.cummax()
    mdd = drawdown.min()

    isin_dd = drawdown < 0
    grouped_dd = (isin_dd != isin_dd.shift()).cumsum()
    dd_group_ids = grouped_dd[isin_dd]
    if len(dd_group_ids) > 0:
        longest_dd = grouped_dd.groupby(dd_group_ids).count().max()
    else:
        longest_dd = 0

    # Ratios
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    calmar = ann_return / abs(mdd) if mdd != 0 else np.nan

    downside = excess_returns[excess_returns < 0]
    downside_vol = downside.std() * sqrt_ann
    sortino = ann_return / downside_vol if downside_vol != 0 else np.nan

    win_rate = (excess_returns >= 0).mean()

    return {
        "Total Return": float(btst["EquityCurve"].iloc[-1] - 1),
        "Annual Return": float(ann_return),
        "Annual Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Calmar Ratio": float(calmar),
        "Max Drawdown": float(mdd),
        "Longest Drawdown (bars)": int(longest_dd),
        "Win Rate": float(win_rate),
        "Number of Trades": int((btst["TotalLong"].diff().abs() > 0.01).sum()),
    }


def run_walk_forward_backtest(
    df_prices: pd.DataFrame,
    df_log_prices: pd.DataFrame,
    train_window: int,
    test_window: int,
    zscore_window: int = 50,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Walk-forward backtest with rolling reestimation.

    Parameters
    ----------
    df_prices, df_log_prices : pd.DataFrame
        Price data (raw and log)
    train_window : int
        Training window size
    test_window : int
        Test window size
    zscore_window : int
        Z-score calculation window
    entry_threshold, exit_threshold : float
        Signal thresholds

    Returns
    -------
    pd.DataFrame
        Combined backtest results
    """
    from .cointegration import get_OLS, measure_mean_reversion
    from .signals import calc_rolling_zscore, zscore_signal_threshold

    results = []
    n = len(df_prices)

    for start in range(0, n - train_window - test_window, test_window):
        train_end = start + train_window
        test_end = min(train_end + test_window, n)

        # Train
        train_prices = df_log_prices.iloc[start:train_end]
        cols = train_prices.columns
        model = get_OLS(train_prices[cols[0]], train_prices[cols[1]])

        # Test
        test_log = df_log_prices.iloc[train_end:test_end]
        test_raw = df_prices.iloc[train_end:test_end]

        # Calculate spread
        weights = model.iloc[-1][[c for c in model.columns if "_weight" in c]]
        intercept = model.iloc[-1]["intercept"]
        spread = (test_log.values * weights.values).sum(axis=1) - intercept
        spread = pd.Series(spread, index=test_log.index)

        # Signals
        zscore = calc_rolling_zscore(spread, zscore_window)
        signals = zscore_signal_threshold(
            zscore,
            enter_long=-entry_threshold,
            exit_long=exit_threshold,
            enter_short=entry_threshold,
            exit_short=-exit_threshold
        )

        results.append({
            "train_start": df_prices.index[start],
            "train_end": df_prices.index[train_end],
            "test_end": df_prices.index[test_end - 1],
            "half_life": measure_mean_reversion(model["spread"])[0],
            "n_signals": (signals["signal"].diff().abs() > 0).sum(),
        })

    return pd.DataFrame(results)
