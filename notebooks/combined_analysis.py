#!/usr/bin/env python3
"""
Statistical Arbitrage Portfolio Analysis
Combined Forex and Equities Cointegration Analysis

This module demonstrates a comprehensive cointegration-based pairs trading framework
applied to both FX currency pairs and US equities.

Author: Karan Chavan
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Analysis configuration parameters."""

    # Time periods (based on QF603 project: Jan 2015 - Oct 2025)
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2020-12-31"
    VALIDATION_START = "2021-01-01"
    VALIDATION_END = "2022-12-31"
    TEST_START = "2023-01-01"
    TEST_END = "2025-10-31"

    # Model parameters
    BETA_WINDOW = 200           # Rolling OLS window
    ZSCORE_WINDOW = 50          # Z-score calculation window

    # Trading parameters (Bayesian optimized values)
    ENTRY_THRESHOLD = 2.0       # Z-score entry threshold
    EXIT_THRESHOLD = 0.5        # Z-score exit threshold
    TRANSACTION_COSTS = 0.002   # 20 bps per trade

    # Risk parameters
    RISK_FREE_RATE = 0.04       # Annual risk-free rate
    PERIODS_PER_YEAR = 252      # Trading days

    # Bayesian Optimization parameters
    BO_ITERATIONS = 50          # Number of BO iterations
    BO_INIT_POINTS = 10         # Initial random samples


# ============================================================================
# FOREX PAIRS ANALYSIS
# ============================================================================

def analyze_forex_pair():
    """
    Analyze cointegration in forex currency pairs.
    Uses KCTradings OHLC API for data retrieval.
    """
    print("\n" + "=" * 70)
    print("FOREX COINTEGRATION ANALYSIS")
    print("Pair: AUDUSD / USDCAD")
    print("=" * 70)

    from src.data_providers import ForexDataProvider
    from src.cointegration import (
        get_integration, pair_cointegration_tests,
        get_OLS, get_rolling_OLS, measure_mean_reversion, halflife_ou
    )
    from src.signals import calc_rolling_zscore, zscore_signal_threshold
    from src.backtest import backtest, set_rates, get_backtest_metrics

    # 1. Data Retrieval
    print("\n[1] Fetching forex data...")
    forex = ForexDataProvider()

    try:
        df_aud = forex.get_ohlc("AUDUSD", "m15", "2020-01-01", "2025-01-01")
        df_cad = forex.get_ohlc("USDCAD", "m15", "2020-01-01", "2025-01-01")
        print(f"    AUDUSD: {len(df_aud):,} bars")
        print(f"    USDCAD: {len(df_cad):,} bars")
    except Exception as e:
        print(f"    [!] API unavailable: {e}")
        print("    Using simulated data for demonstration...")
        # Create simulated cointegrated forex data
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", "2025-01-01", freq="15min")
        n = len(dates)
        common = np.cumsum(np.random.randn(n) * 0.0001)
        df_aud = pd.DataFrame({
            "close": 0.7 + common + np.cumsum(np.random.randn(n) * 0.00005)
        }, index=dates)
        df_cad = pd.DataFrame({
            "close": 1.3 - 0.5 * common + np.cumsum(np.random.randn(n) * 0.00005)
        }, index=dates)
        print(f"    Simulated AUDUSD: {len(df_aud):,} bars")
        print(f"    Simulated USDCAD: {len(df_cad):,} bars")

    # Align and convert to log prices
    common_idx = df_aud.index.intersection(df_cad.index)
    y_log = np.log(df_aud.loc[common_idx, "close"]).rename("AUDUSD_logprice")
    x_log = np.log(df_cad.loc[common_idx, "close"]).rename("USDCAD_logprice")

    print(f"    Aligned: {len(common_idx):,} observations")

    # 2. Integration Tests
    print("\n[2] Order of Integration Tests...")
    i_aud = get_integration(y_log)
    i_cad = get_integration(x_log)
    print(f"    AUDUSD: I({i_aud})")
    print(f"    USDCAD: I({i_cad})")

    # 3. Cointegration Tests
    print("\n[3] Cointegration Tests...")
    train_mask = (y_log.index >= Config.TRAIN_START) & (y_log.index <= Config.TRAIN_END)
    y_train = y_log[train_mask]
    x_train = x_log[train_mask]

    coint_results = pair_cointegration_tests(y_train, x_train)
    print(coint_results.to_string(index=False))

    is_cointegrated = coint_results["cointegrated"].sum() >= 3
    print(f"\n    Conclusion: {'COINTEGRATED' if is_cointegrated else 'NOT COINTEGRATED'}")

    # 4. Build Spread
    print("\n[4] Spread Analysis...")
    ols_model = get_OLS(y_train, x_train)
    beta = ols_model["USDCAD_weight"].iloc[0]
    alpha = ols_model["intercept"].iloc[0]

    spread = y_log - (alpha + beta * x_log)
    spread = spread.rename("spread")

    hl, phi = measure_mean_reversion(spread[train_mask])
    hl_ou, kappa = halflife_ou(spread[train_mask])

    print(f"    Hedge Ratio (beta): {beta:.6f}")
    print(f"    Intercept (alpha): {alpha:.6f}")
    print(f"    AR(1) Half-life: {hl:.1f} bars ({hl * 15 / 60:.1f} hours)")
    print(f"    OU Half-life: {hl_ou:.1f} bars ({hl_ou * 15 / 60:.1f} hours)")
    print(f"    Mean reversion strength (phi): {phi:.4f}")

    # 5. Generate Signals
    print("\n[5] Signal Generation...")
    test_mask = (y_log.index >= Config.TEST_START) & (y_log.index <= Config.TEST_END)

    zscore = calc_rolling_zscore(spread, Config.ZSCORE_WINDOW)
    signals = zscore_signal_threshold(
        zscore[test_mask],
        enter_long=-Config.ENTRY_THRESHOLD,
        exit_long=Config.EXIT_THRESHOLD,
        enter_short=Config.ENTRY_THRESHOLD,
        exit_short=-Config.EXIT_THRESHOLD
    )

    n_trades = (signals["signal"].diff().abs() > 0).sum()
    print(f"    Test period signals: {n_trades} trades")

    return {
        "asset_class": "Forex",
        "pair": "AUDUSD/USDCAD",
        "is_cointegrated": is_cointegrated,
        "half_life": hl,
        "phi": phi,
        "n_trades": n_trades,
        "spread": spread,
        "zscore": zscore
    }


# ============================================================================
# EQUITIES PAIRS ANALYSIS
# ============================================================================

def analyze_equities_pair():
    """
    Analyze cointegration in equity pairs.
    Uses Yahoo Finance (yfinance) for data retrieval.

    Primary pair: PODD (Insulet Corp) / RMD (ResMed Inc)
    Sector: Healthcare Equipment & Supplies
    """
    print("\n" + "=" * 70)
    print("EQUITIES COINTEGRATION ANALYSIS")
    print("Pair: PODD (Insulet Corp) / RMD (ResMed Inc)")
    print("Sector: Healthcare Equipment & Supplies")
    print("=" * 70)

    from src.data_providers import EquitiesDataProvider
    from src.cointegration import (
        get_integration, pair_cointegration_tests, vector_cointegration_test,
        get_OLS, get_VECM, get_kalman_filter_model, measure_mean_reversion
    )
    from src.signals import calc_rolling_zscore, zscore_signal_linear
    from src.backtest import (
        backtest, set_rates, get_backtest_metrics, normalize_model_weights
    )

    # 1. Data Retrieval
    print("\n[1] Fetching equity data...")
    tickers = ["PODD", "RMD"]

    try:
        import yfinance as yf
        data = yf.download(tickers, start="2015-01-01", end="2025-11-01", auto_adjust=True)
        df_prices = data["Close"].dropna()
        df_log_prices = np.log(df_prices)
        print(f"    {tickers[0]} (Insulet Corp): {len(df_prices):,} days")
        print(f"    {tickers[1]} (ResMed Inc): {len(df_prices):,} days")
    except Exception as e:
        print(f"    [!] yfinance unavailable: {e}")
        print("    Using simulated data for demonstration...")
        np.random.seed(123)
        dates = pd.bdate_range("2015-01-01", "2025-11-01")
        n = len(dates)
        common = np.cumsum(np.random.randn(n) * 0.02)
        df_prices = pd.DataFrame({
            "PODD": 50 * np.exp(common + np.cumsum(np.random.randn(n) * 0.01)),
            "RMD": 80 * np.exp(0.8 * common + np.cumsum(np.random.randn(n) * 0.01))
        }, index=dates)
        df_log_prices = np.log(df_prices)

    df_log_prices.columns = [t + "_logprice" for t in tickers]
    df_prices.columns = [t + "_price" for t in tickers]

    # Split train/test
    train_mask = (df_prices.index >= Config.TRAIN_START) & (df_prices.index <= Config.TRAIN_END)
    test_mask = (df_prices.index >= Config.TEST_START) & (df_prices.index <= Config.TEST_END)

    df_log_train = df_log_prices[train_mask]
    df_log_test = df_log_prices[test_mask]
    df_prices_test = df_prices[test_mask]

    # 2. Integration Tests
    print("\n[2] Order of Integration Tests...")
    for ticker in tickers:
        col = ticker + "_logprice"
        i_order = get_integration(df_log_train[col])
        print(f"    {ticker}: I({i_order})")

    # 3. Cointegration Tests
    print("\n[3] Cointegration Tests...")
    coint_results = pair_cointegration_tests(
        df_log_train[tickers[0] + "_logprice"],
        df_log_train[tickers[1] + "_logprice"]
    )
    print(coint_results.to_string(index=False))

    is_cointegrated = coint_results["cointegrated"].sum() >= 3
    print(f"\n    Conclusion: {'COINTEGRATED' if is_cointegrated else 'NOT COINTEGRATED'}")

    # 4. VECM Analysis
    print("\n[4] VECM Model...")
    lag, rank = vector_cointegration_test(df_log_train)
    print(f"    Optimal lag order: {lag}")
    print(f"    Cointegration rank: {rank}")

    if rank > 0:
        try:
            vecm_models = get_VECM(df_log_train, lag, 1)
            vecm_model = vecm_models[0]

            weights = vecm_model.iloc[0][[c for c in vecm_model.columns if "_weight" in c]]
            intercept = vecm_model.iloc[0]["intercept"]

            print(f"    VECM weights: {dict(weights.round(4))}")
            print(f"    VECM intercept: {intercept:.4f}")
        except Exception as e:
            print(f"    [!] VECM failed: {e}")
            vecm_model = None
    else:
        vecm_model = None

    # 5. OLS Analysis (fallback)
    print("\n[5] OLS Spread Analysis...")
    ols_model = get_OLS(
        df_log_train[tickers[0] + "_logprice"],
        df_log_train[tickers[1] + "_logprice"]
    )

    beta = ols_model[tickers[1] + "_weight"].iloc[0]
    alpha = ols_model["intercept"].iloc[0]

    hl, phi = measure_mean_reversion(ols_model["spread"])
    print(f"    OLS Hedge Ratio: {beta:.6f}")
    print(f"    Half-life: {hl:.1f} days")
    print(f"    Mean reversion (phi): {phi:.4f}")

    # 6. Kalman Filter (if VECM available)
    if vecm_model is not None:
        print("\n[6] Kalman Filter Dynamic Hedge...")
        try:
            kalman_model = get_kalman_filter_model(
                df_log_test,
                tickers[0],
                weights,
                intercept,
                kf_P=1e-5,
                kf_Q_beta=1e-3,
                kf_Q_intercept=1e-6,
                kf_R=1
            )
            kalman_model = normalize_model_weights(kalman_model)
            spread = kalman_model["spread"]
            print(f"    Kalman spread std: {spread.std():.4f}")
        except Exception as e:
            print(f"    [!] Kalman filter failed: {e}")
            spread = ols_model["spread"]
    else:
        spread = ols_model["spread"]

    # 7. Backtesting
    print("\n[7] Backtest Results...")

    # Extend OLS to test period
    full_spread = (
        df_log_prices[tickers[0] + "_logprice"]
        - (alpha + beta * df_log_prices[tickers[1] + "_logprice"])
    )

    zscore = calc_rolling_zscore(full_spread, Config.ZSCORE_WINDOW)
    signals = zscore_signal_linear(
        zscore[test_mask],
        threshold_long=-3,
        threshold_short=3
    )

    # Setup backtest
    weights_df = pd.DataFrame({
        tickers[0] + "_weight": 1.0,
        tickers[1] + "_weight": beta
    }, index=df_prices_test.index)

    rates = set_rates(
        index=signals.index,
        rfr=Config.RISK_FREE_RATE / Config.PERIODS_PER_YEAR,
        short_fee=-0.0025 / Config.PERIODS_PER_YEAR,
        transaction_costs=-Config.TRANSACTION_COSTS,
    )

    try:
        btst = backtest(
            df_prices_test,
            weights_df,
            signals["signal"],
            rates
        )

        metrics = get_backtest_metrics(btst, rates["rfr"], Config.PERIODS_PER_YEAR)
        print("\n    Performance Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")
    except Exception as e:
        print(f"    [!] Backtest failed: {e}")
        metrics = {}

    return {
        "asset_class": "Equities",
        "pair": f"{tickers[0]}/{tickers[1]}",
        "is_cointegrated": is_cointegrated,
        "half_life": hl,
        "phi": phi,
        "vecm_rank": rank if rank else 0,
        "metrics": metrics
    }


# ============================================================================
# PORTFOLIO SUMMARY
# ============================================================================

def generate_portfolio_summary(forex_results, equities_results):
    """Generate combined portfolio analysis summary."""
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)

    print("\n[Asset Allocation]")
    print("-" * 50)
    print(f"{'Asset Class':<15} {'Pair':<20} {'Cointegrated':<15} {'Half-Life':<10}")
    print("-" * 50)

    for r in [forex_results, equities_results]:
        hl_str = f"{r['half_life']:.1f}" if r['half_life'] < 1000 else "N/A"
        coint_str = "Yes" if r['is_cointegrated'] else "No"
        print(f"{r['asset_class']:<15} {r['pair']:<20} {coint_str:<15} {hl_str:<10}")

    print("\n[Key Insights]")
    print("-" * 50)
    print("1. Forex pairs typically show faster mean reversion (intraday)")
    print("2. Equity pairs require longer holding periods (days to weeks)")
    print("3. Kalman filter provides adaptive hedge ratios for regime changes")
    print("4. Z-score thresholds should be calibrated to half-life")

    print("\n[Risk Considerations]")
    print("-" * 50)
    print("- Cointegration can break down during regime changes")
    print("- Transaction costs significantly impact short-term strategies")
    print("- Position sizing should account for spread volatility")
    print("- Rolling stability tests recommended before live trading")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete analysis pipeline."""
    print("\n" + "#" * 70)
    print("# STATISTICAL ARBITRAGE PORTFOLIO ANALYSIS")
    print("# Cointegration-Based Pairs Trading Framework")
    print(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("#" * 70)

    # Analyze both asset classes
    forex_results = analyze_forex_pair()
    equities_results = analyze_equities_pair()

    # Generate summary
    generate_portfolio_summary(forex_results, equities_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    return forex_results, equities_results


if __name__ == "__main__":
    results = main()
