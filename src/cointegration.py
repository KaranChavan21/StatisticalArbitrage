"""
Cointegration Analysis Module
Statistical tests and modeling for cointegrated pairs and baskets.

Methods Implemented:
- Engle-Granger Two-Step Test
- Phillips-Ouliaris Test
- Johansen Trace Test
- Vector Error Correction Model (VECM)
- Kalman Filter for Dynamic Hedge Ratios
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from scipy.stats import chi2
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports with graceful fallback
try:
    from arch.unitroot.cointegration import engle_granger, phillips_ouliaris
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: arch package not installed. Some cointegration tests unavailable.")

try:
    from filterpy.kalman import KalmanFilter as FilterPyKalman
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("Warning: filterpy package not installed. Kalman filter unavailable.")


# ============================================================================
# ORDER OF INTEGRATION TESTS
# ============================================================================

def get_integration(series: pd.Series, alpha: float = 0.05) -> int:
    """
    Determine order of integration (0 <= d <= 2) using ADF and KPSS tests.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    alpha : float
        Significance level

    Returns
    -------
    int
        Order of integration (0, 1, 2, or NaN if nonstationary after twice-differencing)
    """
    for d in range(3):
        test_series = series.copy().dropna()
        for _ in range(d):
            test_series = test_series.diff().dropna()

        adf_p = tsa.adfuller(test_series, regression="c", autolag="AIC")[1]
        kpss_p = tsa.kpss(test_series, regression="c")[1]

        if (adf_p < alpha) and (kpss_p > alpha):
            return d

    return np.nan


def integration_summary(series: pd.Series, name: str = "Series") -> pd.DataFrame:
    """
    Comprehensive integration order summary with ADF and KPSS results.
    """
    results = []

    for level, suffix in [("level", ""), ("diff1", "_d1")]:
        test_series = series.dropna()
        if level == "diff1":
            test_series = test_series.diff().dropna()

        adf_stat, adf_p, *_ = tsa.adfuller(test_series, autolag="AIC")
        kpss_stat, kpss_p, *_ = tsa.kpss(test_series, regression="c", nlags="auto")

        results.append({
            "series": f"{name} ({level})",
            "ADF_stat": round(adf_stat, 4),
            "ADF_p": round(adf_p, 4),
            "ADF_decision": "Stationary" if adf_p < 0.05 else "Non-stationary",
            "KPSS_stat": round(kpss_stat, 4),
            "KPSS_p": round(kpss_p, 4),
            "KPSS_decision": "Non-stationary" if kpss_p < 0.05 else "Stationary",
        })

    return pd.DataFrame(results)


# ============================================================================
# PAIR COINTEGRATION TESTS
# ============================================================================

def pair_cointegration_tests(
    p1: pd.Series,
    p2: pd.Series,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Comprehensive cointegration tests for a pair of series.

    Tests performed:
    1. Engle-Granger (both directions)
    2. Phillips-Ouliaris
    3. Johansen Trace Test

    Parameters
    ----------
    p1, p2 : pd.Series
        Log price series
    alpha : float
        Significance level (0.01, 0.05, or 0.10)

    Returns
    -------
    pd.DataFrame
        Test results with statistics, critical values, and pass/fail
    """
    if alpha not in (0.01, 0.05, 0.10):
        raise ValueError(f"alpha must be 0.01, 0.05, or 0.10 (was {alpha})")

    results = []

    # Engle-Granger and Phillips-Ouliaris Tests (require arch package)
    if HAS_ARCH:
        eg1 = engle_granger(p1, p2, trend="c", method="aic")
        eg2 = engle_granger(p2, p1, trend="c", method="aic")
        po = phillips_ouliaris(p1, p2, trend="c", test_type="Za")

        results.extend([
            {
                "test": "Engle-Granger (Y~X)",
                "stat": eg1.stat,
                "cv90": eg1.critical_values.iloc[0],
                "cv95": eg1.critical_values.iloc[1],
                "cv99": eg1.critical_values.iloc[2],
                "p_value": eg1.pvalue,
                "cointegrated": eg1.pvalue <= alpha
            },
            {
                "test": "Engle-Granger (X~Y)",
                "stat": eg2.stat,
                "cv90": eg2.critical_values.iloc[0],
                "cv95": eg2.critical_values.iloc[1],
                "cv99": eg2.critical_values.iloc[2],
                "p_value": eg2.pvalue,
                "cointegrated": eg2.pvalue <= alpha
            },
            {
                "test": "Phillips-Ouliaris",
                "stat": po.stat,
                "cv90": po.critical_values.iloc[0],
                "cv95": po.critical_values.iloc[1],
                "cv99": po.critical_values.iloc[2],
                "p_value": po.pvalue,
                "cointegrated": po.pvalue <= alpha
            }
        ])

    # Johansen Trace Test (always available via statsmodels)
    prices = pd.concat([p1, p2], axis="columns")
    var_model = tsa.VAR(prices)
    lag_order = var_model.select_order().selected_orders["bic"]
    johansen = coint_johansen(prices, det_order=0, k_ar_diff=lag_order)
    cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}[alpha]

    results.extend([
        {
            "test": "Johansen (r=0)",
            "stat": johansen.lr1[0],
            "cv90": johansen.cvt[0, 0],
            "cv95": johansen.cvt[0, 1],
            "cv99": johansen.cvt[0, 2],
            "p_value": np.nan,
            "cointegrated": johansen.lr1[0] > johansen.cvt[0, cv_idx]
        },
        {
            "test": "Johansen (r<=1)",
            "stat": johansen.lr1[1],
            "cv90": johansen.cvt[1, 0],
            "cv95": johansen.cvt[1, 1],
            "cv99": johansen.cvt[1, 2],
            "p_value": np.nan,
            "cointegrated": johansen.lr1[1] < johansen.cvt[1, cv_idx]
        }
    ])

    return pd.DataFrame(results)


def vector_cointegration_test(
    df_prices: pd.DataFrame,
    alpha: float = 0.05
) -> Tuple[int, int]:
    """
    Multi-asset cointegration test using Johansen procedure.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Log price series for multiple assets
    alpha : float
        Significance level

    Returns
    -------
    tuple
        (lag_order, cointegration_rank)
    """
    var_model = tsa.VAR(df_prices)
    lag_order = var_model.select_order().selected_orders["bic"]

    johansen = coint_johansen(df_prices, det_order=0, k_ar_diff=lag_order)
    cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}[alpha]

    trace_stat = johansen.lr1
    trace_cv = johansen.cvt[:, cv_idx]

    n = df_prices.shape[1]

    # Extrapolate critical values for n > 12
    if n > 12:
        valid_cv = trace_cv[-12:]
        chi2_stat = np.array([chi2.ppf(1 - alpha, 2 * r**2) for r in range(12, 0, -1)])
        const, gradient = sm.OLS(valid_cv, sm.add_constant(chi2_stat)).fit().params
        fitted_cv = const + gradient * np.array(
            [chi2.ppf(1 - alpha, 2 * r**2) for r in range(n, 12, -1)]
        )
        trace_cv = np.concatenate([fitted_cv, valid_cv])

    trace_test = trace_stat < trace_cv
    rank = n if trace_test.sum() == 0 else trace_test.argmax()

    return lag_order, rank


# ============================================================================
# COINTEGRATION MODELING
# ============================================================================

def get_OLS(p1: pd.Series, p2: pd.Series) -> pd.DataFrame:
    """
    Static OLS estimation of cointegrating relationship.
    Relationship: p1 = alpha + beta * p2 + epsilon
    """
    t1 = p1.name.replace("_logprice", "").replace("_price", "")
    t2 = p2.name.replace("_logprice", "").replace("_price", "")

    p2_const = sm.add_constant(p2)
    ols = sm.OLS(p1, p2_const).fit()

    w2 = pd.Series(-ols.params.at[p2.name], index=p2.index, name=f"{t2}_weight")
    w1 = pd.Series(1.0, index=p1.index, name=f"{t1}_weight")
    intercept = pd.Series(ols.params.at["const"], index=p1.index, name="intercept")
    spread = ols.resid.rename("spread")

    return pd.concat([p1, p2, w1, w2, intercept, spread], axis="columns")


def get_rolling_OLS(
    p1: pd.Series,
    p2: pd.Series,
    window: int = 200
) -> pd.DataFrame:
    """
    Rolling window OLS for time-varying hedge ratio.
    """
    t1 = p1.name.replace("_logprice", "").replace("_price", "")
    t2 = p2.name.replace("_logprice", "").replace("_price", "")

    p2_const = sm.add_constant(p2)
    ols = RollingOLS(p1, p2_const, window=window).fit()

    w2 = -ols.params[p2.name].rename(f"{t2}_weight").shift(1)
    w1 = pd.Series(np.where(w2.isna(), np.nan, 1), index=p1.index, name=f"{t1}_weight")
    intercept = ols.params["const"].rename("intercept").shift(1)
    spread = (p1 * w1 + p2 * w2 - intercept).rename("spread")

    return pd.concat([p1, p2, w1, w2, intercept, spread], axis="columns")


def get_VECM(
    df_log_prices: pd.DataFrame,
    lag_order: int,
    rank: int
) -> List[pd.DataFrame]:
    """
    Fit VECM model for multi-asset cointegration.

    Returns list of DataFrames, one per cointegrating relationship.
    """
    ticker_names = df_log_prices.columns.str.replace("_logprice", "")
    n_assets = df_log_prices.shape[1]

    if rank == 0:
        raise ValueError("VECM rank 0: no cointegrating relationships exist")
    if rank == n_assets:
        raise ValueError("VECM max rank: all series are stationary")

    vecm = VECM(df_log_prices, k_ar_diff=lag_order, coint_rank=rank, deterministic="co").fit()

    weights = vecm.beta
    intercepts = -(np.linalg.inv(vecm.alpha.T @ vecm.alpha) @ (vecm.alpha.T @ vecm.det_coef))
    spreads = df_log_prices @ vecm.beta - intercepts.T

    output = []
    for r in range(rank):
        df_weights = pd.DataFrame(
            1, index=spreads.index,
            columns=[f"{t}_weight" for t in ticker_names]
        ) * weights[:, r]
        intercept = pd.Series(intercepts[r].item(), index=spreads.index, name="intercept")
        spread = spreads[r].rename("spread")
        df_coint = pd.concat([df_log_prices, df_weights, intercept, spread], axis="columns")
        output.append(df_coint)

    return output


def get_kalman_filter_model(
    df_log_prices: pd.DataFrame,
    endog_ticker: str,
    initial_weights: pd.Series,
    initial_intercept: float,
    kf_Q_beta: float = 1e-5,
    kf_Q_intercept: float = 1e-7,
    kf_R: float = 1e-5,
    kf_P: float = 1e-2
) -> pd.DataFrame:
    """
    Kalman Filter for time-varying cointegration hedge ratios.

    Models: P_endog = sum(beta_i * P_exog_i) + intercept + epsilon

    Parameters
    ----------
    df_log_prices : pd.DataFrame
        Log prices for all assets
    endog_ticker : str
        Ticker to use as dependent variable
    initial_weights, initial_intercept : float
        Starting values from static model (e.g., VECM)
    kf_Q_beta, kf_Q_intercept, kf_R, kf_P : float
        Kalman filter parameters
    """
    if not HAS_FILTERPY:
        raise ImportError("filterpy package required for Kalman filter. Install with: pip install filterpy")

    ticker_names = df_log_prices.columns.str.replace("_logprice", "")
    exog_tickers = ticker_names.drop(endog_ticker)

    # Normalize initial weights
    scaling = initial_weights[endog_ticker + "_weight"]
    init_w = initial_weights.loc[initial_weights.index.difference([endog_ticker + "_weight"])]
    init_w["intercept"] = initial_intercept
    init_w = init_w / scaling

    # Initialize Kalman Filter
    dim_x = len(init_w)
    kf = FilterPyKalman(dim_x, 1)
    kf.Q = kf_Q_beta * np.eye(dim_x)
    kf.Q[-1, -1] = kf_Q_intercept
    kf.R = kf_R * np.eye(1)
    kf.P = kf_P * np.eye(dim_x)
    kf.x = init_w.values.reshape(-1, 1)
    kf.F = np.eye(dim_x)

    exog_weights = []

    for row in df_log_prices.index:
        endog = df_log_prices.loc[row, endog_ticker + "_logprice"]
        exog = df_log_prices.loc[row, exog_tickers + "_logprice"]

        kf.predict()
        exog_weights.append(kf.x.flatten())

        kf.H = np.array([[*(-exog.to_numpy()), 1]])
        kf.update(endog)

    df_weights = pd.DataFrame(
        exog_weights,
        index=df_log_prices.index,
        columns=[*(exog_tickers + "_weight"), "intercept"]
    )
    df_weights[endog_ticker + "_weight"] = 1.0
    df_weights = df_weights[[*(ticker_names + "_weight"), "intercept"]]

    spread = (df_weights.values[:, :-1] * df_log_prices.to_numpy()).sum(axis=1) - df_weights["intercept"]

    output = pd.concat([df_log_prices, df_weights], axis="columns")
    output["spread"] = spread

    return output


# ============================================================================
# MEAN REVERSION ANALYSIS
# ============================================================================

def measure_mean_reversion(spread: pd.Series) -> Tuple[float, float]:
    """
    Measure mean reversion using AR(1) model.

    Returns
    -------
    tuple
        (half_life_bars, phi_coefficient)
    """
    ar = tsa.AutoReg(spread.dropna(), lags=1).fit()
    phi = float(ar.params.iloc[1])

    if phi >= 1 or not np.isfinite(phi):
        half_life = np.inf
    else:
        half_life = float(np.log(0.5) / np.log(abs(phi)))

    return half_life, phi


def halflife_ou(spread: pd.Series) -> Tuple[float, float]:
    """
    OU process half-life estimation.
    ds = kappa * (mu - s) dt + sigma dW
    """
    s = spread.dropna()
    if len(s) < 50:
        return np.inf, np.nan

    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna().loc[ds.index]
    X = sm.add_constant(-s_lag.values)
    kappa = sm.OLS(ds.values, X).fit().params[1]

    if kappa <= 0 or not np.isfinite(kappa):
        return np.inf, kappa

    return float(np.log(2) / kappa), float(kappa)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """
    Estimate Hurst exponent for mean-reversion detection.
    H < 0.5: mean-reverting, H > 0.5: trending
    """
    series = series.dropna()
    lags = range(2, min(max_lag, len(series) // 2))
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
