"""
Bayesian Optimization Module
Hyperparameter tuning for pairs trading strategies using Gaussian Process optimization.

Key Parameters Optimized:
- Z-score entry/exit thresholds
- Rolling window sizes
- Kalman filter noise parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Optional import for Bayesian Optimization
try:
    from bayes_opt import BayesianOptimization
    HAS_BAYESOPT = True
except ImportError:
    HAS_BAYESOPT = False
    print("Warning: bayesian-optimization not installed. Run: pip install bayesian-optimization")


def optimize_strategy_params(
    objective_func: Callable,
    param_bounds: Dict[str, Tuple[float, float]],
    n_iter: int = 50,
    init_points: int = 10,
    random_state: int = 42
) -> Dict:
    """
    Bayesian Optimization for strategy hyperparameters.

    Parameters
    ----------
    objective_func : Callable
        Function that returns Sharpe ratio (or other metric to maximize)
        Should accept parameters as keyword arguments
    param_bounds : Dict
        Parameter bounds, e.g., {'entry_z': (1.5, 3.0), 'exit_z': (0.0, 1.0)}
    n_iter : int
        Number of optimization iterations
    init_points : int
        Number of random initial samples
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    Dict
        Best parameters and optimization results
    """
    if not HAS_BAYESOPT:
        raise ImportError("bayesian-optimization required. Install with: pip install bayesian-optimization")

    optimizer = BayesianOptimization(
        f=objective_func,
        pbounds=param_bounds,
        random_state=random_state,
        verbose=2
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter
    )

    return {
        "best_params": optimizer.max["params"],
        "best_value": optimizer.max["target"],
        "all_results": pd.DataFrame(optimizer.res)
    }


def create_sharpe_objective(
    spread: pd.Series,
    prices_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    backtest_func: Callable,
    signals_func: Callable,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> Callable:
    """
    Create objective function for Sharpe ratio maximization.

    Parameters
    ----------
    spread : pd.Series
        The cointegrating spread series
    prices_df : pd.DataFrame
        Price data for backtesting
    weights_df : pd.DataFrame
        Hedge ratio weights
    backtest_func : Callable
        Backtesting function
    signals_func : Callable
        Signal generation function
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Trading periods per year

    Returns
    -------
    Callable
        Objective function for optimizer
    """
    def objective(entry_z: float, exit_z: float, window: float) -> float:
        try:
            window = int(window)

            # Calculate z-score with given window
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            zscore = (spread - rolling_mean) / rolling_std

            # Generate signals
            signals = signals_func(
                zscore,
                enter_long=-entry_z,
                exit_long=exit_z,
                enter_short=entry_z,
                exit_short=-exit_z
            )

            # Run backtest
            results = backtest_func(prices_df, weights_df, signals["signal"])

            # Calculate Sharpe ratio
            returns = results["portfolio_return"]
            excess_returns = returns - (risk_free_rate / periods_per_year)
            sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

            if np.isnan(sharpe) or np.isinf(sharpe):
                return -10.0

            return sharpe

        except Exception:
            return -10.0

    return objective


def create_kalman_objective(
    df_log_prices: pd.DataFrame,
    endog_ticker: str,
    initial_weights: pd.Series,
    initial_intercept: float,
    prices_df: pd.DataFrame,
    backtest_func: Callable,
    signals_func: Callable,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> Callable:
    """
    Create objective function for Kalman filter parameter optimization.

    Optimizes: Q_beta, Q_intercept, R, P, entry_z, exit_z
    """
    from src.cointegration import get_kalman_filter_model
    from src.backtest import normalize_model_weights

    def objective(
        log_Q_beta: float,
        log_Q_intercept: float,
        log_R: float,
        entry_z: float,
        exit_z: float,
        window: float
    ) -> float:
        try:
            window = int(window)

            # Convert log parameters back to original scale
            Q_beta = 10 ** log_Q_beta
            Q_intercept = 10 ** log_Q_intercept
            R = 10 ** log_R

            # Fit Kalman filter
            kalman_model = get_kalman_filter_model(
                df_log_prices,
                endog_ticker,
                initial_weights,
                initial_intercept,
                kf_Q_beta=Q_beta,
                kf_Q_intercept=Q_intercept,
                kf_R=R,
                kf_P=1e-2
            )

            kalman_model = normalize_model_weights(kalman_model)
            spread = kalman_model["spread"]

            # Calculate z-score
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            zscore = (spread - rolling_mean) / rolling_std

            # Generate signals
            signals = signals_func(
                zscore.dropna(),
                enter_long=-entry_z,
                exit_long=exit_z,
                enter_short=entry_z,
                exit_short=-exit_z
            )

            # Get weights from Kalman model
            weight_cols = [c for c in kalman_model.columns if "_weight" in c]
            weights_df = kalman_model[weight_cols].loc[signals.index]

            # Run backtest
            results = backtest_func(
                prices_df.loc[signals.index],
                weights_df,
                signals["signal"]
            )

            # Calculate Sharpe ratio
            returns = results["portfolio_return"]
            excess_returns = returns - (risk_free_rate / periods_per_year)
            sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

            if np.isnan(sharpe) or np.isinf(sharpe):
                return -10.0

            return sharpe

        except Exception:
            return -10.0

    return objective


def get_default_param_bounds() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Default parameter bounds for optimization.

    Returns
    -------
    Dict
        Parameter bounds for different optimization targets
    """
    return {
        "zscore_params": {
            "entry_z": (1.5, 3.5),
            "exit_z": (0.0, 1.0),
            "window": (20, 100)
        },
        "kalman_params": {
            "log_Q_beta": (-6, -2),
            "log_Q_intercept": (-9, -5),
            "log_R": (-3, 1),
            "entry_z": (1.5, 3.5),
            "exit_z": (0.0, 1.0),
            "window": (20, 100)
        }
    }
