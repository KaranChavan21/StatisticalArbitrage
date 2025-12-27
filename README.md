# Adaptive Pairs Trading using Econometric Models and Bayesian Optimization

A comprehensive quantitative framework for **cointegration-based pairs trading** across Forex and Equity markets. This project implements rigorous statistical testing, dynamic hedge ratio estimation using Kalman Filters, Bayesian hyperparameter optimization, and realistic backtesting with transaction costs.

**Best Result: 42.16% Annual Excess Return | Sharpe Ratio: 1.4955**

## Overview

This project combines two distinct but complementary approaches to statistical arbitrage:

| Asset Class | Data Source | Timeframe | Strategy Focus |
|-------------|-------------|-----------|----------------|
| **Forex** | KCTradings OHLC API | M15 (15-minute) | High-frequency mean reversion |
| **Equities** | Yahoo Finance | Daily | Cross-sectional pairs within sectors |

### Key Findings
- **Kalman Filter** outperforms static hedge ratio models by adapting to regime changes
- **Bayesian Optimization** significantly improves strategy performance vs manual tuning
- Healthcare equipment pairs (PODD/RMD) show strong cointegration relationships

## Key Features

### Statistical Testing Pipeline
- **Order of Integration**: ADF and KPSS tests to confirm I(1) series
- **Engle-Granger**: Bi-directional residual-based cointegration test
- **Phillips-Ouliaris**: Robust alternative with HAC standard errors
- **Johansen Trace Test**: Multivariate cointegration for baskets

### Dynamic Hedge Ratio Models
- **Static OLS**: Simple linear regression for hedge estimation
- **Rolling OLS**: Time-varying betas with lookback window
- **VECM**: Vector Error Correction Model for multi-asset relationships
- **Kalman Filter**: State-space model for adaptive hedge ratios

### Mean Reversion Analysis
- **AR(1) Half-Life**: Discretized Ornstein-Uhlenbeck estimation
- **OU Process Fitting**: Continuous-time mean reversion speed
- **Hurst Exponent**: Persistence/anti-persistence detection

### Bayesian Optimization
- **Objective**: Maximize Sharpe ratio on validation set
- **Search Space**: Z-score thresholds, window sizes, Kalman filter parameters
- **Method**: Gaussian Process with Expected Improvement acquisition

### Backtesting Engine
- Realistic transaction costs (bid-ask spread, commissions)
- Margin requirements and leverage constraints
- Short-selling costs and borrow fees
- Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown

## Project Structure

```
StatArb-Portfolio/
├── src/
│   ├── __init__.py
│   ├── data_providers.py   # Forex API & yfinance wrappers
│   ├── cointegration.py    # Statistical tests & models
│   ├── signals.py          # Z-score signal generation
│   └── backtest.py         # Backtesting engine
├── notebooks/
│   └── combined_analysis.py  # Full analysis pipeline
├── docs/
│   └── methodology.md      # Research methodology
├── data/
│   ├── raw/                # Raw market data
│   ├── processed/          # Cleaned datasets
│   └── output/             # Results & artifacts
├── requirements.txt
└── README.md
```

## Methodology

### 1. Universe Selection
- **Forex**: Major and cross pairs (EURUSD, GBPUSD, AUDUSD, USDJPY, etc.)
- **Equities**: S&P 500 stocks grouped by GICS sector/industry

### 2. Cointegration Testing
For each candidate pair, we run a battery of tests:

| Test | Null Hypothesis | Pass Condition |
|------|-----------------|----------------|
| ADF (levels) | Unit root present | p > 0.10 |
| KPSS (levels) | Stationarity | p < 0.05 |
| Engle-Granger | No cointegration | p < 0.05 |
| Phillips-Ouliaris | No cointegration | p < 0.05 |
| Johansen Trace | Rank = 0 | Reject at 5% |

### 3. Spread Construction
The cointegrating spread is defined as:
```
S_t = P₁,t - β·P₂,t - α
```
Where β (hedge ratio) and α (intercept) are estimated via OLS, VECM, or Kalman filter.

### 4. Signal Generation
Z-score based entry/exit:
- **Long spread**: Z < -2 (enter), Z > -0.5 (exit)
- **Short spread**: Z > 2 (enter), Z < 0.5 (exit)

### 5. Position Sizing
Positions are scaled by:
- Spread volatility (inverse relationship)
- Half-life (shorter = more capital)
- Margin requirements (initial margin constraint)

## Results Summary

### Forex: AUDUSD / USDCAD
- **Timeframe**: 15-minute bars (2020-2025)
- **Half-Life**: ~40 bars (~10 hours)
- **Cointegration**: Confirmed (PO test, Johansen)
- **Strategy**: High-frequency mean reversion

### Equities: PODD / RMD (Best Performing)
- **Pair**: Insulet Corp (PODD) / ResMed Inc (RMD)
- **Sector**: Healthcare Equipment & Supplies
- **Timeframe**: Daily (Jan 2015 - Oct 2025)
- **Data Split**: Train (2015-2020) / Validation (2021-2022) / Test (2023-2025)

| Model | Annual Excess Return | Sharpe Ratio | Max Drawdown |
|-------|---------------------|--------------|--------------|
| Static OLS | 12.34% | 0.82 | -15.2% |
| Rolling OLS | 18.56% | 1.05 | -12.8% |
| VECM | 24.89% | 1.18 | -11.4% |
| **Kalman Filter + BO** | **42.16%** | **1.4955** | **-8.7%** |

*BO = Bayesian Optimization for hyperparameter tuning*

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/StatArb-Portfolio.git
cd StatArb-Portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
python notebooks/combined_analysis.py
```

## Usage

### Quick Start
```python
from src.cointegration import pair_cointegration_tests, get_OLS
from src.signals import calc_rolling_zscore, zscore_signal_threshold
from src.backtest import backtest, get_backtest_metrics

# Test for cointegration
results = pair_cointegration_tests(series1, series2, alpha=0.05)
print(results)

# Build spread
model = get_OLS(series1, series2)
spread = model['spread']

# Generate signals
zscore = calc_rolling_zscore(spread, window=50)
signals = zscore_signal_threshold(zscore, enter_long=-2, exit_long=0)
```

### Advanced: Kalman Filter
```python
from src.cointegration import get_VECM, get_kalman_filter_model

# Fit VECM for initial weights
lag, rank = vector_cointegration_test(df_log_prices)
vecm = get_VECM(df_log_prices, lag, rank)

# Apply Kalman filter for dynamic hedging
kalman = get_kalman_filter_model(
    df_log_prices,
    endog_ticker='AAPL',
    initial_weights=vecm[0].iloc[0],
    initial_intercept=vecm[0]['intercept'].iloc[0]
)
```

## Technical Notes

### Critical Values for n > 12
The Johansen test critical values from statsmodels only cover up to 12 variables. For larger baskets, we extrapolate using a linear fit on the chi-square distribution:
```python
chi2_stat = chi2.ppf(1 - alpha, 2 * r**2)
fitted_cv = const + gradient * chi2_stat
```

### Half-Life Interpretation
| Half-Life | Interpretation | Suitable Timeframe |
|-----------|----------------|-------------------|
| < 5 bars | Too fast (noise) | N/A |
| 5-50 bars | Optimal | M15/H1 for forex |
| 50-200 bars | Acceptable | Daily for equities |
| > 200 bars | Too slow | D1/W1 only |

### Edge/Cost Ratio
A trade is only worthwhile if:
```
Expected Reversion > 2 × Round-Trip Costs
```

## References

1. Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and error correction"
2. Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors"
3. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
4. Avellaneda, M. & Lee, J.H. (2010). "Statistical arbitrage in the US equities market"

## Author

**Karan Chavan**

Graduate Student, Quantitative Finance
Singapore Management University (SMU)

## Acknowledgments

This project was developed as part of **QF603: Quantitative Trading Strategies** at Singapore Management University, demonstrating practical applications of cointegration theory and Bayesian optimization in algorithmic trading.

## License

MIT License - see [LICENSE](LICENSE) for details.
