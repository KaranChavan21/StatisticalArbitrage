# Research Methodology

## Cointegration & Statistical Arbitrage Framework

### 1. Theoretical Foundation

#### 1.1 Cointegration Definition
Two or more non-stationary I(1) time series are **cointegrated** if there exists a linear combination that is stationary I(0). For price series $P_1$ and $P_2$:

$$S_t = P_{1,t} - \beta P_{2,t} - \alpha$$

If $S_t \sim I(0)$ while $P_1, P_2 \sim I(1)$, the series are cointegrated with cointegrating vector $(1, -\beta)$.

#### 1.2 Economic Interpretation
- **Equity Pairs**: Companies in the same sector face similar market risks
- **Forex Triangles**: Arbitrage-free pricing implies cointegration
- **Commodities**: Substitutes or complements share common factors

### 2. Testing Pipeline

#### Stage A: Order of Integration
Before testing cointegration, confirm both series are I(1):

| Test | Null Hypothesis | Statistic |
|------|-----------------|-----------|
| ADF | Unit root (non-stationary) | Dickey-Fuller t-stat |
| KPSS | Stationarity | LM statistic |

**Decision Rule**:
- Levels: ADF fails to reject (p > 0.10), KPSS rejects (p < 0.05)
- First differences: ADF rejects (p < 0.05)

#### Stage B: Cointegration Tests

##### Engle-Granger Two-Step
1. Regress $Y$ on $X$: $Y_t = \alpha + \beta X_t + \epsilon_t$
2. Test residuals $\hat{\epsilon}_t$ for stationarity using ADF

**Critical Values**: MacKinnon (1991) simulation-based values, accounting for OLS estimation error.

##### Phillips-Ouliaris
Similar to EG but uses HAC (Newey-West) standard errors to handle:
- Serial correlation in residuals
- Heteroskedasticity

**Advantage**: Direction-invariant (unlike EG)

##### Johansen Trace Test
For n-dimensional system, tests for rank r of cointegrating matrix:
- $H_0$: rank = r vs $H_1$: rank > r
- Sequential testing from r=0 upward

**Output**: Cointegrating vectors (eigenvectors of companion matrix)

### 3. Spread Modeling

#### 3.1 Static OLS
$$\hat{\beta} = \frac{Cov(P_1, P_2)}{Var(P_2)}$$

**Pros**: Simple, interpretable
**Cons**: Assumes constant relationship

#### 3.2 Rolling OLS
Re-estimate $\beta$ over trailing window of $w$ observations:
$$\hat{\beta}_t = f(P_{1,t-w:t}, P_{2,t-w:t})$$

**Pros**: Adapts to regime changes
**Cons**: Lookback lag, window size sensitivity

#### 3.3 VECM (Vector Error Correction Model)
For n assets, the VECM representation:
$$\Delta P_t = \alpha \beta' P_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta P_{t-i} + \epsilon_t$$

Where:
- $\beta$: Cointegrating vectors (long-run relationships)
- $\alpha$: Loading matrix (adjustment speeds)
- $\Gamma_i$: Short-run dynamics

#### 3.4 Kalman Filter
State-space formulation for time-varying $\beta_t$:

**State equation**:
$$\beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$$

**Observation equation**:
$$P_{1,t} = \beta_t' X_t + \epsilon_t, \quad \epsilon_t \sim N(0, R)$$

**Kalman recursion**:
1. Predict: $\hat{\beta}_{t|t-1} = \hat{\beta}_{t-1|t-1}$
2. Update: $\hat{\beta}_{t|t} = \hat{\beta}_{t|t-1} + K_t (P_{1,t} - \hat{P}_{1,t|t-1})$

Where $K_t$ is the Kalman gain balancing prediction vs observation.

### 4. Mean Reversion Analysis

#### 4.1 Ornstein-Uhlenbeck Process
The spread follows:
$$dS_t = \kappa(\mu - S_t)dt + \sigma dW_t$$

Where:
- $\kappa$: Mean reversion speed
- $\mu$: Long-run mean
- $\sigma$: Volatility

#### 4.2 Half-Life Estimation
Discretized AR(1):
$$S_t = c + \phi S_{t-1} + \epsilon_t$$

Half-life (in periods):
$$HL = \frac{\ln(0.5)}{\ln(|\phi|)}$$

#### 4.3 Hurst Exponent
For time series $X$:
$$H = \frac{\log(R/S)}{\log(T)}$$

Where R/S is the rescaled range.

| H Value | Interpretation |
|---------|----------------|
| H < 0.5 | Mean-reverting |
| H = 0.5 | Random walk |
| H > 0.5 | Trending |

### 5. Signal Generation

#### 5.1 Z-Score Calculation
$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

Using rolling mean and standard deviation over window $w$:
- $\mu_t = \frac{1}{w}\sum_{i=0}^{w-1} S_{t-i}$
- $\sigma_t = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1} (S_{t-i} - \mu_t)^2}$

#### 5.2 Threshold Strategy
```
IF Z < -entry_threshold: LONG spread (buy leg1, sell leg2)
IF Z > entry_threshold: SHORT spread (sell leg1, buy leg2)
IF |Z| < exit_threshold: FLATTEN
```

#### 5.3 Linear Scaling
Position size proportional to Z magnitude:
$$Position = -\text{sign}(Z) \times \min\left(\frac{|Z|}{Z_{max}}, 1\right)$$

### 6. Risk Management

#### 6.1 Position Sizing
Target volatility approach:
$$Position_{size} = \frac{\text{Target Vol}}{\sigma_{spread} \times \sqrt{HL}}$$

#### 6.2 Stop Losses
- **Time stop**: Exit if no convergence after 2×HL
- **Z-score stop**: Exit if Z exceeds 4 (likely regime break)
- **Loss stop**: Maximum drawdown limit per trade

#### 6.3 Cointegration Monitoring
Rolling p-value tracking:
- Recalculate EG/PO every N bars
- If p > 0.10 for M consecutive windows, suspend trading

### 7. Backtesting Considerations

#### 7.1 Transaction Costs
| Cost Component | Typical Value |
|----------------|---------------|
| Bid-ask spread (FX major) | 1-2 pips |
| Bid-ask spread (equity) | 5-20 bps |
| Commission | 1-5 bps |
| Slippage | 2-10 bps |
| Short borrow (equities) | 25-100 bps/year |

#### 7.2 Realistic Assumptions
- No look-ahead bias (train/test split)
- Execution delay (1-bar minimum)
- Position limits and leverage constraints
- Weekend/holiday handling for FX

#### 7.3 Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe | $\frac{E[R-R_f]}{\sigma_R}$ | > 1.5 |
| Sortino | $\frac{E[R-R_f]}{\sigma_{down}}$ | > 2.0 |
| Calmar | $\frac{E[R_{ann}]}{MDD}$ | > 1.0 |
| Win Rate | $\frac{\text{Winning trades}}{\text{Total trades}}$ | > 50% |

### 8. Extensions

#### 8.1 Multi-Asset Baskets
Johansen allows r < n cointegrating relationships:
- Each eigenvector defines a separate spread
- Trade multiple spreads as a portfolio

#### 8.2 Regime Detection
Hidden Markov Models for:
- High/low volatility regimes
- Trending vs mean-reverting markets

#### 8.3 Machine Learning Integration
- Feature engineering from cointegration stats
- Gradient boosting for pair selection
- Reinforcement learning for dynamic thresholds

---

## References

1. Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and error correction: Representation, estimation, and testing." *Econometrica*, 55(2), 251-276.

2. Johansen, S. (1988). "Statistical analysis of cointegration vectors." *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.

3. Phillips, P.C.B. & Ouliaris, S. (1990). "Asymptotic properties of residual based tests for cointegration." *Econometrica*, 58(1), 165-193.

4. Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.

5. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.

6. Avellaneda, M. & Lee, J.H. (2010). "Statistical arbitrage in the US equities market." *Quantitative Finance*, 10(7), 761-782.

7. Gatev, E., Goetzmann, W.N., & Rouwenhorst, K.G. (2006). "Pairs trading: Performance of a relative-value arbitrage rule." *Review of Financial Studies*, 19(3), 797-827.
