# 📈 ML-Filtered Momentum Trading Strategy

A Quantitative Research Project on Signal Quality Enhancement

## Overview

This project investigates whether machine learning can improve the quality of a classical momentum trading strategy by filtering trades rather than predicting prices directly.
Instead of replacing a well-understood technical signal (EMA crossover), the model learns when not to trade — an approach aligned with modern systematic trading research, where selectivity often matters more than prediction accuracy.
The core research question:

`Can a lightweight ML classifier improve risk-adjusted returns by filtering low-quality momentum signals, while preserving execution realism and fairness in backtesting?`

## Key Ideas

Baseline strategy: EMA(20/50) crossover
ML role: Conditional trade filter (probability of trade success)
Objective: Improve Sharpe ratio, drawdowns, and trade efficiency
Constraint: Same prices, costs, and execution rules across all variants
This ensures that any performance difference is attributable only to the ML filter.

Repository Structure
Quant-Momentum-Project/
│
├── data_ingestion.py        # Market data loading and preprocessing
├── feature_generation.py   # Feature engineering (technical + statistical)
├── label_generator.py      # Forward-return based labeling logic
│
├── train_model.py           # ML training, evaluation, and diagnostics
├── ml_filter.py             # ML-based signal filtering logic
├── backtest_ml_strategy.py # Fair backtest: raw vs ML-filtered strategy
│
├── figure_plotting.py       # Matplotlib & Plotly visualizations
│
└── README.md

Strategy Design
1. Baseline Momentum Signal

The base trading signal is a trend-following EMA crossover:

Long when EMA(20) > EMA(50)
Flat otherwise

This strategy is deliberately simple and interpretable.

## 2. Feature Engineering

All features are causal, shifted to avoid look-ahead bias:

Feature	Description
Volatility (20d)	Rolling standard deviation of returns
RSI (14)	Momentum exhaustion indicator
ATR (14)	Market range / risk proxy
Volume Z-Score	Relative volume anomaly (252d normalization)

These features are chosen to reflect regime, risk, and signal reliability, not price direction.

## 3. Label Construction

Labels answer a specific economic question:

If I enter this trade tomorrow, will it outperform transaction costs over the next N days?

Entry: `t + 1`
Exit: `t + 1 + horizon`

Label = 1 if forward return > transaction cost
Labels are only defined when the EMA signal is active
This avoids training the model on irrelevant periods.

## 4. Machine Learning Model

Architecture: Lightweight MLP (feed-forward neural network)
Loss: `BCEWithLogitsLoss`
Output: Probability of trade success

Regularization:

Dropout
Early stopping
Reduced model capacity
Scaling: StandardScaler (fit on train only)

The model does not predict returns, only trade viability.

## Backtesting Methodology ⚖️

To ensure fairness and research integrity:

Same price series
Same execution timing
Same transaction costs
Same position sizing
Same holding rules

The only difference:

`Raw Strategy:       EMA Signal`
`ML-Filtered:        EMA Signal × ML Filter`

## Evaluation Metrics

Classification Diagnostics
Train vs test accuracy
Probability distributions
Calibration curve
Confidence stability over time
Trading Metrics
Sharpe ratio
Trade frequency
Cumulative returns
Drawdowns
Acceptance rate vs threshold
Performance is evaluated out-of-sample only.

### Visualization Philosophy 📊

The project uses both static and interactive plots, chosen intentionally:

#### Matplotlib (Static, Research-Grade)

Cumulative return comparison
Drawdown curves
Calibration curves
Distribution histograms

#### Plotly (Interactive, Exploratory)

Threshold × Trade Frequency × Sharpe (3D)
ML confidence vs forward returns
PCA feature projections with confidence
Threshold sensitivity surfaces
Interactive plots are used only where dimensionality or non-linearity matters.

### Key Findings (Representative)

ML probabilities are stable and well-calibrated
Classification accuracy is modest (expected)
Trade frequency decreases meaningfully with ML filtering
Sharpe ratio improvements come primarily from risk reduction, not higher returns
In some regimes, Sharpe parity with lower drawdown is the dominant gain

This aligns with empirical trading literature:

Filtering bad trades is often more valuable than finding new ones.
Limitations & Research Extensions
No regime-specific models (single global classifier)
No short selling
Fixed holding horizon
No walk-forward retraining
No transaction cost stress testing

Natural next steps:

Regime-conditioned filters
Horizon-adaptive labeling
Bayesian or ensemble filters
Walk-forward optimization
Multi-asset generalization

### Reproducibility

Fixed random seeds
Deterministic splits
Explicit feature alignment
No hidden state or data leakage

All results can be reproduced by running:

`python train_model.py`
`python backtest_ml_strategy.py`
