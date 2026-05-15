# What role does machine learning play in the momentum strategy?

## What Machine Learning does in this project?

The ML Model predicts trade level connditional profitablity, not price movement.

### What each label represents?

My label is using Bernoulli's random variable, which:

* Estimates the edge conditional 
* Output is probability not a signal

### Main Point

The model predicts quality of signals, not price direction.

## What the ML Model does not predict?

* Price Direction
* Trend of Price (Up/Down)
* Exact Returns
* Entry Timing
* Exit Timing
* Market Regimes

## What the ML Model does predict?

Takes market data -> Momentum Strategy (EMA 20/ EMA 50) - > Classify the trade (buy/sell) -> ML Filtering -> Final Decision

#### The ML Model improves the signals, it doesn't replace them

## Decision

### How thresholds are chosen?

It is not arbitrary.

Thresholds are optimized against:

* Sharpe Ratio
* Drawdown
* Frequency

