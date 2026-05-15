# ML-Filtered EMA Momentum Strategy

A quantitative research project that applies a machine learning filter to a classical EMA crossover strategy on AAPL (2010–2026). The ML model does not predict price direction — it estimates the conditional probability that a momentum signal will be profitable, and filters out low-confidence trades.

## Requirements

- Python 3.9+
- Install dependencies:

```
pip install yfinance pandas numpy torch scikit-learn pandas_ta matplotlib plotly
```

## Running the Pipeline

Run each script in order:

```
python codes/data_ingestion.py
python codes/feature_generation.py
python codes/label_generator.py
python codes/baseline_strategy.py
python codes/train_model.py
python codes/figure_plotting.py
```

## Project Structure

```
codes/
  data_ingestion.py     — Fetches OHLCV data via yfinance
  feature_generation.py — Computes EMA 20/50, RSI, ATR, Volatility, Z-Score
  label_generator.py    — Labels each signal: 1 = profitable after 5 days, 0 = not
  baseline_strategy.py  — Raw EMA crossover backtest
  train_model.py        — Trains MLP classifier, sweeps threshold, runs filtered backtest
  figure_plotting.py    — Generates feature plots and correlation heatmap
research-notes/         — Strategy definitions, model rationale, backtest assumptions
```

## Results

| Metric          | Raw EMA | ML Filtered |
|-----------------|---------|-------------|
| Sharpe Ratio    | 1.217   | 1.254       |
| Max Drawdown    | -3.4%   | -2.8%       |
| Trades          | 522     | 463         |
| Test Accuracy   | —       | 48.85%      |

The ML filter reduces trade count by 11% and maximum drawdown by 17%. The sub-50% test accuracy is expected — the model is calibrated for risk reduction, not directional prediction.

## Notes

- Features are shifted 1 day forward to prevent look-ahead bias.
- Train/test split is chronological (not random) to preserve time-series integrity.
- Scaler is fit on training data only.
- Transaction costs are not included in the current backtest.
