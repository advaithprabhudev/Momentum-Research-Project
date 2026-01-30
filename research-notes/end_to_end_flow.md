# Workflow

1. Fetches OHLCV data -> `fetch_data()`
2. Compute features -> `features()`
3. EMA 20/EMA 50 signals  -> `ema_20()`, `ema_50()`
4. Label Generation ->  `generate_labels()`
5. MLP Model Selection -> `MLP()`
6. ML Filter -> `train_model.py`
7. Backtest strategy -> `train_model.py`
8. Results shown -> `results.md`

