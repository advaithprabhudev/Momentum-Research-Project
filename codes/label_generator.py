import pandas as pd
from feature_generation import *


def generate_labels(
        price_series: pd.Series,
        signal_series: pd.Series,
        time_horizon: int = 5,
        transaction_cost: float = 0.001
):
    entry_price = price_series.shift(-1)
    exit_price = price_series.shift(-(1 + time_horizon))
    future_return = (exit_price - entry_price) / entry_price
    success = future_return > transaction_cost

    labels = success.where(signal_series == 1)
    return labels.dropna().astype(int)


if __name__ == "main":
    from data_ingestion import fetch_data
    _df = fetch_data()
    _ema_signal = ((_df["Close"].ewm(span=20, adjust=False).mean()) > (
        _df["Close"].ewm(span=50, adjust=False).mean())).astype(int)
    label = generate_labels(
        price_series=_df["Close"], signal_series=_ema_signal)
