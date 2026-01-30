import pandas as pd
from feature_generation import *


def generate_labels(
        price_series: pd.Series,
        signal_series: pd.Series,
        time_horzion: int = 5,
        transaction_cost: float = 0.001
):
    entry_price = price_series.shift(-1)
    exit_price = price_series.shift(-(1 + time_horzion))
    future_return = (exit_price - entry_price) / entry_price
    success = future_return > transaction_cost

    labels = success.where(signal_series == 1)
    return labels.dropna().astype(int)


label = generate_labels(price_series=df["Close"], signal_series=(
    (ema_20() > ema_50()).astype(int)))
print(label)
