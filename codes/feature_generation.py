import pandas as pd
from data_ingestion import fetch_data
import pandas_ta as ta

df = fetch_data()

# Exponential Moving Average 20 Days


def ema_20():
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df["EMA_20"]

# Exponential Moving Average 50 Days


def ema_50():
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df["EMA_50"]

# Calculating Daily, 5 Day, and 10 Day Returns


def daily_returns():
    returns = df["Close"].pct_change(1)
    return returns


def five_day_return():
    day_return = df["Close"].pct_change(5)
    return day_return


def ten_day_returns():
    day_return = df["Close"].pct_change(10)
    return day_return


dr = daily_returns()

# Feature Engineering


def volatility():
    return dr.rolling(window=20).std().shift(1)


def relative_index():
    return ta.rsi(close=df["Close"], length=14).shift(1)


def average_range():
    return ta.atr(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        length=14
    ).shift(1)


def z_score():
    rolling_mean = df["Volume"].rolling(window=252).mean()
    rolling_std = df["Volume"].rolling(window=252).std()
    return ((df["Volume"] - rolling_mean) / rolling_std).shift(1)


def features():
    return pd.DataFrame({
        "Volatility_20": volatility(),
        "RSI": relative_index(),
        "ATR": average_range(),
        "Z_Score": z_score()
    })
