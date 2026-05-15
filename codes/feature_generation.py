import pandas as pd
from data_ingestion import fetch_data
import pandas_ta as ta


# Exponential Moving Average 20 Days


def ema_20(df):
    return df["Close"].ewm(span=20, adjust=False).mean()

# Exponential Moving Average 50 Days


def ema_50(df):
    return df["Close"].ewm(span=50, adjust=False).mean()

# Calculating Daily, 5 Day, and 10 Day Returns


def daily_returns(df):
    return df["Close"].pct_change(1)


def five_day_return(df):
    return df["Close"].pct_change(5)


def ten_day_returns(df):
    return df["Close"].pct_change(10)


# Feature Engineering


def volatility(df):
    dr = daily_returns(df)
    return dr.rolling(window=20).std().shift(1)


def relative_index(df):
    return ta.rsi(close=df["Close"], length=14).shift(1)


def average_range(df):
    a = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=14)
    return (a / df["Close"]).shift(1)


def z_score(df):
    rolling_mean = df["Volume"].rolling(window=252).mean()
    rolling_std = df["Volume"].rolling(window=252).std()
    return ((df["Volume"] - rolling_mean) / rolling_std).shift(1)


def features(df):
    return pd.DataFrame({
        "Volatility_20": volatility(df),
        "RSI": relative_index(df),
        "ATR": average_range(df),
        "Z_Score": z_score(df)
    })
