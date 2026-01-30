import pandas as pd
import numpy as np
from data_ingestion import fetch_data
from feature_generation import *
import matplotlib.pyplot as plt

df = fetch_data()

df["EMA_20"] = ema_20()
df["EMA_50"] = ema_50()

# Buy/Sell Implmentation

df["EMA20_prev"] = df["EMA_20"].shift(1)
df["EMA50_prev"] = df["EMA_50"].shift(1)

# Buy: EMA20 crosses above EMA50
df["Buy"] = (df["EMA_20"] > df["EMA_50"]) & (
    df["EMA20_prev"] <= df["EMA50_prev"])
# Sell: EMA20 crosses below EMA50
df["Sell"] = (df["EMA_20"] < df["EMA_50"]) & (
    df["EMA20_prev"] >= df["EMA50_prev"])

# Buy/Sell Prices
df["Buy_Price"] = df["Close"].where(df["Buy"])
df["Sell_Price"] = df["Close"].where(df["Sell"])

# Giving baseline metrics
daily_return = daily_returns()


# Sharpe Ratio Calculation Annualized
sharpe_ratio = daily_returns().rolling(window=252).mean(
) / daily_returns().rolling(window=30).std() * np.sqrt(252)

# Drawdown Calculation
initial_capital = 100_000
equity = []
current_capital = initial_capital
in_trade = False
previous_price = None

for idx, row in df.iterrows():
    if row["Buy"] and not in_trade:
        in_trade = True
        previous_price = row["Close"]

    if in_trade:
        daily_ret = (row["Close"] - previous_price) / previous_price
        current_capital *= (1 + daily_ret)
        previous_price = row["Close"]

    equity.append(current_capital)

    if row["Sell"] and in_trade:
        in_trade = False
        previous_price = None  # reset


df["Equity"] = equity
df["Rolling_Max"] = df["Equity"].cummax()
df["Drawdown"] = (df["Equity"] - df["Rolling_Max"]) / df["Rolling_Max"]

# Compounded Annual Growth Rate (CAGR) Calculation

start_value = df["Equity"].iloc[0]
end_value = df["Equity"].iloc[-1]
years = len(df) / 252
cagr = (end_value / start_value) ** (1 / years) - 1

# Checking if it works
print(df[["Equity", "Rolling_Max", "Drawdown"]].tail(10))
print(f"CAGR : {cagr}")

# Implementation of logic for buy/sell prices, one buy -> one sell
trades = []
in_trade = False
buy_date = buy_price = None

for idx, row in df.iterrows():
    if row["Buy"] and not in_trade:
        buy_date = idx
        buy_price = row["Close"]
        in_trade = True

    elif row["Sell"] and in_trade:
        sell_date = idx
        sell_price = row["Close"]

        trades.append({
            "Entry_Date": buy_date,
            "Exit_Date": sell_date,
            "Entry_Price": buy_price,
            "Exit_Price": sell_price,
            "Returns": (sell_price - buy_price) / buy_price
        })

        in_trade = False

trades = pd.DataFrame(trades)


# Feature Engineering

five_day_returns = five_day_return()
ten_day_return = ten_day_returns()
twenty_day_volatility = volatility()
relative_strength_index = relative_index()
average_true_range = average_range()
volume_z_score = z_score()

trade_features = pd.DataFrame({
    "Daily_Return": daily_return,
    "Five_Day_Return": five_day_return,
    "Ten_Day_Return": ten_day_return,
    "Volatility": twenty_day_volatility,
    "RSI": relative_strength_index,
    "ATR": average_true_range,
    "Volume_Z_Score": volume_z_score
})

# Plotting Equity Curve

plt.figure(1)
plt.plot(df.index, df["Equity"], label="Equity Curve", c="green")
plt.plot(df.index, df["Rolling_Max"], label="Rolling Max", c="orange")
plt.fill_between(df.index, df["Rolling_Max"], df["Equity"],
                 color='red', alpha=0.2, label="Drawdown")
plt.grid(True)
plt.legend()
plt.title("Plotting the Equity Curve of AAPL")
plt.show()
