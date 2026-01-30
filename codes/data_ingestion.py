import yfinance as yf
import pandas as pd


def fetch_data(ticker="AAPL"):
    stock = yf.download(ticker, start="2010-01-01", auto_adjust=False)

    if stock.empty:
        print("Download failed. Exiting...")
        exit()

    df = pd.DataFrame(stock)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
