import yfinance as yf
import pandas as pd

t = "NVDA"


def fetch_data(ticker=t):
    stock = yf.download(ticker, start="2010-01-01", auto_adjust=False)

    if stock.empty:
        print("Download failed. Exiting...")
        raise RuntimeError(f"YFinance failed to download the ticker.")

    df = pd.DataFrame(stock)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
