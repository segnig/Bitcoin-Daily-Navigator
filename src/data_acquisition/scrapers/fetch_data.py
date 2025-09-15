import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta


def fetch_and_save_btc_data(period="5y", OUTPUT_PATH="data/raw/btc_daily_ohlcv.csv"):
    TICKER = "BTC-USD"

    print(f"--- Fetching daily data for {TICKER} ({period}) ---")

    try:

        btc_ticker = yf.Ticker(TICKER)

        end_date = datetime.now()

        if period == "5y":
            start_date = end_date - timedelta(days=5 * 365)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = None

        if start_date:
            daily_data = btc_ticker.history(
                start=start_date.strftime('%Y-%m-%d'), 
                end=end_date.strftime('%Y-%m-%d'), 
                interval="1d"
                )
        else:
            daily_data = btc_ticker.history(period=period, )

        if daily_data.empty:
            print("Error: Fetched data is empty. Check ticker symbol or date range.")
            return

        data_dir = os.path.exists(OUTPUT_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")

        # Save the data
        daily_data.to_csv(OUTPUT_PATH)
        print(f"\nSuccessfully fetched {len(daily_data)} rows of data.")
        print(f"Data saved to: {OUTPUT_PATH}")
        print(f"Date Range: {daily_data.index.min().strftime('%Y-%m-%d')} to {daily_data.index.max().strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"An error occurred during data fetching: {e}")

if __name__ == "__main__":
    fetch_and_save_btc_data(period='max')
