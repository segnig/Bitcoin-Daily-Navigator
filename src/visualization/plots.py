import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime


def plot_candlestick(
    df: pd.DataFrame,
    plot_title: str,
    volume_plot: bool = True,
    start_date: str | datetime = None,
    end_date: str | datetime = None
):
    """
    Generates and displays a candlestick plot for Bitcoin price data.

    Args:
        df (pd.DataFrame): The DataFrame containing cleaned OHLCV data with a DatetimeIndex.
        plot_title (str): The title for the plot.
        volume_plot (bool, optional): Whether to include the volume subplot. Defaults to True.
        start_date (str or datetime, optional): Start date for plotting. Can be 'YYYY-MM-DD' or datetime. Defaults to None.
        end_date (str or datetime, optional): End date for plotting. Can be 'YYYY-MM-DD' or datetime. Defaults to None.
    """
    print(f"\n--- Generating Candlestick Plot ---")
    print(f"Plot Title: '{plot_title}'")
    if start_date: print(f"Plotting from: {start_date}")
    if end_date: print(f"Plotting to: {end_date}")
    if volume_plot: print("Including Volume subplot.")

    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided. Cannot plot.")
        return

    df_plot = df.copy()  # Work on a copy to avoid modifying the original DataFrame

    # --- Ensure index is DatetimeIndex and sorted ---
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not a DatetimeIndex. Attempting conversion.")
        try:
            df_plot.index = pd.to_datetime(df_plot.index)
        except Exception as e:
            print(f"Could not convert index to DatetimeIndex: {e}")
            return
    df_plot.sort_index(inplace=True)

    # --- Filter by date range for plotting if specified ---
    def _to_timestamp(dt):
        if isinstance(dt, str):
            ts = pd.to_datetime(dt)
        elif isinstance(dt, datetime):
            ts = pd.Timestamp(dt)
        else:
            return None

        # Match timezone if index is timezone-aware
        if df_plot.index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(df_plot.index.tz)
        return ts

    if start_date:
        start_ts = _to_timestamp(start_date)
        if start_ts is not None:
            df_plot = df_plot[df_plot.index >= start_ts]
            print(f"Filtered plot data from {start_ts.strftime('%Y-%m-%d')}.")

    if end_date:
        end_ts = _to_timestamp(end_date)
        if end_ts is not None:
            df_plot = df_plot[df_plot.index <= end_ts]
            print(f"Filtered plot data up to {end_ts.strftime('%Y-%m-%d')}.")

    if df_plot.empty:
        print("Error: No data available for the specified plot date range after filtering.")
        return

    # --- Ensure DataFrame has all required columns ---
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df_plot.columns]
    if missing:
        print(f"Error: Missing required columns for mplfinance in DataFrame: {missing}")
        return

    df_plot = df_plot[required_cols].copy()

    # --- Generate the Plot ---
    mpf_style = 'binance'
    try:
        if volume_plot:
            ap = [mpf.make_addplot(df_plot['Volume'], panel=1, type='bar', color='gray', alpha=0.7, ylabel='Volume')]
            fig, axes = mpf.plot(
                df_plot,
                title=plot_title,
                ylabel='Price (USD)',
                addplot=ap,
                panel_ratios=(3, 1),
                figscale=1.5,
                type='candle',
                style=mpf_style,
                ylabel_lower='Volume',
                returnfig=True
            )
        else:
            fig, axes = mpf.plot(
                df_plot,
                title=plot_title,
                ylabel='Price (USD)',
                figscale=1.5,
                type='candle',
                style=mpf_style,
                returnfig=True
            )

        plt.show()
        print("Plot generated successfully.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")