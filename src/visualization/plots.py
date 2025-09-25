import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt  # Added missing import
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Data Cleaning Function ---
def data_cleaner(input_path: str, output_path: str, start_date_str: str = None, end_date_str: str = None) -> pd.DataFrame | None:
    """
    Loads, cleans, filters, and saves Bitcoin daily OHLCV data.
    """
    print(f"--- Starting Data Cleaning ---")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    
    # Validate date formats early
    try:
        if start_date_str:
            pd.to_datetime(start_date_str)
        if end_date_str:
            pd.to_datetime(end_date_str)
    except ValueError as e:
        print(f"Error: Invalid date format. Use 'YYYY-MM-DD'. Error: {e}")
        return None

    try:
        # Check if file exists and has content
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at '{input_path}'")
            return None
            
        file_size = os.path.getsize(input_path)
        if file_size == 0:
            print(f"Error: Input file is empty: '{input_path}'")
            return None

        df = pd.read_csv(input_path)
        print(f"Raw data loaded. Shape: {df.shape}")

        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return None
            
        if 'Date' not in df.columns:
            print("Error: 'Date' column not found in the input file.")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Convert and set date index
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        if df["Date"].isnull().any():
            print("Warning: Some dates could not be parsed and were set to NaT.")
            
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        original_shape = df.shape
        print(f"Original date range: {df.index.min()} to {df.index.max()}")

        # Date filtering
        if start_date_str:
            start_date = pd.to_datetime(start_date_str)
            df = df[df.index >= start_date]
            print(f"Filtered data from {start_date.strftime('%Y-%m-%d')}.")
            
        if end_date_str:
            end_date = pd.to_datetime(end_date_str)
            df = df[df.index <= end_date]
            print(f"Filtered data up to {end_date.strftime('%Y-%m-%d')}.")

        if df.empty:
            print(f"Warning: DataFrame is empty after date filtering.")
            print(f"Requested range: {start_date_str} to {end_date_str}")
            print(f"Available range: {original_shape}")
            return df

        # Handle missing values more robustly
        print("Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # First try forward fill, then backward fill
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # For any remaining NaNs, use interpolation
        if df.isnull().sum().sum() > 0:
            df.interpolate(method='time', inplace=True)
            
        missing_after = df.isnull().sum().sum()
        print(f"Missing values: {missing_before} -> {missing_after}")

        # Clean up unnecessary columns
        columns_to_drop = ["Dividends", "Stock Splits", "Capital Gains"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df.drop(columns=existing_columns_to_drop, inplace=True)
            print(f"Dropped columns: {existing_columns_to_drop}")

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}.")
            return None

        # Ensure numeric data types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Save cleaned data
        df.to_csv(output_path)
        print(f"Data cleaning completed. Saved to '{output_path}'.")
        print(f"Final DataFrame shape: {df.shape}")
        print(f"Final date range: {df.index.min()} to {df.index.max()}")
        
        return df

    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        return None

# --- Enhanced Candlestick Plotting Function ---
def plot_candlestick(df: pd.DataFrame, plot_title: str, volume_plot: bool = True, 
                     plot_start_date_str: str = None, plot_end_date_str: str = None,
                     save_path: str = None, style: str = 'yahoo'):
    """
    Enhanced candlestick plotting with better error handling and customization.
    """
    print(f"\n--- Generating Candlestick Plot ---")
    print(f"Plot Title: '{plot_title}'")
    
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided. Cannot plot.")
        return None, None

    df_plot = df.copy()

    # Date filtering for plot
    if plot_start_date_str:
        try:
            start_date = pd.to_datetime(plot_start_date_str)
            df_plot = df_plot[df_plot.index >= start_date]
            print(f"Filtered plot data from {start_date.strftime('%Y-%m-%d')}.")
        except Exception as e:
            print(f"Error parsing start date: {e}")

    if plot_end_date_str:
        try:
            end_date = pd.to_datetime(plot_end_date_str)
            df_plot = df_plot[df_plot.index <= end_date]
            print(f"Filtered plot data up to {end_date.strftime('%Y-%m-%d')}.")
        except Exception as e:
            print(f"Error parsing end date: {e}")

    if df_plot.empty:
        print("Error: No data available for the specified plot date range.")
        return None, None

    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df_plot.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}.")
        return None, None

    # Ensure proper index
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        print("Converting index to DatetimeIndex...")
        try:
            df_plot.index = pd.to_datetime(df_plot.index)
            df_plot.sort_index(inplace=True)
        except Exception as e:
            print(f"Could not convert index to DatetimeIndex: {e}")
            return None, None

    # Select only required columns
    df_plot = df_plot[required_cols].copy()

    try:
        # Configure plot style
        mpf_style = style
        
        # Create plot
        if volume_plot:
            ap = [mpf.make_addplot(df_plot['Volume'], panel=1, type='bar', 
                                 color='gray', alpha=0.7, ylabel='Volume')]
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
                returnfig=True,
                volume=volume_plot
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

        # Save plot if path provided
        if save_path:
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
        print("Plot generated successfully.")
        return fig, axes
        
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        return None, None

# --- Enhanced Plotting Scenarios with Better Configuration ---
def plot_full_history_analysis(raw_csv_path: str, processed_csv_dir: str):
    """Enhanced full history analysis with better reporting."""
    print("\n" + "="*60)
    print(">>> Running Analysis: Full History <<<")
    print("="*60)
    
    full_data_output_path = os.path.join(processed_csv_dir, "btc_daily_cleaned_full.csv")
    
    cleaned_df_full = data_cleaner(
        input_path=raw_csv_path, 
        output_path=full_data_output_path
    )

    if cleaned_df_full is not None and not cleaned_df_full.empty:
        # Generate descriptive title
        date_range = f"{cleaned_df_full.index.min().strftime('%Y-%m-%d')} to {cleaned_df_full.index.max().strftime('%Y-%m-%d')}"
        title = f"Bitcoin (BTC-USD) Daily Chart - Full History ({date_range})"
        
        fig, axes = plot_candlestick(
            df=cleaned_df_full, 
            plot_title=title,
            volume_plot=True,
            save_path=os.path.join(processed_csv_dir, "plots/full_history.png")
        )
        return cleaned_df_full, fig
    else:
        print("Skipping full history plot: Cleaning failed or no data was processed.")
        return None, None

def plot_recent_period_analysis(raw_csv_path: str, processed_csv_dir: str, years: int = 2):
    """Enhanced recent period analysis with configurable timeframe."""
    print("\n" + "="*60)
    print(f">>> Running Analysis: Recent {years} Year(s) <<<")
    print("="*60)
    
    recent_data_output_path = os.path.join(processed_csv_dir, f"btc_daily_cleaned_recent_{years}y.csv")
    
    # Calculate dynamic date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    recent_start_date_str = start_date.strftime("%Y-%m-%d")
    recent_end_date_str = end_date.strftime("%Y-%m-%d")

    cleaned_df_recent = data_cleaner(
        input_path=raw_csv_path,
        output_path=recent_data_output_path,
        start_date_str=recent_start_date_str,
        end_date_str=recent_end_date_str
    )

    if cleaned_df_recent is not None and not cleaned_df_recent.empty:
        title = f"Bitcoin (BTC-USD) Daily Chart - Last {years} Year(s) ({recent_start_date_str} to {recent_end_date_str})"
        
        fig, axes = plot_candlestick(
            df=cleaned_df_recent,
            plot_title=title,
            volume_plot=True,
            save_path=os.path.join(processed_csv_dir, f"plots/recent_{years}y.png")
        )
        return cleaned_df_recent, fig
    else:
        print(f"Skipping recent {years} year plot: Cleaning failed or no data.")
        return None, None

def plot_custom_period_analysis(raw_csv_path: str, processed_csv_dir: str, 
                               start_date: str, end_date: str, period_name: str = "Custom"):
    """Enhanced custom period analysis."""
    print("\n" + "="*60)
    print(f">>> Running Analysis: {period_name} Period <<<")
    print("="*60)
    
    filename = f"btc_daily_cleaned_{period_name.lower().replace(' ', '_')}.csv"
    custom_data_output_path = os.path.join(processed_csv_dir, filename)

    cleaned_df_custom = data_cleaner(
        input_path=raw_csv_path,
        output_path=custom_data_output_path,
        start_date_str=start_date,
        end_date_str=end_date
    )

    if cleaned_df_custom is not None and not cleaned_df_custom.empty:
        title = f"Bitcoin (BTC-USD) Daily Chart - {period_name} ({start_date} to {end_date})"
        
        fig, axes = plot_candlestick(
            df=cleaned_df_custom,
            plot_title=title,
            volume_plot=True,
            save_path=os.path.join(processed_csv_dir, f"plots/{period_name.lower().replace(' ', '_')}.png")
        )
        return cleaned_df_custom, fig
    else:
        print(f"Skipping {period_name} period plot: Cleaning failed or no data.")
        return None, None

# --- Main Execution Function ---
def main():
    """Main function to run all analyses."""
    print("🚀 Bitcoin Daily Navigator - Analysis Pipeline")
    print("=" * 50)
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/btc_daily_ohlcv.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(PROCESSED_DIR, "plots"), exist_ok=True)
    
    # Check if raw data exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Raw data file not found at: {RAW_DATA_PATH}")
        print("Please ensure your CSV file exists in the correct location.")
        return
    
    # Run analyses
    results = {}
    
    # 1. Full history analysis
    results['full_history'] = plot_full_history_analysis(RAW_DATA_PATH, PROCESSED_DIR)
    
    # 2. Recent periods (1 year, 2 years)
    results['recent_1y'] = plot_recent_period_analysis(RAW_DATA_PATH, PROCESSED_DIR, years=1)
    results['recent_2y'] = plot_recent_period_analysis(RAW_DATA_PATH, PROCESSED_DIR, years=2)
    
    # 3. Custom historical periods
    bull_market_2017 = plot_custom_period_analysis(
        RAW_DATA_PATH, PROCESSED_DIR, 
        start_date="2017-01-01", end_date="2017-12-31",
        period_name="2017 Bull Market"
    )
    
    covid_crash_2020 = plot_custom_period_analysis(
        RAW_DATA_PATH, PROCESSED_DIR,
        start_date="2020-01-01", end_date="2020-06-30", 
        period_name="2020 COVID Crash"
    )
    
    print("\n" + "="*50)
    print("✅ Analysis Pipeline Completed!")
    print("="*50)
    
    # Summary report
    successful_analyses = [k for k, v in results.items() if v[0] is not None]
    print(f"Successful analyses: {len(successful_analyses)}")
    print(f"Check the 'data/processed/' directory for results.")

if __name__ == "__main__":
    main()