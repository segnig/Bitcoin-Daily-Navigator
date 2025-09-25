import pandas as pd
import os

def data_cleaner(input_path: str, output_path: str, start_date_str: str = None, end_date_str: str = None) -> pd.DataFrame | None:
    """
    Loads, cleans, filters, and saves Bitcoin daily OHLCV data.

    Args:
        input_path (str): Path to the raw CSV file (e.g., 'data/raw/btc_daily_ohlcv.csv').
        output_path (str): Path where the cleaned CSV will be saved (e.g., 'data/processed/btc_daily_cleaned.csv').
        start_date_str (str, optional): Start date for filtering data in 'YYYY-MM-DD' format. Defaults to None (no start date filter).
        end_date_str (str, optional): End date for filtering data in 'YYYY-MM-DD' format. Defaults to None (no end date filter).

    Returns:
        pd.DataFrame | None: The cleaned and filtered DataFrame, or None if an error occurs or no data is found.
    """
    print(f"--- Starting Data Cleaning ---")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    if start_date_str: print(f"Filtering from: {start_date_str}")
    if end_date_str: print(f"Filtering to: {end_date_str}")

    try:
        # Load the raw data
        df = pd.read_csv(input_path)

        # 'Date' column exists
        if 'Date' not in df.columns:
            print("Error: 'Date' column not found in the input file.")
            return None
            
        # Convert 'Date' to datetime objects and set as index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index('Date', inplace=True)
        
        # Sort by date to ensure chronological order
        df.sort_index(inplace=True)

        # Filter by date range if provided
        original_shape = df.shape
        if start_date_str:
            start_date = pd.to_datetime(start_date_str)
            df = df[df.index >= start_date]
            print(f"Filtered data from {start_date.strftime('%Y-%m-%d')}.")
        if end_date_str:
            end_date = pd.to_datetime(end_date_str)
            df = df[df.index <= end_date]
            print(f"Filtered data up to {end_date.strftime('%Y-%m-%d')}.")
        
        if df.empty:
            print(f"Warning: DataFrame is empty after date filtering (Original shape: {original_shape}, Filtered shape: {df.shape}).")
            return df # Return empty df if no data matches the filter

        # 5. Handle missing values using ffill and bfill for time-series
        print("Handling missing values using forward fill (ffill) and backward fill (bfill)...")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True) # Handles NaNs at the very beginning if ffill couldn't

        # Check if any NaNs remain (e.g., if an entire column was NaN)
        if df.isnull().sum().sum() > 0:
            print("Warning: Some NaNs remain after ffill/bfill. This might indicate issues with the source data.")
            

        # Drop unnecessary columns if they exist
        columns_to_drop = ["Dividends", "Stock Splits"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df.drop(columns=existing_columns_to_drop, inplace=True)
            print(f"Dropped columns: {existing_columns_to_drop}")
        else:
            print("No 'Dividends' or 'Stock Splits' columns found to drop.")

        # Ensure required columns are present for plotting/analysis
        required_cols_for_plot = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols_for_plot):
            missing = [col for col in required_cols_for_plot if col not in df.columns]
            print(f"Error: Missing columns required for analysis/plotting: {missing}.")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Save cleaned data to the specified output path
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        df.to_csv(output_path)
        print(f"Data cleaning completed. Saved to '{output_path}'.")
        print(f"Cleaned DataFrame shape: {df.shape} (Original shape was: {original_shape})")
        return df

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return None
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        return None