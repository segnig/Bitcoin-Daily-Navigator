import pandas as pd
import numpy as np
import os
# Uncomment the line below if you have TA-Lib installed and want to use it
import talib

def create_features(df: pd.DataFrame, technical_indicator_library: str = 'pandas') -> pd.DataFrame:
    """
    Adds various features to a cleaned Bitcoin daily OHLCV DataFrame.

    Features include:
    - Daily Returns
    - Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, OBV)
    - Lagged Features (previous day's Close, Return, Volume)
    - Interaction Features (Price vs. SMA, Volume vs. Avg Volume)

    Args:
        df (pd.DataFrame): A pandas DataFrame containing cleaned OHLCV data.
                           Must have columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
                           Index must be a DatetimeIndex and sorted chronologically.
        technical_indicator_library (str): Specifies which library to use for indicators.
                                           'pandas' (default) uses pandas implementations.
                                           'talib' uses the TA-Lib library (requires installation).

    Returns:
        pd.DataFrame: The DataFrame with engineered features added. Returns None if input df is invalid.
    """
    if df is None or df.empty:
        print("Error: Input DataFrame is empty or None. Cannot create features.")
        return None

    # Ensure DataFrame is a copy to avoid modifying the original outside the function
    df_features = df.copy()

    print("\n--- Starting Feature Engineering ---")
    
    # --- 1. Calculate Daily Returns ---
    print("Calculating Daily Returns...")
    df_features['daily_return'] = df_features['Close'].pct_change()
    print("  - Calculated 'daily_return'.")

    # --- 2. Calculate Technical Indicators ---
    print(f"Calculating Technical Indicators using '{technical_indicator_library}'...")
    
    try:
        if technical_indicator_library.lower() == 'talib':
            # --- TA-Lib Implementation ---
            # Make sure TA-Lib is installed: pip install TA-Lib
            # On some systems, you might need to install the C library first: sudo dnf install ta-lib-devel
            try:
                import talib 
            except ImportError:
                print("  - Warning: TA-Lib not found. Falling back to pandas implementation.")
                technical_indicator_library = 'pandas' # Fallback to pandas

        if technical_indicator_library.lower() == 'talib':
            # TA-Lib functions typically require numpy arrays
            close_prices = df_features['Close'].values
            volumes = df_features['Volume'].values

            # Moving Averages
            df_features['SMA_5'] = talib.SMA(close_prices, timeperiod=5)
            df_features['EMA_5'] = talib.EMA(close_prices, timeperiod=5)
            df_features['SMA_10'] = talib.SMA(close_prices, timeperiod=10)
            df_features['EMA_10'] = talib.EMA(close_prices, timeperiod=10)

            # RSI
            df_features['RSI'] = talib.RSI(close_prices, timeperiod=14)

            # MACD
            macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            df_features['MACD'] = macd
            df_features['MACD_Signal'] = macdsignal

            # Bollinger Bands
            window_bb_talib = 20
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=window_bb_talib, nbdevup=2, nbdevdn=2, matype=0) 
            df_features['Bollinger_Upper'] = upper
            df_features['Bollinger_Lower'] = lower
            df_features['SMA_BB'] = middle # TA-Lib's BBANDS also returns the SMA used for calculation

            # OBV
            df_features['OBV'] = talib.OBV(close_prices, volumes)
            print("  - Calculated indicators using TA-Lib.")

        else: # Pandas Implementation
            # --- Moving Averages ---
            df_features['SMA_5'] = df_features['Close'].rolling(window=5).mean()
            df_features['EMA_5'] = df_features['Close'].ewm(span=5, adjust=False).mean()
            df_features['SMA_10'] = df_features['Close'].rolling(window=10).mean()
            df_features['EMA_10'] = df_features['Close'].ewm(span=10, adjust=False).mean()

            # --- RSI ---
            delta = df_features['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(com=13, adjust=False).mean() 
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / avg_loss
            df_features['RSI'] = 100 - (100 / (1 + rs))

            # --- MACD ---
            ema_12 = df_features['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df_features['Close'].ewm(span=26, adjust=False).mean()
            df_features['MACD'] = ema_12 - ema_26
            df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean() 

            # --- Bollinger Bands ---
            window_bb = 20
            df_features['SMA_BB'] = df_features['Close'].rolling(window=window_bb).mean()
            df_features['STD_BB'] = df_features['Close'].rolling(window=window_bb).std()
            df_features['Bollinger_Upper'] = df_features['SMA_BB'] + (df_features['STD_BB'] * 2)
            df_features['Bollinger_Lower'] = df_features['SMA_BB'] - (df_features['STD_BB'] * 2)

            # --- OBV ---
            df_features['OBV'] = (df_features['Volume'] * np.sign(df_features['Close'].diff())).cumsum()
            
            print("  - Calculated indicators using Pandas.")

        # Drop intermediate columns used for calculation if they are not needed as features
        cols_to_drop_intermediates = ['SMA_BB', 'STD_BB'] 
        for col in cols_to_drop_intermediates:
            if col in df_features.columns:
                df_features.drop(columns=[col], inplace=True)
                
    except Exception as e:
        print(f"  - An error occurred during technical indicator calculation: {e}")


    # --- 3. Create Lagged Features ---
    print("Creating Lagged Features...")
    df_features['Close_Lag_1'] = df_features['Close'].shift(1)
    df_features['Close_Lag_2'] = df_features['Close'].shift(2)
    df_features['Close_Lag_3'] = df_features['Close'].shift(3)
    df_features['Return_Lag_1'] = df_features['daily_return'].shift(1)
    df_features['Volume_Lag_1'] = df_features['Volume'].shift(1)
    print("  - Created lagged features.")

    # --- 4. Create Interaction Features ---
    print("Creating Interaction Features...")
    if 'SMA_10' in df_features.columns:
        df_features['Price_vs_SMA10'] = df_features['Close'] / df_features['SMA_10']
        print("  - Created 'Price_vs_SMA10' feature.")

    if 'Volume_Lag_1' in df_features.columns: # Check if lag feature exists
        # Calculate a rolling average of volume to compare against
        df_features['Avg_Volume_10'] = df_features['Volume'].rolling(window=10).mean()
        df_features['Volume_vs_Avg_Vol10'] = df_features['Volume'] / df_features['Avg_Volume_10']
        # Drop the intermediate 'Avg_Volume_10' column if not needed as a standalone feature
        if 'Avg_Volume_10' in df_features.columns:
             df_features.drop(columns=['Avg_Volume_10'], inplace=True)
        print("  - Created 'Volume_vs_Avg_Vol10' feature.")
    
    # --- 5. Handle NaNs Introduced by Feature Engineering ---
    print("Handling NaNs introduced by feature engineering...")
    initial_rows = df_features.shape[0]
    
    # Drop rows where ANY engineered feature has NaN.
    # This is crucial before modeling, as models cannot handle NaNs.
    # Identify which columns actually have NaNs to report more clearly.
    nan_cols_before = df_features.columns[df_features.isnull().any()].tolist()
    
    # Drop rows with NaNs
    df_features.dropna(inplace=True)
    
    final_rows = df_features.shape[0]
    print(f"  - Removed {initial_rows - final_rows} rows with NaNs.")
    print(f"  - DataFrame shape after dropping NaNs: {df_features.shape}")
    
    # Final check for any remaining NaNs (should ideally be 0 after dropna)
    if df_features.isnull().sum().sum() > 0:
        print("  - Warning: Some NaNs still exist. This is unexpected after dropna().")

    print("Feature engineering completed.")
    return df_features