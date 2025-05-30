import pandas as pd
import numpy as np
import hashlib


def generate_unique_id(symbol: str, date: str, title: str):
    """
    Generates a unique id from news or press release symbol, date and title
    :param self:
    :param symbol:
    :param date:
    :param title:
    :return:
    """
    # Combine the values into a single string
    unique_string = f"{symbol}:{date}:{title}"

    # Generate an MD5 hash of the combined string
    unique_id = hashlib.md5(unique_string.encode()).hexdigest()

    return unique_id


def standardize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes an OHLCV dataframe by renaming columns, handling infinites, dropping NaNs, and converting date column to datetime.

    Parameters:
        df (pd.DataFrame): The OHLCV data frame.

    Returns:
        pd.DataFrame: The standardized data frame.
    """
    # Define a mapping for renaming columns
    rename_mapping = {
        'Date': 'date',
        'Datetime': 'date',
        'DateTime': 'date',
        'Time': 'time',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'AdjClose': 'adj_close',
        'Adj Close': 'adj_close',
        'adjclose': 'adj_close',
        'Volume': 'volume'
    }

    # Rename columns
    df = df.rename(columns=str.lower)
    df = df.rename(columns=rename_mapping)

    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Forward fill
    df.ffill(inplace=True)

    # Ensure numeric columns have a numeric format
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

