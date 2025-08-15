import pandas as pd
from fredapi import Fred
import logging
from typing import Dict, Optional, Tuple

# Attempt to import config. If running as a script, this might fail, which is handled.
try:
    from config import FRED_API_KEY, FRED_INDICATOR_SERIES
except ImportError:
    FRED_API_KEY = None
    FRED_INDICATOR_SERIES = {}

class EconomicDataLoader:
    """
    Handles fetching economic data from the Federal Reserve Economic Data (FRED) API.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the FRED client.

        Args:
            api_key (str): The API key for the FRED API. If not provided, it will
                           fall back to the FRED_API_KEY from the config.
        """
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError("FRED API key is not set. Please set the FRED_API_KEY environment variable or pass it to the constructor.")

        try:
            self.fred = Fred(api_key=self.api_key)
        except Exception as e:
            logging.error(f"Failed to initialize Fred client: {e}")
            raise

    def fetch_series_data(self, series_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches data for a single FRED series and returns it as a DataFrame.

        Args:
            series_id (str): The ID of the FRED series (e.g., 'GDP').
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: A DataFrame with a 'date' index and 'value' column, or None if an error occurs.
        """
        try:
            logging.info(f"Fetching FRED data for series '{series_id}' from {start_date} to {end_date}...")
            data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

            if data.empty:
                logging.warning(f"No data returned for FRED series '{series_id}' for the given date range.")
                return None

            df = data.to_frame(name='value')
            df.index.name = 'date'
            # Forward-fill missing values, as economic data is often reported periodically
            df['value'] = df['value'].ffill()
            return df
        except Exception as e:
            logging.error(f"Failed to fetch data for FRED series '{series_id}': {e}", exc_info=False)
            return None

    def fetch_all_indicator_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for all indicators defined in FRED_INDICATOR_SERIES.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are indicator names (e.g., 'GDP')
                                     and values are their corresponding DataFrames.
        """
        all_data = {}
        if not FRED_INDICATOR_SERIES:
            logging.warning("FRED_INDICATOR_SERIES is not defined in config. Cannot fetch data.")
            return all_data

        for indicator_name, series_id in FRED_INDICATOR_SERIES.items():
            df = self.fetch_series_data(series_id, start_date, end_date)
            if df is not None:
                all_data[indicator_name] = df
        return all_data

if __name__ == '__main__':
    # This block allows for standalone testing of the data loader.
    # It requires a .env file with FRED_API_KEY in the same directory.
    import os
    from dotenv import load_dotenv

    # Since this is run as a script, we need to load .env and config manually
    print("Running economic_data_loader.py as a standalone script for testing...")

    # Assuming the script is in news_analyzer/event_driven, .env should be there too
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(".env file loaded.")
    else:
        print("Warning: .env file not found. Make sure FRED_API_KEY is set in your environment.")

    # Manually re-import FRED_API_KEY from environment after loading .env
    FRED_API_KEY_TEST = os.getenv("FRED_API_KEY")

    # Manually define series for testing if config import failed
    if not FRED_INDICATOR_SERIES:
        FRED_INDICATOR_SERIES = {
            "GDP": "GDP",
            "CPI": "CPIAUCSL",
            "FED_FUNDS": "FEDFUNDS",
            "UNEMPLOYMENT": "UNRATE",
        }
        print("Using default series for testing.")

    logging.basicConfig(level=logging.INFO)

    if not FRED_API_KEY_TEST:
        logging.error("FATAL: FRED_API_KEY is not set. Cannot run test.")
    else:
        try:
            loader = EconomicDataLoader(api_key=FRED_API_KEY_TEST)

            from datetime import datetime, timedelta
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            start_date_str = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d') # 5 years of data

            logging.info("\n--- Fetching all indicator data... ---")
            all_indicator_data = loader.fetch_all_indicator_data(start_date=start_date_str, end_date=end_date_str)

            if all_indicator_data:
                logging.info("\n--- Test Results ---")
                logging.info(f"Successfully fetched data for {len(all_indicator_data)} indicators.")
                for name, df in all_indicator_data.items():
                    if not df.empty:
                        print(f"\n--- Indicator: {name} (Series: {FRED_INDICATOR_SERIES.get(name)}) ---")
                        print(f"Data points: {len(df)}")
                        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
                        print(f"Latest value: {df.iloc[-1]['value']:.2f}")
                        print("Last 5 entries:")
                        print(df.tail())
                    else:
                        print(f"\n--- Indicator: {name} (Series: {FRED_INDICATOR_SERIES.get(name)}) ---")
                        print("No data returned for this indicator in the specified date range.")
            else:
                logging.error("Test failed: No data was fetched for any indicator.")

        except Exception as e:
            logging.error(f"An error occurred during the test run: {e}", exc_info=True)
