import logging
from datetime import datetime, timedelta
import os
import pandas as pd

# Since this is a standalone script, we need to handle imports carefully
# and ensure the environment is set up.
try:
    from economic_data_loader import EconomicDataLoader
    import database
    from config import FRED_INDICATOR_SERIES
except ImportError as e:
    logging.error(f"Failed to import necessary modules. Make sure you are running from the correct directory and venv is active: {e}")
    exit(1)

def backfill_data():
    """
    Fetches historical economic data for a long period and populates the database.
    This is intended as a one-time utility script to run during setup.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define the historical period for the backfill
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10) # Fetch 10 years of historical data
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    logging.info(f"Starting backfill for economic data from {start_date_str} to {end_date_str}.")

    try:
        # 1. Initialize database and data loader
        logging.info("Initializing database to ensure tables exist...")
        database.init_db()
        loader = EconomicDataLoader()

        # 2. Fetch all historical data from the API
        logging.info("Fetching data for all configured economic indicators...")
        all_indicator_data = loader.fetch_all_indicator_data(start_date=start_date_str, end_date=end_date_str)

        if not all_indicator_data:
            logging.warning("No economic data was fetched from the API. Exiting backfill process.")
            return

        # 3. Process and prepare data for batch logging
        logging.info("Processing and preparing data for database insertion...")
        data_to_log = []
        for indicator_name, df in all_indicator_data.items():
            if df is not None and not df.empty:
                # Resample to daily frequency and forward-fill missing values.
                # This ensures that for any given day, we have the most recently reported value.
                # For example, GDP is quarterly, but we want a value for it every day.
                df_resampled = df.resample('D').ffill().dropna()
                for date, row in df_resampled.iterrows():
                    data_to_log.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "indicator_name": indicator_name,
                        "value": row['value']
                    })

        if not data_to_log:
            logging.warning("Data was fetched but resulted in an empty list to log. This might happen if the fetched data had no values.")
            return

        # 4. Log the prepared data to the database in a single transaction
        logging.info(f"Logging {len(data_to_log)} processed data points to the database...")
        conn = database.get_db_connection()
        try:
            database.log_economic_data_batch(data_to_log, conn)
            conn.commit()
            logging.info(f"Successfully backfilled and logged data for {len(all_indicator_data)} indicators.")
        except Exception as e:
            conn.rollback()
            logging.error(f"Failed to commit backfilled data to the database: {e}", exc_info=True)
        finally:
            conn.close()

    except Exception as e:
        logging.error(f"A critical error occurred during the backfill process: {e}", exc_info=True)


if __name__ == '__main__':
    # This block allows running the script directly from the command line.
    # It requires a .env file with FRED_API_KEY in the same directory.
    print("--- Running Economic Data Backfill Utility ---")

    from dotenv import load_dotenv
    # Assuming the script is in news_analyzer/event_driven, .env should be there too
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print("-> .env file found and loaded.")
    else:
        print("-> WARNING: .env file not found. The script will rely on environment variables being set manually.")

    if not os.getenv("FRED_API_KEY"):
         print("-> FATAL: FRED_API_KEY is not set. Cannot run the backfill script. Please add it to your .env file or environment.")
    else:
        print("-> FRED_API_KEY found. Starting backfill process...")
        backfill_data()
        print("--- Backfill Process Finished ---")
