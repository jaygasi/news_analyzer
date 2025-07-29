import pandas as pd
from ftplib import FTP
import os
from datetime import datetime
import config  # FTP configuration

# --- Path Configuration ---
# For Google Colab, set this to your Google Drive path. e.g., '/content/drive/My Drive/MarginAnalysis/'
GDRIVE_PATH = ""
MARGIN_DATA_CSV = os.path.join(GDRIVE_PATH, "margin_rates_history.csv")
LAST_PROCESSED_FILE_MARKER = os.path.join(GDRIVE_PATH, "last_processed_file.txt")


def get_latest_file_from_server(ftp):
    """Finds the name of the most recent usa.txt file on the server."""
    try:
        ftp.cwd("usa")
    except Exception:
        print("INFO: Could not change to 'usa' directory. Looking in root directory.")

    files = ftp.nlst()
    stock_files = sorted(
        [f for f in files if f.startswith("usa") and f.endswith(".txt")], reverse=True
    )

    if not stock_files:
        return None
    return stock_files[0]


def process_rate_changes(filepath):
    """Parses the downloaded file and appends only new rate changes to the master CSV."""
    print(f"\n--- Processing Rate Changes from: {os.path.basename(filepath)} ---")

    try:
        # Define names for ALL columns, including the last empty one caused by the trailing '|'
        column_names = [
            "SYM",
            "CUR",
            "NAME",
            "CON",
            "ISIN",
            "REBATERATE",
            "FEERATE",
            "AVAILABLE",
            "FIGI",
            "EMPTY_COL",
        ]

        # Load all columns from the file
        latest_df = pd.read_csv(
            filepath,
            sep="|",
            skiprows=2,
            header=None,
            names=column_names,
            on_bad_lines="skip",  # Safely skip any malformed rows
        )

        # Drop the empty column that results from the trailing '|'
        if "EMPTY_COL" in latest_df.columns:
            latest_df = latest_df.drop(columns=["EMPTY_COL"])

        # --- Data Cleaning Step ---
        # Drop rows where FEERATE is missing (NaN) to ensure data integrity before any processing.
        original_count = len(latest_df)
        latest_df.dropna(subset=["FEERATE"], inplace=True)
        cleaned_count = len(latest_df)
        if original_count > cleaned_count:
            print(
                f"DEBUG: Dropped {original_count - cleaned_count} rows with missing FEERATE values."
            )
        # --- End of Cleaning Step ---

        print(
            f"DEBUG: Successfully loaded {len(latest_df)} valid records from the file."
        )

        # Load existing history to check for changes
        if os.path.exists(MARGIN_DATA_CSV):
            print(
                f"DEBUG: History file found at '{MARGIN_DATA_CSV}'. Comparing for changes."
            )
            history_df = pd.read_csv(MARGIN_DATA_CSV, parse_dates=["timestamp"])

            # Get the most recent entry for each stock to know its last fee rate
            last_known_rates = history_df.loc[
                history_df.groupby("SYM")["timestamp"].idxmax()
            ].set_index("SYM")
            print(f"DEBUG: Loaded {len(last_known_rates)} unique symbols from history.")

            # Merge new data with the last known rates
            merged_df = latest_df.set_index("SYM").join(
                last_known_rates[["FEERATE"]], rsuffix="_last"
            )

            # Find rows where the fee rate has changed, or for stocks we've never seen before
            new_changes_df = merged_df[
                (merged_df["FEERATE"] != merged_df["FEERATE_last"])
                | (merged_df["FEERATE_last"].isna())
            ].copy()

        else:  # No history exists yet, so everything is a new change
            print(
                "DEBUG: No history file found. Treating all valid records as new changes."
            )
            new_changes_df = latest_df.set_index("SYM")

        if new_changes_df.empty:
            print(
                "\nRESULT: No new fee rate changes detected. Your history file is already up-to-date."
            )
            return

        print(f"\nRESULT: Found {len(new_changes_df)} new/changed fee rates to log.")

        # --- FIX: Prepare the new data to be appended, keeping ALL original columns ---
        new_changes_to_log = new_changes_df.reset_index()
        # Drop the helper column used for comparison if it exists
        if "FEERATE_last" in new_changes_to_log.columns:
            new_changes_to_log = new_changes_to_log.drop(columns=["FEERATE_last"])

        new_changes_to_log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add a blank column to be populated by the analyzer
        new_changes_to_log["price_5min_after"] = None

        # Append the new changes (with all columns) to the master CSV
        new_changes_to_log.to_csv(
            MARGIN_DATA_CSV,
            mode="a",
            header=not os.path.exists(MARGIN_DATA_CSV),
            index=False,
        )
        print(f"Successfully appended new changes to {MARGIN_DATA_CSV}")

    except Exception as e:
        print(f"ERROR: An error occurred while processing the file: {e}")


if __name__ == "__main__":
    print(f"Data collector running at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        with FTP(config.FTP_HOST, config.FTP_USER, config.FTP_PASS) as ftp:
            print("Successfully connected to FTP server.")
            latest_server_file = get_latest_file_from_server(ftp)

            if not latest_server_file:
                print("ERROR: No data file found on server.")
                exit()

            # Check if we've already processed this exact server file
            last_processed = ""
            if os.path.exists(LAST_PROCESSED_FILE_MARKER):
                with open(LAST_PROCESSED_FILE_MARKER, "r") as f:
                    last_processed = f.read().strip()

            if latest_server_file == last_processed:
                print(
                    f"INFO: Latest file on server ('{latest_server_file}') has already been processed. Nothing to do."
                )
                exit()

            # Download the new file
            print(f"INFO: Found a new server file to process: '{latest_server_file}'")
            temp_local_file = f"downloaded_{latest_server_file}"
            with open(temp_local_file, "wb") as f:
                ftp.retrbinary("RETR " + latest_server_file, f.write)

            # Process the new file for changes
            process_rate_changes(temp_local_file)

            # Update the marker file to remember we've processed this file
            with open(LAST_PROCESSED_FILE_MARKER, "w") as f:
                f.write(latest_server_file)
            print(f"\nINFO: Marked '{latest_server_file}' as processed.")

            # Clean up the downloaded file
            os.remove(temp_local_file)

    except Exception as e:
        print(f"ERROR: An unhandled error occurred: {e}")
