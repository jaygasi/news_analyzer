# --- reset_trade_tracking.py ---
import sqlite3
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow importing project modules
# This assumes the script is in the 'event_driven' directory.
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import database
    from logger_config import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is run from the 'event_driven' directory.")
    sys.exit(1)


def reset_all_trades():
    """
    Resets all trades in the database to 'pending' status and clears their
    performance data, allowing price_tracker.py to re-process them.
    """
    print("--- Trade Tracking Reset Utility ---")
    print("\nThis script will perform the following actions on the 'trades' table:")
    print("1. Set 'tracking_status' to 'pending' for ALL rows.")
    print("2. Set ALL performance columns (perf_..._pct and perf_..._timestamp) to NULL.")
    print("\nThis is useful if you want to re-run the price tracker from scratch.")
    print("\nWARNING: This action is irreversible.")

    confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()

    if confirm != "yes":
        print("Operation cancelled by user.")
        return

    try:
        conn = database.get_db_connection()
        cursor = conn.cursor()

        logging.info(
            "Resetting all trades to 'pending' and clearing performance data..."
        )

        sql_update = """
        UPDATE trades
        SET
            tracking_status = 'pending',
            perf_30_min_pct = NULL,
            perf_30_min_timestamp = NULL,
            perf_60_min_pct = NULL,
            perf_60_min_timestamp = NULL,
            perf_240_min_pct = NULL,
            perf_240_min_timestamp = NULL,
            perf_eod_pct = NULL,
            perf_eod_timestamp = NULL;
        """

        cursor.execute(sql_update)
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        logging.info(f"Successfully reset {rows_affected} trades.")
        print(f"\n✅ Success! {rows_affected} trades have been reset.")
        print("You can now run 'python3 price_tracker.py' to re-process them.")

    except (sqlite3.Error, Exception) as e:
        logging.error(f"An error occurred during the reset process: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    # Setup basic logging for the script
    setup_logging()
    reset_all_trades()
