# --- migrate_timestamps_to_utc.py ---
import sqlite3
import logging
import sys
from pathlib import Path
from datetime import datetime
import pytz

# Add project root to sys.path to allow importing project modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    import database
    from logger_config import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is in the 'event_driven' directory.")
    sys.exit(1)


def migrate_timestamps():
    """
    One-time migration script to convert existing naive timestamps in the
    'trades' table to timezone-aware UTC timestamps.
    """
    # --- IMPORTANT ---
    # ASSUMPTION: The original naive timestamps were recorded in this timezone.
    # If your server was in a different timezone, change this value.
    # e.g., 'Europe/London', 'Asia/Tokyo', 'UTC'
    original_tz_str = "US/Eastern"
    # --- IMPORTANT ---

    print("--- Timestamp Migration Utility ---")
    print(
        "This script will convert all naive timestamps in the 'trades' table to UTC."
    )
    print(
        f"It assumes the original timestamps were recorded in the '{original_tz_str}' timezone."
    )
    print("\nWARNING: This action modifies your database and should only be run ONCE.")

    confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Operation cancelled by user.")
        return

    try:
        original_tz = pytz.timezone(original_tz_str)
        utc_tz = pytz.utc
        conn = database.get_db_connection()
        cursor = conn.cursor()

        # Fetch only rows with naive timestamps (that don't contain timezone info)
        cursor.execute(
            "SELECT id, timestamp FROM trades WHERE timestamp NOT LIKE '%+%' AND timestamp NOT LIKE '%Z'"
        )
        trades_to_migrate = cursor.fetchall()

        if not trades_to_migrate:
            print(
                "\n✅ No naive timestamps found to migrate. Your database may already be up to date."
            )
            conn.close()
            return

        logging.info(
            f"Found {len(trades_to_migrate)} trades with naive timestamps to migrate."
        )
        updates_to_perform = []
        for trade in trades_to_migrate:
            trade_id, naive_ts_str = trade["id"], trade["timestamp"]
            try:
                naive_dt = datetime.fromisoformat(naive_ts_str)
                # Localize to the original timezone, then convert to UTC
                aware_dt = original_tz.localize(naive_dt).astimezone(utc_tz)
                utc_ts_str = aware_dt.isoformat()
                updates_to_perform.append((utc_ts_str, trade_id))
            except (ValueError, TypeError) as e:
                logging.error(
                    f"Could not process timestamp '{naive_ts_str}' for trade ID {trade_id}: {e}"
                )

        if updates_to_perform:
            cursor.executemany(
                "UPDATE trades SET timestamp = ? WHERE id = ?",
                updates_to_perform,
            )
            conn.commit()
            updated_count = cursor.rowcount
            logging.info(f"Successfully migrated {updated_count} timestamps.")
            print(f"\n✅ Success! {updated_count} timestamps have been converted to UTC.")
        else:
            print("\nNo valid timestamps were found to update.")

        conn.close()

    except Exception as e:
        logging.critical(
            f"A critical error occurred during migration: {e}", exc_info=True
        )
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    setup_logging()
    migrate_timestamps()
