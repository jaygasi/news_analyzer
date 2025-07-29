from datetime import datetime, timedelta, date, timezone
from typing import Dict, List, Optional, Any, Tuple
import pytz
import asyncio
import logging
import pandas as pd
from collections import defaultdict
import pandas_market_calendars as mcal

from data_loader import FMPDataLoader
from config import FMP_API_KEY, PERFORMANCE_CHECKPOINTS
import database
from logger_config import setup_logging


def get_effective_trading_day(trade_log_datetime_utc: datetime) -> Optional[date]:
    """
    Determines the effective trading day for performance evaluation.
    The effective day is the first valid trading day whose market close is *after* the trade was logged.
    """
    if trade_log_datetime_utc.tzinfo is None:
        logging.error("Received a naive datetime. Cannot determine effective trading day.")
        return None

    nyse = mcal.get_calendar("NYSE")

    # Start searching from the day the trade was logged (in UTC) and look forward 7 days.
    search_start_date = trade_log_datetime_utc.date()
    search_end_date = search_start_date + timedelta(days=7)

    # Get the market schedule for the potential trading days.
    schedule = nyse.schedule(start_date=search_start_date, end_date=search_end_date)

    # Find the first day in the schedule where the market close is after the trade time.
    for index, day_schedule in schedule.iterrows():
        market_close_utc = day_schedule["market_close"]  # This is a timezone-aware pandas.Timestamp
        if trade_log_datetime_utc <= market_close_utc:
            return market_close_utc.date()  # Return the date part of that trading day.

    logging.error(f"Could not find a valid trading day for trade logged at {trade_log_datetime_utc}.")
    return None


def get_checkpoint_reference_time(
    trade_log_dt_utc: datetime,
    market_open_utc: datetime,
    market_close_utc: datetime,
) -> datetime:
    """
    Determines the reference time for performance checkpoints based on README logic.
    """
    if market_open_utc <= trade_log_dt_utc <= market_close_utc:
        # Scenario 1: Trade logged during market hours. Use the exact log time.
        return trade_log_dt_utc
    else:
        # Scenarios 2, 3, 4, 5, 6: Trade logged outside market hours. Use market open.
        return market_open_utc


def find_price_at_checkpoint(
    target_time: datetime,
    intraday_df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Finds the price at a specific target time from intraday data.
    Uses pandas' `asof` for robust time-based lookups.
    """
    if intraday_df.empty:
        return None, None

    try:
        # 'asof' finds the last row at or before the target time.
        match = intraday_df.asof(target_time)
        if pd.notna(match["close"]):
            return float(match["close"]), match.name.isoformat()
    except Exception as e:
        logging.error(f"Error finding price at checkpoint for time {target_time}: {e}")

    return None, None


def calculate_intraday_performance(
    entry_price: float,
    reference_time: datetime,
    intraday_df: pd.DataFrame,
    market_close_utc: datetime,
) -> Optional[Dict[str, Any]]:
    """Calculates performance at all configured intraday checkpoints."""
    if entry_price <= 0 or intraday_df.empty:
        return None

    performance_data = {}
    for name, minutes in PERFORMANCE_CHECKPOINTS.items():
        checkpoint_time = reference_time + timedelta(minutes=minutes)
        price, timestamp = find_price_at_checkpoint(checkpoint_time, intraday_df)

        if price is not None:
            perf_pct = ((price - entry_price) / entry_price) * 100
            performance_data[f"perf_{name}_pct"] = perf_pct
            performance_data[f"perf_{name}_timestamp"] = timestamp
        else:
            performance_data[f"perf_{name}_pct"] = None
            performance_data[f"perf_{name}_timestamp"] = None

    # Calculate EOD performance
    eod_price, eod_timestamp = find_price_at_checkpoint(market_close_utc, intraday_df)
    if eod_price is not None:
        eod_perf_pct = ((eod_price - entry_price) / entry_price) * 100
        performance_data["perf_eod_pct"] = eod_perf_pct
        performance_data["perf_eod_timestamp"] = eod_timestamp
    else:
        performance_data["perf_eod_pct"] = None
        performance_data["perf_eod_timestamp"] = None

    return performance_data


async def _process_trade_group_intraday(
    ticker: str,
    effective_date_str: str,
    trades_in_group: List[Dict],
    fmp_loader: FMPDataLoader,
    market_schedule: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Fetches 1-minute intraday data for a group and calculates performance
    at multiple checkpoints (30min, 60min, 240min, EOD).
    """
    logging.info(
        f"  -> Processing group: Ticker {ticker}, Date {effective_date_str}, {len(trades_in_group)} trades."
    )
    updates_to_perform = []

    intraday_data_list = await fmp_loader.get_intraday_historical_data(
        ticker, date_str=effective_date_str
    )

    if not intraday_data_list:
        logging.warning(
            f"Could not retrieve intraday data for {ticker} on {effective_date_str}. Marking {len(trades_in_group)} trades as failed."
        )
        for trade in trades_in_group:
            updates_to_perform.append(
                {"trade_id": trade.get("id"), "status": "failed", "performance_data": None}
            )
        return updates_to_perform

    # Convert to DataFrame for efficient lookups
    try:
        intraday_df = pd.DataFrame(intraday_data_list)
        # The FMP API returns naive timestamps in US/Eastern time.
        # We must first localize them to 'US/Eastern' and then convert to 'UTC' for consistency.
        intraday_df["date"] = (
            pd.to_datetime(intraday_df["date"])
            .dt.tz_localize("US/Eastern")
            .dt.tz_convert("UTC")
        )
        intraday_df = intraday_df.set_index("date").sort_index()
    except Exception as e:
        logging.error(
            f"Error processing intraday data for {ticker} on {effective_date_str}: {e}"
        )
        for trade in trades_in_group:
            updates_to_perform.append(
                {"trade_id": trade.get("id"), "status": "failed", "performance_data": None}
            )
        return updates_to_perform

    # Get market open/close for the effective day
    market_open_utc = market_schedule.iloc[0]["market_open"].to_pydatetime()
    market_close_utc = market_schedule.iloc[0]["market_close"].to_pydatetime()

    for trade in trades_in_group:
        trade_id = trade.get("id")
        entry_price = trade.get("entry_price")
        trade_log_dt_utc = datetime.fromisoformat(trade["timestamp"])

        reference_time = get_checkpoint_reference_time(
            trade_log_dt_utc, market_open_utc, market_close_utc
        )

        performance_data = calculate_intraday_performance(
            entry_price, reference_time, intraday_df, market_close_utc
        )

        if performance_data:
            status = "completed"
            logging.info(
                f"  Successfully calculated intraday performance for trade ID {trade_id}."
            )
        else:
            status = "failed"
            logging.warning(
                f"  Failed to calculate performance for trade ID {trade_id} (likely zero entry price or missing intraday data)."
            )

        updates_to_perform.append(
            {
                "trade_id": trade_id,
                "status": status,
                "performance_data": performance_data,
            }
        )

    return updates_to_perform


async def run_price_tracker():
    """
    Main function to track intraday and end-of-day performance of trades.
    """
    logging.info("--- Starting Price Tracker / Performance Backfill ---")
    untracked_trades = database.get_untracked_trades(status="pending")
    if not untracked_trades:
        logging.info("No new trades to track. System is up to date.")
        return

    logging.info(f"Found {len(untracked_trades)} untracked trades to process.")

    failed_trade_updates = []
    trades_by_group = defaultdict(list)
    nyse_calendar = mcal.get_calendar("NYSE")
    today_utc = datetime.now(timezone.utc)

    for trade in untracked_trades:
        trade_id = trade.get("id")
        try:
            # Ensure timestamp is timezone-aware
            trade_log_datetime_utc = datetime.fromisoformat(trade["timestamp"])
            effective_trade_date = get_effective_trading_day(trade_log_datetime_utc)

            if not effective_trade_date:
                failed_trade_updates.append({"trade_id": trade_id, "status": "failed"})
                continue

            # --- Logic to skip trades for the current, unclosed market day ---
            if effective_trade_date == today_utc.date():
                market_schedule = nyse_calendar.schedule(
                    start_date=effective_trade_date, end_date=effective_trade_date
                )
                if not market_schedule.empty:
                    market_close_utc = market_schedule.iloc[0][
                        "market_close"
                    ].to_pydatetime()
                    if today_utc < market_close_utc:
                        logging.info(
                            f"Skipping trade ID {trade_id}: its effective day is today, but the market is not yet closed."
                        )
                        continue

            group_key = (trade["ticker"], effective_trade_date.strftime("%Y-%m-%d"))
            trades_by_group[group_key].append(trade)

        except (ValueError, KeyError) as e:
            logging.error(f"Could not parse timestamp for trade ID {trade_id}. Skipping.")
            failed_trade_updates.append({"trade_id": trade_id, "status": "failed"})

    if not trades_by_group:
        if failed_trade_updates:
            logging.info(f"Updating status for {len(failed_trade_updates)} trades that failed pre-processing...")
            database.update_trades_status_batch(failed_trade_updates)
        logging.info(
            "No trades are ready for tracking at this time (all may be for the current, open market day)."
        )
        return

    logging.info(f"Grouped trades into {len(trades_by_group)} unique API calls.")

    fmp_loader = FMPDataLoader(api_key=FMP_API_KEY)

    tasks = [
        _process_trade_group_intraday(
            ticker=group_key[0],
            effective_date_str=group_key[1],
            trades_in_group=trades,
            fmp_loader=fmp_loader,
            # Pass the market schedule for the specific day to the processing function
            market_schedule=nyse_calendar.schedule(
                start_date=group_key[1], end_date=group_key[1]
            ),
        )
        for group_key, trades in trades_by_group.items()
    ]

    all_performance_updates = []
    all_status_updates = []

    try:
        logging.info(f"Calculating performance for {len(tasks)} groups concurrently...")
        results_from_all_groups = await asyncio.gather(*tasks)

        for group_results in results_from_all_groups:
            for update in group_results:
                if update.get("performance_data"):
                    # Add trade_id to performance data for batch update
                    update["performance_data"]["trade_id"] = update["trade_id"]
                    all_performance_updates.append(update["performance_data"])
                all_status_updates.append(update)

        logging.info("All calculations complete. Writing results to the database...")
        with database.get_db_connection() as conn:
            if all_performance_updates:
                database.update_trades_performance_batch(all_performance_updates, conn)
            # Batch update statuses for all processed trades
            all_status_updates.extend(failed_trade_updates)
            if all_status_updates:
                database.update_trades_status_batch(all_status_updates, conn)
            conn.commit()
            logging.info("Database updates committed successfully.")

    except Exception as e:
        logging.critical(f"A critical error occurred during price tracking: {e}", exc_info=True)
    finally:
        if fmp_loader:
            await fmp_loader.close()
    logging.info("--- Price Tracker Finished ---")


def main():
    setup_logging()
    database.init_db()
    try:
        asyncio.run(run_price_tracker())
    except KeyboardInterrupt:
        logging.info("Price tracker run interrupted by user.")
    except Exception as e:
        logging.critical(f"A critical error occurred in the price tracker: {e}", exc_info=True)


if __name__ == "__main__":
    main()
