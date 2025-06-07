import pandas as pd
import csv
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config  # Correctly import Config from your config.py
import pytz
import time


@dataclass
class HistoricalPrice:
    """Historical price data point"""
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    
    def get_price_at_time(self, preferred_type: str = 'close') -> float:
        """Get price based on preference (open, high, low, close)"""
        price_map = {
            'open': self.open_price,
            'high': self.high_price, 
            'low': self.low_price,
            'close': self.close_price
        }
        return price_map.get(preferred_type, self.close_price)


class HistoricalPriceFetcher:
    """Fetch historical prices from FMP API for precise timestamps with configurable intervals"""
    
    def __init__(self, fmp_loader: BaseFMPLoader):
        """Initialize historical price fetcher"""
        self.fmp_loader = fmp_loader
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.timezone('UTC')
        
        # Cache to avoid duplicate API calls
        self.price_cache: Dict[str, List[HistoricalPrice]] = {}
        
        # Log configuration for debugging
        checkpoint_info = Config.get_checkpoint_info()
        intervals = [f"{info['label']}" for info in checkpoint_info] # Use 'label' which is already descriptive
        log_info(f"Historical price fetcher initialized with configurable intervals: {', '.join(intervals)}")
        
    def get_price_at_timestamp(self, ticker: str, target_timestamp: datetime, 
                              interval: str = '5min', preferred_type: str = 'close') -> Optional[Tuple[float, datetime]]:
        """
        Get historical price at specific timestamp along with the actual timestamp of the fetched price.
        
        Args:
            ticker: Stock symbol
            target_timestamp: When you want the price (UTC)
            interval: '1min', '5min', '15min', '30min', '1hour' 
            preferred_type: 'open', 'high', 'low', 'close'
            
        Returns:
            Optional[Tuple[float, datetime]]: (price, actual_timestamp) if found, else None
        """
        # Skip fetching if target is in the future beyond a small buffer
        now_utc = datetime.now(timezone.utc)
        if target_timestamp > now_utc + timedelta(minutes=Config.PRICE_FETCH_FUTURE_BUFFER_MINUTES):
            log_info(f"⏰ Skipping price fetch for {ticker} at {target_timestamp} as it's in the future and likely unavailable.")
            return None

        try:
            # Get historical data for the date
            target_date = target_timestamp.date()
            historical_prices = self._get_intraday_data(ticker, target_date, interval)
            
            if not historical_prices:
                log_warning(f"No intraday historical data available for {ticker} on {target_date} for interval {interval}.")
                return None
            
            # Find the closest price to target timestamp
            closest_price_obj = self._find_closest_price(historical_prices, target_timestamp)
            
            if closest_price_obj:
                price = closest_price_obj.get_price_at_time(preferred_type)
                actual_price_timestamp = closest_price_obj.timestamp
                time_diff = abs((actual_price_timestamp - target_timestamp).total_seconds() / 60)
                
                log_debug(f"Historical price for {ticker} at {target_timestamp}: ${price:.2f} (actual: {actual_price_timestamp.isoformat()}, ±{time_diff:.0f}min)")
                return price, actual_price_timestamp
            else:
                log_warning(f"Could not find historical price for {ticker} near {target_timestamp} within tolerance.")
                return None
                
        except Exception as e:
            log_error(f"Error getting historical price for {ticker} at {target_timestamp}: {e}")
            return None
    
    def get_daily_historical_price(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get daily closing price for a specific date (simpler method)"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            
            # FMP daily historical endpoint
            data = self.fmp_loader.make_request(f"historical-price-full/{ticker}", {
                'from': date_str,
                'to': date_str
            })
            
            if data and 'historical' in data and data['historical']:
                daily_data = data['historical'][0]
                close_price = float(daily_data.get('close', 0))
                
                if close_price > 0:
                    log_debug(f"Daily close for {ticker} on {date_str}: ${close_price:.2f}.")
                    return close_price
            
            log_warning(f"No daily price data for {ticker} on {date_str} from FMP daily endpoint.")
            return None
            
        except Exception as e:
            log_error(f"Error getting daily price for {ticker} on {date_str}: {e}")
            return None

    def _is_during_market_hours(self, dt_est: datetime) -> bool:
        """Check if datetime (EST) is during market hours"""
        # Check if weekend
        if dt_est.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check time range (9:30am-4:00pm EST)
        market_open_time = dt_est.time().replace(hour=9, minute=30)
        market_close_time = dt_est.time().replace(hour=Config.CLOSE_PRICE_HOUR, minute=Config.CLOSE_PRICE_MINUTE) # Use Config for close time
        
        return market_open_time <= dt_est.time() <= market_close_time
    
    def _adjust_for_next_trading_day(self, recommendation_time_est: datetime, minutes_offset: int) -> datetime:
        """
        Adjust target time to the next trading day's market open + offset.
        This function should only be called if the recommendation was outside market hours.
        """
        # Start with the day after the recommendation
        next_trading_day_est = recommendation_time_est + timedelta(days=1)
        
        # Keep advancing until a weekday
        while next_trading_day_est.weekday() >= 5: # Saturday or Sunday
            next_trading_day_est += timedelta(days=1)

        # Set to market open on the next trading day
        market_open_next_day_est = next_trading_day_est.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Apply the minute offset
        adjusted_time_est = market_open_next_day_est + timedelta(minutes=minutes_offset)
        
        return adjusted_time_est.astimezone(self.utc_tz)
    
    def _get_next_trading_day_close(self, rec_time_est: datetime) -> datetime:
        """
        Determine the target close time. If recommendation was during market hours,
        it's today's close. Otherwise, it's the next trading day's close.
        """
        target_close_date_est = rec_time_est
        
        # If after market close or on a weekend, move to the next trading day for close price
        if not self._is_during_market_hours(rec_time_est):
            target_close_date_est = rec_time_est + timedelta(days=1)
            while target_close_date_est.weekday() >= 5: # Saturday or Sunday
                target_close_date_est += timedelta(days=1)
        # If during market hours, target close remains same day
        
        close_time_est = target_close_date_est.replace(
            hour=Config.CLOSE_PRICE_HOUR,
            minute=Config.CLOSE_PRICE_MINUTE,
            second=0,
            microsecond=0
        )
        
        return close_time_est.astimezone(self.utc_tz)
    
    def _get_intraday_data(self, ticker: str, date: datetime.date, interval: str) -> List[HistoricalPrice]:
        """Get intraday historical data for a specific date"""
        
        # Check cache first
        cache_key = f"{ticker}_{date}_{interval}"
        if cache_key in self.price_cache:
            log_debug(f"Using cached data for {cache_key}")
            return self.price_cache[cache_key]
        
        try:
            # FMP intraday endpoint: /historical-chart/{interval}/{symbol}?from=date&to=date
            date_str = date.strftime('%Y-%m-%d')
            
            params = {
                'from': date_str,
                'to': date_str
            }
            
            log_debug(f"Fetching {interval} data for {ticker} on {date_str}.")
            data = self.fmp_loader.make_request(f"historical-chart/{interval}/{ticker}", params)
            
            if not data or not isinstance(data, list):
                log_warning(f"No intraday data returned for {ticker} on {date_str} from FMP API (might be empty or errored).")
                return []
            
            historical_prices = []
            
            for item in data:
                try:
                    # Parse FMP timestamp format
                    timestamp_str = item.get('date', '')
                    timestamp = self._parse_fmp_timestamp(timestamp_str)
                    
                    if timestamp:
                        price_point = HistoricalPrice(
                            timestamp=timestamp,
                            open_price=float(item.get('open', 0)),
                            high_price=float(item.get('high', 0)),
                            low_price=float(item.get('low', 0)),
                            close_price=float(item.get('close', 0)),
                            volume=int(item.get('volume', 0))
                        )
                        historical_prices.append(price_point)
                        
                except (ValueError, TypeError) as e:
                    log_warning(f"Error parsing price data point from FMP for {ticker}: {e}, item: {item}. Skipping.")
                    continue
            
            # Cache the results
            self.price_cache[cache_key] = historical_prices
            
            log_info(f"Fetched {len(historical_prices)} intraday price points for {ticker} on {date_str}.")
            return historical_prices
            
        except Exception as e:
            log_error(f"Error fetching intraday data for {ticker} on {date_str} from FMP: {e}")
            return []
    
    def _parse_fmp_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse FMP timestamp format to datetime"""
        try:
            # FMP returns timestamps like "YYYY-MM-DD HH:MM:SS" (EST)
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # FMP timestamps are in EST, localize and convert to UTC
            est_dt = self.est_tz.localize(dt)
            utc_dt = est_dt.astimezone(self.utc_tz)
            
            return utc_dt
            
        except (ValueError, TypeError) as e:
            log_warning(f"Error parsing FMP timestamp '{timestamp_str}': {e}. Returning None.")
            return None
    
    def _find_closest_price(self, historical_prices: List[HistoricalPrice], 
                           target_timestamp: datetime) -> Optional[HistoricalPrice]:
        """Find the historical price closest to target timestamp with flexible tolerance"""
        if not historical_prices:
            return None
        
        closest_price_obj = None
        min_time_diff = float('inf')
        
        for price_obj in historical_prices:
            time_diff = abs((price_obj.timestamp - target_timestamp).total_seconds())
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_price_obj = price_obj
        
        target_est = target_timestamp.astimezone(self.est_tz)
        
        # Determine tolerance based on target time and if it's a trading day
        tolerance_settings = Config._get_tolerance_settings() # Use the new helper method
        
        if target_est.weekday() >= 5: # Saturday or Sunday
            max_tolerance = tolerance_settings['weekend_hours'] * 3600 # hours to seconds
            log_debug(f"Target is on a weekend ({target_est.strftime('%A')}), using broader tolerance of {max_tolerance/3600:.0f} hours.")
        elif 9.5 <= target_est.hour + target_est.minute/60 <= 16: # During market hours (9:30 AM - 4:00 PM EST)
            max_tolerance = tolerance_settings['market_minutes'] * 60  # minutes to seconds
            log_debug(f"Target is during market hours, using stricter tolerance of {max_tolerance/60:.0f} minutes.")
        else: # Pre-market or after-hours on a weekday
            max_tolerance = tolerance_settings['after_hours_hours'] * 3600  # hours to seconds
            log_debug(f"Target is pre/after-market, using lenient tolerance of {max_tolerance/3600:.0f} hours.")
        
        if closest_price_obj and min_time_diff <= max_tolerance:
            log_debug(f"Found closest price {min_time_diff/60:.1f} minutes from target (tolerance: {max_tolerance/60:.0f}m).")
            return closest_price_obj
        else:
            log_warning(f"Closest price is {min_time_diff/60:.1f} minutes away, exceeds tolerance ({max_tolerance/60:.0f}m). No price found within acceptable range.")
            return None
    

# CSVBackfillUtility is now only used for initial CSV analysis and validation, not continuous backfilling
class CSVBackfillUtility:
    """Utility for initial CSV analysis and validation."""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize utility"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
        log_info("CSV Backfill Utility initialized (for analysis/validation only).")

    def analyze_missing_prices(self) -> Dict[str, Any]:
        """Analyze how many entries are missing price data."""
        try:
            checkpoint_info = Config.get_checkpoint_info()
            analysis = {
                'total_rows': 0,
                'trading_decisions': 0,
                'missing_entry_prices': 0,
                'missing_price_updates': 0,
                'completed_tracking': 0,
                'configuration': {
                    'intervals': [info['label'] for info in checkpoint_info]
                }
            }
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    analysis['total_rows'] += 1
                    decision = row.get('decision', '')
                    entry_price_val = str(row.get('recommendation_price', '')).strip().lower()
                    
                    if decision in ['LONG', 'SHORT']:
                        analysis['trading_decisions'] += 1
                        # Check for missing or effectively missing (0.00) entry prices
                        if not entry_price_val or entry_price_val == 'none':
                            analysis['missing_entry_prices'] += 1
                        else:
                            try:
                                if float(entry_price_val) == 0:
                                    analysis['missing_entry_prices'] += 1
                                elif self._has_all_checkpoint_prices(row):
                                    analysis['completed_tracking'] += 1
                                else:
                                    analysis['missing_price_updates'] += 1
                            except ValueError: # If price is not a valid number
                                analysis['missing_entry_prices'] += 1
            
            log_info(f"📊 Analysis Summary:")
            log_info(f"   Total rows: {analysis['total_rows']}")
            log_info(f"   Trading decisions (LONG/SHORT): {analysis['trading_decisions']}")
            log_info(f"   Missing entry prices (or invalid/zero): {analysis['missing_entry_prices']}")
            log_info(f"   Missing checkpoint updates: {analysis['missing_price_updates']}")
            log_info(f"   Completed tracking: {analysis['completed_tracking']}")
            
            return analysis
            
        except Exception as e:
            log_error(f"Error analyzing CSV: {e}")
            return {}

    def _has_all_checkpoint_prices(self, row: Dict[str, str]) -> bool:
        """Check if row has all configured checkpoint price data (including non-zero values)."""
        rec_price_val = str(row.get('recommendation_price', '')).strip().lower()
        if not rec_price_val or rec_price_val == 'none':
            return False
        try:
            if float(rec_price_val) == 0:
                return False
        except ValueError: # Not a number
            return False

        checkpoint_info = Config.get_checkpoint_info()
        for checkpoint in checkpoint_info:
            field_name = checkpoint['field_prefix']
            # If any checkpoint price is missing, or is an empty/None string, or is '0.00'
            cp_price_val = str(row.get(field_name, '')).strip().lower()
            if not cp_price_val or cp_price_val == '' or cp_price_val == 'none':
                return False
            try:
                if float(cp_price_val) == 0:
                    return False
            except ValueError: # Not a number
                return False
        return True
    
    def validate_existing_prices(self, sample_size: int = 10) -> Dict[str, Any]:
        """Validate existing entry prices against historical data for a sample."""
        log_info(f"🔍 Validating {sample_size} existing entry prices...")
        
        validation_results = {
            'validated': 0,
            'accurate': 0,
            'inaccurate': 0,
            'errors': 0,
            'accuracy_details': [],
            'configuration': {
                'sample_size': sample_size
            }
        }
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                entries_with_prices = []
                for row in reader:
                    rec_price_val = str(row.get('recommendation_price', '')).strip().lower()
                    if (rec_price_val and 
                        rec_price_val != 'none' and # Ensure price exists as string
                        row.get('recommendation_timestamp') and 
                        row.get('decision') in ['LONG', 'SHORT']):
                        try:
                            if float(rec_price_val) != 0: # Ensure price is not 0.00
                                entries_with_prices.append(row)
                        except ValueError:
                            # Skip if not a valid number (will be caught by the overall error count)
                            pass
                
                # Sample random entries
                import random
                sample_entries = random.sample(entries_with_prices, 
                                             min(sample_size, len(entries_with_prices)))
                
                for entry in sample_entries:
                    ticker = entry['ticker']
                    logged_price_str = entry['recommendation_price']
                    timestamp_str = entry['recommendation_timestamp']
                    
                    try:
                        logged_price = float(logged_price_str)
                        target_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if target_timestamp.tzinfo is None:
                            target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                        
                        # Get historical price
                        price_data = self.historical_fetcher.get_price_at_timestamp(
                            ticker, target_timestamp, interval='5min'
                        )
                        
                        validation_results['validated'] += 1
                        
                        if price_data:
                            historical_price, actual_ts = price_data
                            price_diff = abs(logged_price - historical_price)
                            if historical_price == 0:
                                price_diff_pct = float('inf')
                                log_warning(f"Historical price for {ticker} is 0, cannot calculate percentage difference for validation.")
                            else:
                                price_diff_pct = (price_diff / historical_price) * 100
                            
                            if price_diff_pct <= 1.0: # 1% tolerance for accuracy
                                validation_results['accurate'] += 1
                                status = "✅ ACCURATE"
                            else:
                                validation_results['inaccurate'] += 1
                                status = "❌ INACCURATE"
                            
                            validation_results['accuracy_details'].append({
                                'ticker': ticker,
                                'timestamp': timestamp_str,
                                'logged_price': logged_price,
                                'historical_price': historical_price,
                                'difference_pct': price_diff_pct,
                                'status': status
                            })
                            
                            log_info(f"{status} {ticker}: Logged ${logged_price:.2f}, Historical ${historical_price:.2f} ({price_diff_pct:.1f}% diff)")
                        else:
                             log_warning(f"Skipping validation for {ticker} at {timestamp_str}: No historical price found from fetcher.")
                        
                    except ValueError as ve:
                        validation_results['errors'] += 1
                        log_error(f"Error converting price/timestamp for {ticker} (logged: {logged_price_str}, ts: {timestamp_str}): {ve}")
                    except Exception as e:
                        validation_results['errors'] += 1
                        log_error(f"Error validating {ticker}: {e}")
            
            if validation_results['validated'] > 0:
                accuracy_rate = validation_results['accurate'] / validation_results['validated']
                log_info(f"📊 Validation complete: {accuracy_rate:.1%} accuracy rate.")
            else:
                log_info("No entries with prices found for validation.")
            
            return validation_results
            
        except Exception as e:
            log_error(f"Error in price validation: {e}")
            return validation_results


class TradeMonitor:
    """Monitors and updates pending trades in the CSV."""
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
        self.csv_utility = CSVBackfillUtility(csv_path, fmp_loader) # Use for _has_all_checkpoint_prices
        self.est_tz = pytz.timezone('US/Eastern')
        log_info("📈 TradeMonitor initialized for continuous tracking.")

    def run_single_cycle(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Executes one cycle of monitoring and updating pending trades.
        Handles initial entry price backfill and subsequent checkpoint fills.
        """
        results = {
            'total_trades_scanned': 0,
            'pending_trades_before_cycle': 0,
            'trades_updated_this_cycle': 0,
            'trades_completed_this_cycle': 0,
            'trades_still_pending_after_cycle': 0,
            'failed_fetches_this_cycle': 0
        }

        try:
            # Read all rows from CSV
            rows = []
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            if not rows:
                log_warning("TradeMonitor: CSV file is empty, no trades to monitor.")
                return results
            
            # Use CSV headers from the first row of actual data for robust field name handling.
            # This is more dynamic than Config.get_csv_headers() if the CSV evolves.
            headers = list(rows[0].keys()) if rows else [] 
            
            # Identify trades that are not yet 'completed'
            trades_to_process_indices = []
            for i, row in enumerate(rows):
                if row.get('decision') in ['LONG', 'SHORT'] and row.get('tracking_status') != 'completed':
                    trades_to_process_indices.append(i)
            
            results['total_trades_scanned'] = len(rows)
            results['pending_trades_before_cycle'] = len(trades_to_process_indices)
            
            log_info(f"TradeMonitor: Starting cycle. Found {len(trades_to_process_indices)} trades to process (not yet completed).")

            for row_index in trades_to_process_indices:
                row = rows[row_index] # Get the mutable row reference
                ticker = row.get('ticker', '')
                
                # Explicitly get raw values as strings for debugging and robust fallback
                raw_timestamp_col = str(row.get('timestamp', ''))
                raw_rec_timestamp_col = str(row.get('recommendation_timestamp', ''))
                raw_rec_price_col = str(row.get('recommendation_price', ''))
                
                log_debug(f"DEBUG: Processing {ticker} (row {row_index}): "
                          f"timestamp='{raw_timestamp_col}', "
                          f"recommendation_timestamp='{raw_rec_timestamp_col}', "
                          f"recommendation_price='{raw_rec_price_col}'")

                # Determine the primary timestamp to use.
                # If recommendation_timestamp is empty or 'none', use the general 'timestamp' column.
                primary_timestamp_str = raw_rec_timestamp_col.strip()
                if not primary_timestamp_str or primary_timestamp_str.lower() == 'none':
                    primary_timestamp_str = raw_timestamp_col.strip()
                
                log_debug(f"DEBUG: Determined primary_timestamp_str for {ticker}: '{primary_timestamp_str}' (type: {type(primary_timestamp_str)})")
                
                # CRITICAL ASSERTION: Ensure a valid timestamp is available before proceeding.
                assert primary_timestamp_str and primary_timestamp_str.lower() != 'none', \
                    f"CRITICAL ERROR: Primary timestamp is missing or 'None' for {ticker} (row index {row_index}). " \
                    f"Raw CSV values: timestamp='{raw_timestamp_col}', " \
                    f"recommendation_timestamp='{raw_rec_timestamp_col}', " \
                    f"recommendation_price='{raw_rec_price_col}'"

                try:
                    # Parse the primary timestamp
                    rec_timestamp = datetime.fromisoformat(primary_timestamp_str.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                    
                    # Ensure recommendation_timestamp column itself is populated if it was blank (using the primary_timestamp_str)
                    current_rec_timestamp_val = str(row.get('recommendation_timestamp', '')).strip()
                    if not current_rec_timestamp_val or current_rec_timestamp_val.lower() == 'none':
                        if not dry_run:
                            row['recommendation_timestamp'] = primary_timestamp_str # Populate it from fallback
                        log_debug(f"TradeMonitor: Populated recommendation_timestamp for {ticker} from primary timestamp.")
                        results['trades_updated_this_cycle'] += 1 # Counts as an update if this changes the row

                    # --- Stage 1: Ensure recommendation_price is filled (if missing or invalid) ---
                    current_rec_price_val = str(row.get('recommendation_price', '')).strip()
                    should_backfill_rec_price = False
                    if not current_rec_price_val or current_rec_price_val.lower() == 'none':
                        should_backfill_rec_price = True
                    else:
                        try:
                            if float(current_rec_price_val) == 0:
                                should_backfill_rec_price = True
                        except ValueError: # If it's not a valid number (e.g., 'N/A', 'ERROR')
                            should_backfill_rec_price = True
                    
                    if should_backfill_rec_price:
                        log_debug(f"TradeMonitor: Recommendation price missing/invalid for {ticker} ({current_rec_price_val}). Attempting to backfill from {primary_timestamp_str}.")
                        price_data = self.historical_fetcher.get_price_at_timestamp(
                            ticker, rec_timestamp, interval='5min', preferred_type='close'
                        )
                        if price_data:
                            fetched_rec_price, actual_rec_ts = price_data
                            if not dry_run:
                                row['recommendation_price'] = f"{fetched_rec_price:.2f}"
                                # It's critical to update recommendation_timestamp to the *actual* fetched time for consistency
                                row['recommendation_timestamp'] = actual_rec_ts.isoformat() 
                            log_info(f"TradeMonitor: ✅ Backfilled {ticker} recommendation_price: ${fetched_rec_price:.2f} at {actual_rec_ts.isoformat()}.")
                            results['trades_updated_this_cycle'] += 1 # Counts as an update
                        else:
                            log_warning(f"TradeMonitor: ❌ Failed to backfill recommendation_price for {ticker}. Cannot proceed with checkpoints.")
                            results['failed_fetches_this_cycle'] += 1
                            results['trades_still_pending_after_cycle'] += 1
                            continue # Skip to next trade if entry price cannot be obtained

                    # After potential backfill, ensure recommendation_price is valid for calculations
                    final_rec_price_str = str(row.get('recommendation_price')).strip()
                    try:
                        entry_price = float(final_rec_price_str)
                        if entry_price == 0: # Still 0 after backfill attempt
                            log_warning(f"TradeMonitor: Skipping {ticker} as recommendation_price is still zero after backfill attempt. Cannot calculate performance.")
                            results['trades_still_pending_after_cycle'] += 1
                            continue
                    except ValueError: # Still not a valid number after backfill attempt
                        log_warning(f"TradeMonitor: Skipping {ticker} as recommendation_price is still invalid after backfill attempt. Cannot calculate performance.")
                        results['trades_still_pending_after_cycle'] += 1
                        continue

                    # --- Stage 2: Fill subsequent checkpoint prices ---
                    trade_updated_this_iteration_checkpoints = False
                    checkpoint_info = Config.get_checkpoint_info()
                    
                    for cp_def in checkpoint_info:
                        cp_field_prefix = cp_def['field_prefix']
                        cp_price_field = cp_field_prefix
                        cp_timestamp_field = f"{cp_field_prefix}_timestamp"
                        cp_change_pct_field = f"{cp_field_prefix}_change_pct"

                        # Check if this checkpoint is already filled (string check for robustness)
                        current_cp_price_val = str(row.get(cp_price_field, '')).strip()
                        # Also check if price is zero, as this indicates a failed fetch or invalid data
                        if current_cp_price_val and current_cp_price_val.lower() != 'none':
                            try:
                                if float(current_cp_price_val) != 0:
                                    continue # Checkpoint already has a valid non-zero value, skip to next
                            except ValueError: # If it's not a valid number, treat as missing
                                pass # Fall through to attempt fetching
                        
                        # If not filled, calculate its target time and try to fetch
                        target_cp_time_utc: datetime
                        rec_time_est_for_timing = rec_timestamp.astimezone(self.est_tz) # Use for market timing logic
                        
                        if cp_def['minutes'] is not None:
                            target_cp_time_utc = rec_timestamp + timedelta(minutes=cp_def['minutes'])
                            # Adjust for next trading day if recommendation was after-hours for this specific checkpoint
                            if not self.historical_fetcher._is_during_market_hours(rec_time_est_for_timing):
                                target_cp_time_utc = self.historical_fetcher._adjust_for_next_trading_day(rec_time_est_for_timing, cp_def['minutes'])
                        else: # This is the 'close' checkpoint
                            target_cp_time_utc = self.historical_fetcher._get_next_trading_day_close(rec_time_est_for_timing)
                        
                        now_utc = datetime.now(timezone.utc)
                        
                        # Only attempt fetch if the target time has passed (plus a small buffer)
                        if target_cp_time_utc <= now_utc + timedelta(minutes=Config.PRICE_FETCH_FUTURE_BUFFER_MINUTES):
                            log_debug(f"TradeMonitor: Attempting to fetch {cp_def['label']} for {ticker} at {target_cp_time_utc.isoformat()}")
                            price_data = self.historical_fetcher.get_price_at_timestamp(ticker, target_cp_time_utc)
                            
                            if price_data:
                                fetched_price, actual_fetched_ts = price_data
                                if not dry_run:
                                    row[cp_price_field] = f"{fetched_price:.2f}"
                                    row[cp_timestamp_field] = actual_fetched_ts.isoformat()
                                    
                                    try:
                                        change_pct = ((fetched_price - entry_price) / entry_price) * 100
                                        row[cp_change_pct_field] = f"{change_pct:.2f}"
                                    except ZeroDivisionError:
                                        row[cp_change_pct_field] = "0.00"
                                        log_warning(f"TradeMonitor: Zero entry price for {ticker}, cannot calculate change_pct for {cp_def['label']}. Setting to 0.00.")

                                log_info(f"TradeMonitor: ✅ Updated {ticker} {cp_def['label']} to ${fetched_price:.2f}.")
                                trade_updated_this_iteration_checkpoints = True
                            else:
                                results['failed_fetches_this_cycle'] += 1
                                log_warning(f"TradeMonitor: ❌ Failed to fetch {cp_def['label']} for {ticker}. Will retry next cycle.")
                                # If this checkpoint fails, stop processing further checkpoints for this trade this cycle
                                # because subsequent checkpoints would also rely on time passing from this point.
                                break 
                        else:
                            # This checkpoint is in the future, so subsequent ones will be too. Stop for this trade.
                            log_debug(f"TradeMonitor: {cp_def['label']} for {ticker} is still in the future ({target_cp_time_utc.isoformat()}). Skipping remaining checkpoints for this trade this cycle.")
                            break 
                    
                    if trade_updated_this_iteration_checkpoints:
                        results['trades_updated_this_cycle'] += 1

                    # After attempting all relevant checkpoints, check if the trade is now completed
                    if self.csv_utility._has_all_checkpoint_prices(row):
                        if row.get('tracking_status') != 'completed':
                            if not dry_run:
                                row['tracking_status'] = 'completed'
                            results['trades_completed_this_cycle'] += 1
                            log_info(f"TradeMonitor: 🎉 Marked {ticker} as 'completed'.")
                        else:
                            log_debug(f"TradeMonitor: {ticker} was already marked 'completed'.")
                    else:
                        results['trades_still_pending_after_cycle'] += 1
                        log_debug(f"TradeMonitor: {ticker} remains 'pending'.")

                except ValueError as e:
                    log_error(f"TradeMonitor: Data conversion error for {ticker}: {e}. Skipping trade for this cycle.")
                    results['trades_still_pending_after_cycle'] += 1
                except Exception as e:
                    log_error(f"TradeMonitor: Unexpected error processing {ticker}: {e}. Skipping trade for this cycle.")
                    results['trades_still_pending_after_cycle'] += 1

            # This block was correctly de-indented in the previous fix. It should be at this level.
            if not dry_run and (results['trades_updated_this_cycle'] > 0 or results['trades_completed_this_cycle'] > 0):
                backup_path = f"{self.csv_path}.backup_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy2(self.csv_path, backup_path)
                log_info(f"TradeMonitor: Created backup: {backup_path}")
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                log_info(f"TradeMonitor: Saved updates to CSV: {results['trades_updated_this_cycle']} updated, {results['trades_completed_this_cycle']} newly completed.")
            elif dry_run:
                log_info(f"TradeMonitor: DRY RUN: Would update {results['trades_updated_this_cycle']} trades and complete {results['trades_completed_this_cycle']} trades.")
            else:
                log_info("TradeMonitor: No trades were updated or completed in this cycle. CSV not modified.")
            
            return results

        except Exception as e:
            log_error(f"TradeMonitor: Critical error during monitoring cycle: {e}")
            return results


class HistoricalPerformanceAnalyzer:
    """Analyze historical performance of trading decisions with configurable interval support"""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize performance analyzer"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader) 
        
        # Log configuration
        checkpoint_info = Config.get_checkpoint_info()
        log_info(f"Performance analyzer initialized for intervals: {[info['label'] for info in checkpoint_info]}.")
    
    def calculate_performance_metrics(self, days_to_analyze: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics for completed trades with FIXED configurable interval analysis"""
        
        checkpoint_info = Config.get_checkpoint_info()
        
        log_info(f"📊 Analyzing performance for last {days_to_analyze} days across {len(checkpoint_info)} intervals.")
        
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'best_trade_pct': -float('inf'),
            'worst_trade_pct': float('inf'),
            'total_return_pct': 0.0,
            'trade_details': [],
            'checkpoint_performance': {},
            'configuration': {
                'intervals_analyzed': [info['label'] for info in checkpoint_info],
                'days_analyzed': days_to_analyze
            }
        }
        
        # Initialize checkpoint performance tracking
        for checkpoint in checkpoint_info:
            label = checkpoint['label']
            metrics['checkpoint_performance'][label] = {
                'total_trades': 0,
                'winning_trades': 0,
                'total_return_pct': 0.0,
                'best_trade_pct': -float('inf'),
                'worst_trade_pct': float('inf')
            }
        
        try:
            # Use timezone-aware cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_analyze)
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Only analyze trades that are 'completed' for overall performance
                    if row.get('tracking_status') != 'completed':
                        continue 

                    ticker = row['ticker']
                    decision = row['decision']
                    
                    # Validate required fields
                    timestamp_str = row.get('recommendation_timestamp', '')
                    entry_price_str = row.get('recommendation_price', '')

                    if not timestamp_str or not entry_price_str:
                        log_warning(f"Missing recommendation_timestamp or price for 'completed' trade {ticker}. Skipping trade from analysis.")
                        continue

                    try:
                        entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=timezone.utc)
                        
                        if entry_time < cutoff_date:
                            continue # Skip trades older than days_to_analyze
                            
                        entry_price = float(entry_price_str)
                        if entry_price == 0:
                            log_warning(f"Skipping performance calculation for {ticker} due to zero entry price in 'completed' trade.")
                            continue

                    except (ValueError, TypeError) as e:
                        log_warning(f"Invalid entry data for 'completed' trade {ticker} ({entry_price_str}, {timestamp_str}): {e}. Skipping trade from analysis.")
                        continue
                    
                    # Analyze performance at each configured checkpoint
                    for checkpoint in checkpoint_info:
                        field_prefix = checkpoint['field_prefix']
                        label = checkpoint['label']
                        
                        exit_price_str = row.get(field_prefix)
                        if exit_price_str and str(exit_price_str).strip().lower() != 'none':
                            try:
                                exit_price = float(exit_price_str)
                                if entry_price == 0:
                                    return_pct = 0.0
                                    log_warning(f"Zero entry price for {ticker} (checkpoint {label}), cannot calculate % change. Setting to 0.")
                                else:
                                    if decision == 'LONG':
                                        return_pct = ((exit_price - entry_price) / entry_price) * 100
                                    else: # SHORT
                                        return_pct = ((entry_price - exit_price) / entry_price) * 100
                                
                                checkpoint_metrics = metrics['checkpoint_performance'][label]
                                checkpoint_metrics['total_trades'] += 1
                                checkpoint_metrics['total_return_pct'] += return_pct
                                
                                if return_pct > 0:
                                    checkpoint_metrics['winning_trades'] += 1
                                
                                checkpoint_metrics['best_trade_pct'] = max(
                                    checkpoint_metrics['best_trade_pct'], return_pct
                                )
                                checkpoint_metrics['worst_trade_pct'] = min(
                                    checkpoint_metrics['worst_trade_pct'], return_pct
                                )
                                
                            except (ValueError, TypeError) as e:
                                log_warning(f"Error converting exit price for {ticker} at {label} ({exit_price_str}): {e}. Skipping checkpoint for analysis.")

                    # Calculate overall trade performance using price_close
                    final_exit_price_str = row.get('price_close', '0')
                    if final_exit_price_str and str(final_exit_price_str).strip().lower() != 'none':
                        try:
                            final_exit_price = float(final_exit_price_str)
                            if entry_price == 0:
                                return_pct = 0.0
                                log_warning(f"Zero entry price for {ticker} (overall), cannot calculate % change. Setting to 0.")
                            else:
                                if decision == 'LONG':
                                    return_pct = ((final_exit_price - entry_price) / entry_price) * 100
                                else: # SHORT
                                    return_pct = ((entry_price - final_exit_price) / entry_price) * 100
                            
                            metrics['total_trades'] += 1
                            metrics['total_return_pct'] += return_pct
                            
                            if return_pct > 0:
                                metrics['avg_win_pct'] += return_pct
                                metrics['winning_trades'] += 1
                                metrics['best_trade_pct'] = max(metrics['best_trade_pct'], return_pct)
                            else:
                                metrics['avg_loss_pct'] += abs(return_pct)
                                metrics['losing_trades'] += 1
                                metrics['worst_trade_pct'] = min(metrics['worst_trade_pct'], return_pct)
                            
                            metrics['trade_details'].append({
                                'ticker': ticker,
                                'decision': decision,
                                'entry_price': entry_price,
                                'exit_price': final_exit_price,
                                'return_pct': return_pct,
                                'timestamp': timestamp_str,
                                'checkpoint': 'close'
                            })
                            
                        except (ValueError, TypeError, ZeroDivisionError) as e:
                            log_error(f"Error calculating overall performance for {ticker} (final price: {final_exit_price_str}): {e}. This trade might have corrupted 'price_close'. Skipping from overall metrics.")
            
            # Calculate final metrics aggregates
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                if metrics['winning_trades'] > 0:
                    metrics['avg_win_pct'] /= metrics['winning_trades']
                if metrics['losing_trades'] > 0:
                    metrics['avg_loss_pct'] /= metrics['losing_trades']
            else:
                metrics['best_trade_pct'] = 0.0
                metrics['worst_trade_pct'] = 0.0
            
            for label, cp_metrics in metrics['checkpoint_performance'].items():
                if cp_metrics['total_trades'] > 0:
                    cp_metrics['win_rate'] = cp_metrics['winning_trades'] / cp_metrics['total_trades']
                    cp_metrics['avg_return_pct'] = cp_metrics['total_return_pct'] / cp_metrics['total_trades']
                else:
                    cp_metrics['win_rate'] = 0.0
                    cp_metrics['avg_return_pct'] = 0.0
                    cp_metrics['best_trade_pct'] = 0.0
                    cp_metrics['worst_trade_pct'] = 0.0
            
            log_info(f"📈 Performance Analysis Complete:")
            log_info(f"   Total Trades (based on completed status): {metrics['total_trades']}")
            log_info(f"   Win Rate: {metrics['win_rate']:.1%}")
            log_info(f"   Avg Win: {metrics['avg_win_pct']:.2f}%")
            log_info(f"   Avg Loss: {metrics['avg_loss_pct']:.2f}%")
            log_info(f"   Total Return: {metrics['total_return_pct']:.2f}%")
            
            for label, cp_metrics in metrics['checkpoint_performance'].items():
                if cp_metrics['total_trades'] > 0:
                    log_info(f"   {label.upper()} Performance: {cp_metrics['win_rate']:.1%} win rate, {cp_metrics['avg_return_pct']:.2f}% avg return.")
                else:
                     log_info(f"   {label.upper()} Performance: No trades with data for this checkpoint.")
            
            return metrics
            
        except Exception as e:
            log_error(f"Error calculating performance metrics: {e}")
            return metrics


# --- Standalone functions ---

def analyze_trading_performance(csv_path: str, fmp_api_key: str, days: int = 30):
    """Standalone function to analyze trading performance with configurable intervals"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    fmp_loader = BaseFMPLoader(fmp_api_key)
    analyzer = HistoricalPerformanceAnalyzer(csv_path, fmp_loader)
    return analyzer.calculate_performance_metrics(days_to_analyze=days)


def diagnose_missing_prices(csv_path: str, fmp_api_key: str, limit: int = 10):
    """Diagnose why prices are missing with detailed market timing analysis"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    log_info(f"🔍 Diagnosing missing prices for up to {limit} entries")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            missing_entries = []
            for row in reader:
                decision = row.get('decision', '')
                
                # Check for missing or invalid entry prices *or* missing checkpoints
                if (decision in ['LONG', 'SHORT'] and 
                    not backfill_utility._has_all_checkpoint_prices(row)): 
                    
                    missing_entries.append(row)
                    if len(missing_entries) >= limit:
                        break
            
            log_info(f"Found {len(missing_entries)} entries with missing or incomplete price data.")
            
            for i, row in enumerate(missing_entries, 1):
                ticker = row['ticker']
                # Use robust primary timestamp logic here too for diagnosis
                raw_timestamp_col_diag = str(row.get('timestamp', ''))
                raw_rec_timestamp_col_diag = str(row.get('recommendation_timestamp', ''))

                timestamp_str_for_diag = raw_rec_timestamp_col_diag.strip()
                if not timestamp_str_for_diag or timestamp_str_for_diag.lower() == 'none':
                    timestamp_str_for_diag = raw_timestamp_col_diag.strip()
                
                if not timestamp_str_for_diag or timestamp_str_for_diag.lower() == 'none':
                    log_warning(f"Cannot diagnose {ticker} at row {i} due to missing primary timestamp. Skipping.")
                    continue

                try:
                    target_timestamp = datetime.fromisoformat(timestamp_str_for_diag.replace('Z', '+00:00'))
                    if target_timestamp.tzinfo is None:
                        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                    
                    est_tz = pytz.timezone('US/Eastern')
                    target_est = target_timestamp.astimezone(est_tz)
                    
                    log_info(f"\n📊 {i}. {ticker} - {target_est.strftime('%Y-%m-%d %H:%M EST (%A)')}")
                    
                    is_weekend = target_est.weekday() >= 5
                    is_market_hours = (9.5 <= target_est.hour + target_est.minute/60 <= 16) and not is_weekend # Standard market hours
                    
                    log_info(f"   Decision time: {target_est.strftime('%H:%M EST')} on {target_est.strftime('%A')}")
                    log_info(f"   Market status: {'✅ Open' if is_market_hours else '❌ Closed'}")
                    
                    if is_weekend:
                        log_info(f"   Reason for potential issue: Weekend - no trading data available.")
                    elif target_est.hour < 9 or (target_est.hour == 9 and target_est.minute < 30):
                        log_info(f"   Reason for potential issue: Before market open (9:30 AM EST).")
                    elif target_est.hour >= 16:
                        log_info(f"   Reason for potential issue: After market close (4:00 PM EST).")
                    else:
                        log_info(f"   Reason for potential issue: Recommendation appears to be during market hours, but data might be sparse or outside fetcher's tolerance.")
                    
                    checkpoint_info = Config.get_checkpoint_info()
                    for checkpoint in checkpoint_info:
                        label = checkpoint['label']
                        minutes = checkpoint.get('minutes')
                        
                        target_check_time_utc: datetime
                        if minutes is not None:
                            target_check_time_utc = target_timestamp + timedelta(minutes=minutes)
                            # Re-check market hours for specific checkpoint calculation if original was off-hours
                            # This re-uses the logic from _adjust_for_next_trading_day but only for diagnosis display
                            if not backfill_utility.historical_fetcher._is_during_market_hours(target_est):
                                target_check_time_utc_adj = backfill_utility.historical_fetcher._adjust_for_next_trading_day(target_est, minutes)
                                log_info(f"   Calculated {label} target (adjusted): {target_check_time_utc_adj.astimezone(est_tz).strftime('%H:%M EST on %A')}")
                            else:
                                log_info(f"   Calculated {label} target: {target_check_time_utc.astimezone(est_tz).strftime('%H:%M EST on %A')}")
                        else: # This is the 'close' checkpoint
                            target_check_time_utc = backfill_utility.historical_fetcher._get_next_trading_day_close(target_est)
                            log_info(f"   Calculated {label} target: {target_check_time_utc.astimezone(est_tz).strftime('%H:%M EST on %A')}")
                        
                        if target_check_time_utc > datetime.now(timezone.utc) + timedelta(minutes=Config.PRICE_FETCH_FUTURE_BUFFER_MINUTES):
                             log_info(f"   Note: {label} target time is in the future ({Config.PRICE_FETCH_FUTURE_BUFFER_MINUTES}min buffer) and cannot be fetched yet.")
                        else:
                             log_info(f"   Note: {label} target time has passed (or is within buffer) and should be fetchable.")
                        
                except Exception as e:
                    log_error(f"Error analyzing {ticker} for diagnosis: {e}")
            
    except Exception as e:
        log_error(f"Error in diagnosis process: {e}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to sys.path to allow importing 'config' and 'data_loaders'
    # This assumes 'config.py' is in the parent directory and 'data_loaders' is a sibling directory.
    sys.path.append(str(Path(__file__).parent.parent)) 
    
    try:
        # Import Config and BaseFMPLoader AFTER setting sys.path
        from config import Config
        from data_loaders.base_fmp_loader import BaseFMPLoader
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure 'config.py', 'data_loaders/base_fmp_loader.py', and 'utils/simple_logger.py' are accessible in your PYTHONPATH.")
        sys.exit(1)

    if not Config.FMP_API_KEY:
        print("FMP_API_KEY is required in config.py or .env file. Please set it before running.")
        sys.exit(1)
    
    csv_path = "output/trading_decisions.csv" # Ensure this path is correct
    
    print("🔄 Enhanced Historical Price Utilities with Integrated & Continuous Monitoring")
    
    config_summary = Config.get_config_summary()
    print(f"Current configuration summary:")
    for key, value in config_summary.items():
        if key == 'enabled_services':
            print(f"   Enabled Services: {', '.join(value)}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
            
    tolerance_info = Config._get_tolerance_settings()
    print(f"Price Fetch Tolerance: Market hours = {tolerance_info['market_minutes']}min, After hours = {tolerance_info['after_hours_hours']} hours, Weekend = {tolerance_info['weekend_hours']} hours")
    print(f"Trade Monitor will run every {Config.TRADE_MONITOR_INTERVAL_MINUTES} minutes.")
    print()
    
    # Initialize components
    fmp_loader = BaseFMPLoader(Config.FMP_API_KEY)
    csv_utility = CSVBackfillUtility(csv_path, fmp_loader)
    trade_monitor = TradeMonitor(csv_path, fmp_loader)
    performance_analyzer = HistoricalPerformanceAnalyzer(csv_path, fmp_loader)

    # --- Initial Diagnostics and Validation (One-time at startup) ---
    try:
        print("1. Initial Diagnostics & Validation (at startup):")
        
        print("   a. Analyzing overall data completeness (entry and checkpoints)...")
        initial_analysis_summary = csv_utility.analyze_missing_prices()
        print(f"      Initial State: {initial_analysis_summary.get('completed_tracking', 0)} completed, {initial_analysis_summary.get('missing_entry_prices', 0)} missing entry, {initial_analysis_summary.get('missing_price_updates', 0)} missing checkpoints.")
        
        print("\n   b. Diagnosing issues for a sample of incomplete trades (limit 3)...")
        diagnose_missing_prices(csv_path, Config.FMP_API_KEY, limit=3)
        
        print("\n   c. Validating existing entry prices for accuracy (sample size 5)...")
        validation = csv_utility.validate_existing_prices(sample_size=5)
        if validation and validation.get('validated', 0) > 0:
            accuracy = validation.get('accurate', 0) / validation.get('validated', 1)
            print(f"      Entry Price Accuracy: {accuracy:.1%} ({validation.get('accurate', 0)}/{validation.get('validated', 0)} validated).")
        else:
            print("      No entries with prices found for validation or an error occurred during validation.")
        print("\n--- Initial setup complete. Proceeding to continuous monitoring. ---")

    except Exception as e:
        print(f"❌ Error during initial setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Continuous Monitoring Loop ---
    print("\n🚀 Starting continuous trade monitoring. Press Ctrl+C to stop.")
    last_performance_analysis_time = datetime.min # Initialize to run immediately

    try:
        while True:
            current_time = datetime.now()
            log_info(f"\n--- Trade Monitoring Cycle Start: {current_time.isoformat()} ---")
            
            # Run a single monitoring and update cycle
            monitor_cycle_results = trade_monitor.run_single_cycle(dry_run=False) # Live updates
            
            log_info(f"Cycle Summary:")
            log_info(f"  Scanned {monitor_cycle_results['total_trades_scanned']} total trades. {monitor_cycle_results['pending_trades_before_cycle']} were pending at start.")
            log_info(f"  Updated {monitor_cycle_results['trades_updated_this_cycle']} trades (price/timestamp fills).")
            log_info(f"  Newly completed {monitor_cycle_results['trades_completed_this_cycle']} trades.")
            log_info(f"  {monitor_cycle_results['trades_still_pending_after_cycle']} trades still pending after this cycle.")
            log_info(f"  {monitor_cycle_results['failed_fetches_this_cycle']} price fetches failed (will retry).")

            # Periodically run performance analysis (e.g., once every X minutes/hours)
            # Use TRADE_MONITOR_INTERVAL_MINUTES for performance analysis frequency as well, for simplicity
            if (current_time - last_performance_analysis_time).total_seconds() >= (60 * Config.TRADE_MONITOR_INTERVAL_MINUTES):
                log_info("\n--- Running Performance Analysis ---")
                performance = performance_analyzer.calculate_performance_metrics(days_to_analyze=30)
                if performance and performance.get('total_trades', 0) > 0:
                    log_info(f"  Total completed trades analyzed: {performance['total_trades']}")
                    log_info(f"  Overall Win rate: {performance['win_rate']:.1%}")
                    log_info(f"  Overall Avg return per trade: {performance['total_return_pct']/performance['total_trades']:.2f}%")
                else:
                    log_info("  No completed trades found for performance analysis within the last 30 days.")
                last_performance_analysis_time = current_time # Update time only if analysis ran

            log_info(f"--- Cycle End. Sleeping for {Config.TRADE_MONITOR_INTERVAL_MINUTES} minutes... ---")
            time.sleep(Config.TRADE_MONITOR_INTERVAL_MINUTES * 60) # Sleep before next cycle

    except KeyboardInterrupt:
        print("\n👋 Trade monitoring stopped by user.")
    except Exception as e:
        print(f"❌ An unexpected error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Historical price analysis process finished.")