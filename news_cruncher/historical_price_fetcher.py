"""
Historical price fetcher and backfill utilities with configurable interval support
Python 3.13.3 compatible
"""
import pandas as pd
import csv
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config
import pytz


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
        intervals = [f"{info['label']} ({info.get('minutes', 'close')})" for info in checkpoint_info]
        log_info(f"Historical price fetcher initialized with configurable intervals: {', '.join(intervals)}")
        
    def get_price_at_timestamp(self, ticker: str, target_timestamp: datetime, 
                              interval: str = '5min', preferred_type: str = 'close') -> Optional[float]:
        """
        Get historical price at specific timestamp
        
        Args:
            ticker: Stock symbol
            target_timestamp: When you want the price (UTC)
            interval: '1min', '5min', '15min', '30min', '1hour' 
            preferred_type: 'open', 'high', 'low', 'close'
        """
        try:
            # Get historical data for the date
            target_date = target_timestamp.date()
            historical_prices = self._get_intraday_data(ticker, target_date, interval)
            
            if not historical_prices:
                log_warning(f"No historical data available for {ticker} on {target_date}")
                return None
            
            # Find the closest price to target timestamp
            closest_price = self._find_closest_price(historical_prices, target_timestamp)
            
            if closest_price:
                price = closest_price.get_price_at_time(preferred_type)
                time_diff = abs((closest_price.timestamp - target_timestamp).total_seconds() / 60)
                
                log_debug(f"Historical price for {ticker} at {target_timestamp}: ${price:.2f} (±{time_diff:.0f}min)")
                return price
            else:
                log_warning(f"Could not find historical price for {ticker} near {target_timestamp}")
                return None
                
        except Exception as e:
            log_error(f"Error getting historical price for {ticker}: {e}")
            return None
    
    def get_checkpoint_prices(self, ticker: str, recommendation_timestamp: datetime) -> Dict[str, Optional[float]]:
        """
        Get historical prices at all configured checkpoint intervals with smart market timing
        
        Returns dict with keys like 'price_checkpoint1', 'price_checkpoint2', 'price_close' based on configuration
        """
        results = {}
        checkpoint_info = Config.get_checkpoint_info()
        
        # Check if recommendation was made during market hours
        rec_time_est = recommendation_timestamp.astimezone(self.est_tz)
        is_market_hours = self._is_during_market_hours(rec_time_est)
        
        if not is_market_hours:
            log_info(f"📅 {ticker} recommendation made outside market hours ({rec_time_est.strftime('%H:%M EST')}), adjusting strategy")
        
        for checkpoint in checkpoint_info:
            label = checkpoint['label']
            field_prefix = checkpoint['field_prefix']
            minutes = checkpoint.get('minutes')
            
            try:
                if minutes is not None:
                    # Regular interval checkpoint
                    target_time = recommendation_timestamp + timedelta(minutes=minutes)
                    
                    # If recommendation was after hours, try to get next trading day prices
                    if not is_market_hours:
                        target_time = self._adjust_for_next_trading_day(target_time, minutes)
                        log_debug(f"Adjusted {label} target time to next trading day: {target_time}")
                    
                    price = self.get_price_at_timestamp(ticker, target_time)
                    results[field_prefix] = price
                    
                    if price:
                        log_info(f"✅ Historical {label} price for {ticker}: ${price:.2f}")
                    else:
                        log_warning(f"❌ Could not get historical {label} price for {ticker}")
                        
                else:
                    # Close time checkpoint
                    if is_market_hours:
                        # Same day close
                        close_time = rec_time_est.replace(
                            hour=Config.CLOSE_PRICE_HOUR,
                            minute=Config.CLOSE_PRICE_MINUTE,
                            second=0,
                            microsecond=0
                        ).astimezone(self.utc_tz)
                    else:
                        # Next trading day close
                        close_time = self._get_next_trading_day_close(rec_time_est)
                        log_debug(f"Using next trading day close for {ticker}: {close_time}")
                    
                    price = self.get_price_at_timestamp(ticker, close_time)
                    results[field_prefix] = price
                    
                    if price:
                        log_info(f"✅ Historical {label} price for {ticker}: ${price:.2f}")
                    else:
                        log_warning(f"❌ Could not get historical {label} price for {ticker}")
                        
            except Exception as e:
                log_error(f"Error getting {label} price for {ticker}: {e}")
                results[field_prefix] = None
        
        return results
    
    def _is_during_market_hours(self, dt_est: datetime) -> bool:
        """Check if datetime (EST) is during market hours"""
        # Check if weekend
        if dt_est.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check time range (9:30am-4:00pm EST)
        market_open_time = dt_est.time().replace(hour=9, minute=30)
        market_close_time = dt_est.time().replace(hour=16, minute=0)
        
        return market_open_time <= dt_est.time() <= market_close_time
    
    def _adjust_for_next_trading_day(self, target_time: datetime, minutes_offset: int) -> datetime:
        """Adjust target time to next trading day if needed"""
        target_est = target_time.astimezone(self.est_tz)
        
        # If it's weekend, move to Monday
        if target_est.weekday() >= 5:
            days_to_monday = 7 - target_est.weekday()
            target_est = target_est + timedelta(days=days_to_monday)
        
        # Set to market open + offset
        market_open = target_est.replace(hour=9, minute=30, second=0, microsecond=0)
        adjusted_time = market_open + timedelta(minutes=minutes_offset)
        
        return adjusted_time.astimezone(self.utc_tz)
    
    def _get_next_trading_day_close(self, rec_time_est: datetime) -> datetime:
        """Get the next trading day's close time"""
        next_day = rec_time_est + timedelta(days=1)
        
        # If next day is weekend, move to Monday
        if next_day.weekday() >= 5:
            days_to_monday = 7 - next_day.weekday()
            next_day = next_day + timedelta(days=days_to_monday)
        
        close_time = next_day.replace(
            hour=Config.CLOSE_PRICE_HOUR,
            minute=Config.CLOSE_PRICE_MINUTE,
            second=0,
            microsecond=0
        )
        
        return close_time.astimezone(self.utc_tz)
    
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
            
            log_debug(f"Fetching {interval} data for {ticker} on {date_str}")
            data = self.fmp_loader.make_request(f"historical-chart/{interval}/{ticker}", params)
            
            if not data or not isinstance(data, list):
                log_warning(f"No intraday data returned for {ticker} on {date_str}")
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
                    log_warning(f"Error parsing price data point: {e}")
                    continue
            
            # Cache the results
            self.price_cache[cache_key] = historical_prices
            
            log_info(f"Fetched {len(historical_prices)} historical price points for {ticker} on {date_str}")
            return historical_prices
            
        except Exception as e:
            log_error(f"Error fetching intraday data for {ticker}: {e}")
            return []
    
    def _parse_fmp_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse FMP timestamp format to datetime"""
        try:
            # FMP returns timestamps like "2025-01-15 09:30:00" (EST)
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # FMP timestamps are in EST, convert to UTC
            est_dt = self.est_tz.localize(dt)
            utc_dt = est_dt.astimezone(self.utc_tz)
            
            return utc_dt
            
        except (ValueError, TypeError) as e:
            log_warning(f"Error parsing timestamp '{timestamp_str}': {e}")
            return None
    
    def _find_closest_price(self, historical_prices: List[HistoricalPrice], 
                           target_timestamp: datetime) -> Optional[HistoricalPrice]:
        """Find the historical price closest to target timestamp with flexible tolerance"""
        if not historical_prices:
            return None
        
        # Find the price with minimum time difference
        closest_price = None
        min_time_diff = float('inf')
        
        for price in historical_prices:
            time_diff = abs((price.timestamp - target_timestamp).total_seconds())
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_price = price
        
        # Use more flexible time windows based on market conditions
        target_est = target_timestamp.astimezone(self.est_tz)
        
        # During market hours: stricter tolerance (30 minutes)
        if 9.5 <= target_est.hour + target_est.minute/60 <= 16:
            max_tolerance = 1800  # 30 minutes during market hours
        else:
            # After hours or pre-market: more lenient (60 minutes)
            max_tolerance = 3600  # 60 minutes outside market hours
        
        if min_time_diff <= max_tolerance:
            log_debug(f"Found price {min_time_diff/60:.1f} minutes from target (tolerance: {max_tolerance/60:.0f}m)")
            return closest_price
        else:
            log_warning(f"Closest price is {min_time_diff/60:.1f} minutes away, exceeds tolerance ({max_tolerance/60:.0f}m)")
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
                    log_debug(f"Daily close for {ticker} on {date_str}: ${close_price:.2f}")
                    return close_price
            
            log_warning(f"No daily price data for {ticker} on {date_str}")
            return None
            
        except Exception as e:
            log_error(f"Error getting daily price for {ticker}: {e}")
            return None


class CSVBackfillUtility:
    """Utility to backfill missing entry prices in existing CSV data with configurable interval support"""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize backfill utility"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
        
        # Log current configuration for reference
        checkpoint_info = Config.get_checkpoint_info()
        log_info(f"CSV Backfill utility initialized for intervals: {[info['label'] for info in checkpoint_info]}")
        
    def analyze_missing_prices(self) -> Dict[str, Any]:
        """Analyze how many entries are missing price data with configurable interval awareness"""
        try:
            checkpoint_info = Config.get_checkpoint_info()
            
            analysis = {
                'total_rows': 0,
                'missing_entry_prices': 0,
                'missing_price_updates': 0,
                'completed_tracking': 0,
                'backfill_candidates': [],
                'configuration': {
                    'intervals': [info['label'] for info in checkpoint_info],
                    'check1_minutes': Config.PRICE_CHECK_1_MINUTES,
                    'check2_minutes': Config.PRICE_CHECK_2_MINUTES,
                    'close_time': f"{Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d}"
                }
            }
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    analysis['total_rows'] += 1
                    
                    ticker = row.get('ticker', '')
                    decision = row.get('decision', '')
                    entry_price = row.get('recommendation_price', '')
                    tracking_status = row.get('tracking_status', '')
                    timestamp = row.get('recommendation_timestamp', '')
                    
                    # Count missing entry prices
                    if not entry_price and decision in ['LONG', 'SHORT']:
                        analysis['missing_entry_prices'] += 1
                        
                        if timestamp:  # Can potentially backfill
                            analysis['backfill_candidates'].append({
                                'ticker': ticker,
                                'decision': decision,
                                'timestamp': timestamp
                            })
                    
                    # Count tracking completion
                    if tracking_status == 'completed':
                        analysis['completed_tracking'] += 1
                    elif not self._has_any_checkpoint_price(row) and entry_price:
                        analysis['missing_price_updates'] += 1
            
            return analysis
            
        except Exception as e:
            log_error(f"Error analyzing CSV: {e}")
            return {}
    
    def _has_any_checkpoint_price(self, row: Dict[str, str]) -> bool:
        """Check if row has any checkpoint price data based on current configuration"""
        checkpoint_info = Config.get_checkpoint_info()
        
        for checkpoint in checkpoint_info:
            field_prefix = checkpoint['field_prefix']
            if row.get(field_prefix):
                return True
        
        return False
    
    def backfill_missing_entry_prices(self, dry_run: bool = True) -> Dict[str, Any]:
        """Backfill missing entry prices using historical data"""
        
        log_info("🔄 Starting entry price backfill process with configurable intervals...")
        
        results = {
            'processed': 0,
            'successful_backfills': 0,
            'failed_backfills': 0,
            'backfilled_entries': [],
            'configuration': {
                'intervals': Config.get_price_check_labels(),
                'dry_run': dry_run
            }
        }
        
        try:
            # Read existing CSV
            rows = []
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            
            if not rows:
                log_warning("CSV file is empty")
                return results
            
            headers = rows[0].keys() if rows else []
            
            # Process each row
            for i, row in enumerate(rows):
                ticker = row.get('ticker', '')
                decision = row.get('decision', '')
                entry_price = row.get('recommendation_price', '')
                timestamp_str = row.get('recommendation_timestamp', '')
                
                # Skip if already has entry price or not a trading decision
                if entry_price or decision not in ['LONG', 'SHORT'] or not timestamp_str:
                    continue
                
                results['processed'] += 1
                
                try:
                    # Parse timestamp
                    target_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if target_timestamp.tzinfo is None:
                        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                    
                    # Get historical price
                    historical_price = self.historical_fetcher.get_price_at_timestamp(
                        ticker, target_timestamp, interval='5min', preferred_type='close'
                    )
                    
                    if historical_price:
                        if not dry_run:
                            row['recommendation_price'] = f"{historical_price:.2f}"
                        
                        results['successful_backfills'] += 1
                        results['backfilled_entries'].append({
                            'ticker': ticker,
                            'timestamp': timestamp_str,
                            'backfilled_price': historical_price
                        })
                        
                        log_info(f"✅ Backfilled {ticker}: ${historical_price:.2f} at {timestamp_str}")
                    else:
                        results['failed_backfills'] += 1
                        log_warning(f"❌ Could not backfill {ticker} at {timestamp_str}")
                
                except Exception as e:
                    results['failed_backfills'] += 1
                    log_error(f"Error backfilling {ticker}: {e}")
            
            # Write updated CSV (if not dry run)
            if not dry_run and results['successful_backfills'] > 0:
                backup_path = f"{self.csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create backup
                import shutil
                shutil.copy2(self.csv_path, backup_path)
                log_info(f"Created backup: {backup_path}")
                
                # Write updated file
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                
                log_info(f"✅ Updated CSV with {results['successful_backfills']} backfilled prices")
            
            return results
            
        except Exception as e:
            log_error(f"Error in backfill process: {e}")
            return results
    
    def backfill_missing_checkpoint_prices(self, dry_run: bool = True) -> Dict[str, Any]:
        """Backfill missing checkpoint prices using historical data with configurable intervals"""
        
        checkpoint_info = Config.get_checkpoint_info()
        intervals_summary = ", ".join([info['label'] for info in checkpoint_info])
        
        log_info(f"🔄 Starting checkpoint price backfill for intervals: {intervals_summary}")
        
        results = {
            'processed': 0,
            'successful_backfills': 0,
            'failed_backfills': 0,
            'backfilled_checkpoints': [],
            'configuration': {
                'intervals': [info['label'] for info in checkpoint_info],
                'dry_run': dry_run
            }
        }
        
        try:
            # Read existing CSV
            rows = []
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            
            if not rows:
                log_warning("CSV file is empty")
                return results
            
            headers = rows[0].keys() if rows else []
            
            # Process each row
            for i, row in enumerate(rows):
                ticker = row.get('ticker', '')
                decision = row.get('decision', '')
                entry_price = row.get('recommendation_price', '')
                timestamp_str = row.get('recommendation_timestamp', '')
                
                # Skip if not a trading decision or missing entry data
                if decision not in ['LONG', 'SHORT'] or not entry_price or not timestamp_str:
                    continue
                
                results['processed'] += 1
                
                try:
                    # Parse timestamp
                    target_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if target_timestamp.tzinfo is None:
                        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                    
                    # Get historical prices for all checkpoints
                    checkpoint_prices = self.historical_fetcher.get_checkpoint_prices(ticker, target_timestamp)
                    
                    backfilled_any = False
                    for field_prefix, price in checkpoint_prices.items():
                        # Only backfill if missing
                        if not row.get(field_prefix) and price is not None:
                            if not dry_run:
                                row[field_prefix] = f"{price:.2f}"
                                
                                # Calculate and set change percentage
                                change_pct = ((price - float(entry_price)) / float(entry_price)) * 100
                                row[f"{field_prefix}_change_pct"] = f"{change_pct:.2f}"
                                
                                # Set timestamp
                                row[f"{field_prefix}_timestamp"] = target_timestamp.isoformat()
                            
                            backfilled_any = True
                            
                            # Find checkpoint label for logging
                            checkpoint_label = "unknown"
                            for checkpoint in checkpoint_info:
                                if checkpoint['field_prefix'] == field_prefix:
                                    checkpoint_label = checkpoint['label']
                                    break
                            
                            results['backfilled_checkpoints'].append({
                                'ticker': ticker,
                                'checkpoint': checkpoint_label,
                                'price': price,
                                'timestamp': timestamp_str
                            })
                            
                            log_info(f"✅ Backfilled {ticker} {checkpoint_label}: ${price:.2f}")
                    
                    if backfilled_any:
                        results['successful_backfills'] += 1
                        
                except Exception as e:
                    results['failed_backfills'] += 1
                    log_error(f"Error backfilling checkpoints for {ticker}: {e}")
            
            # Write updated CSV (if not dry run)
            if not dry_run and results['successful_backfills'] > 0:
                backup_path = f"{self.csv_path}.backup_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create backup
                import shutil
                shutil.copy2(self.csv_path, backup_path)
                log_info(f"Created backup: {backup_path}")
                
                # Write updated file
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                
                log_info(f"✅ Updated CSV with checkpoint prices for {results['successful_backfills']} decisions")
            
            return results
            
        except Exception as e:
            log_error(f"Error in checkpoint backfill process: {e}")
            return results
    
    def validate_existing_prices(self, sample_size: int = 10) -> Dict[str, Any]:
        """Validate existing entry prices against historical data"""
        
        log_info(f"🔍 Validating {sample_size} existing entry prices...")
        
        validation_results = {
            'validated': 0,
            'accurate': 0,
            'inaccurate': 0,
            'errors': 0,
            'accuracy_details': [],
            'configuration': {
                'intervals': Config.get_price_check_labels(),
                'sample_size': sample_size
            }
        }
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                entries_with_prices = []
                for row in reader:
                    if (row.get('recommendation_price') and 
                        row.get('recommendation_timestamp') and 
                        row.get('decision') in ['LONG', 'SHORT']):
                        entries_with_prices.append(row)
                
                # Sample random entries
                import random
                sample_entries = random.sample(entries_with_prices, 
                                             min(sample_size, len(entries_with_prices)))
                
                for entry in sample_entries:
                    ticker = entry['ticker']
                    logged_price = float(entry['recommendation_price'])
                    timestamp_str = entry['recommendation_timestamp']
                    
                    try:
                        # Parse timestamp
                        target_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if target_timestamp.tzinfo is None:
                            target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                        
                        # Get historical price
                        historical_price = self.historical_fetcher.get_price_at_timestamp(
                            ticker, target_timestamp, interval='5min'
                        )
                        
                        validation_results['validated'] += 1
                        
                        if historical_price:
                            price_diff = abs(logged_price - historical_price)
                            price_diff_pct = (price_diff / historical_price) * 100
                            
                            # Consider accurate if within 1% (allows for small timing differences)
                            if price_diff_pct <= 1.0:
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
                        
                    except Exception as e:
                        validation_results['errors'] += 1
                        log_error(f"Error validating {ticker}: {e}")
            
            # Calculate accuracy rate
            if validation_results['validated'] > 0:
                accuracy_rate = validation_results['accurate'] / validation_results['validated']
                log_info(f"📊 Validation complete: {accuracy_rate:.1%} accuracy rate")
            
            return validation_results
            
        except Exception as e:
            log_error(f"Error in price validation: {e}")
            return validation_results


class HistoricalPerformanceAnalyzer:
    """Analyze historical performance of trading decisions with configurable interval support"""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize performance analyzer"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
        
        # Log configuration
        checkpoint_info = Config.get_checkpoint_info()
        log_info(f"Performance analyzer initialized for intervals: {[info['label'] for info in checkpoint_info]}")
    
    def calculate_performance_metrics(self, days_to_analyze: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics for completed trades with configurable interval analysis"""
        
        checkpoint_info = Config.get_checkpoint_info()
        
        log_info(f"📊 Analyzing performance for last {days_to_analyze} days across {len(checkpoint_info)} intervals...")
        
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'best_trade_pct': 0.0,
            'worst_trade_pct': 0.0,
            'total_return_pct': 0.0,
            'trade_details': [],
            'checkpoint_performance': {},  # Performance by interval
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
                'best_trade_pct': 0.0,
                'worst_trade_pct': 0.0
            }
        
        try:
            # Use timezone-aware cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_analyze)
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Only analyze trading decisions with entry price
                    if (row.get('recommendation_price') and 
                        row.get('decision') in ['LONG', 'SHORT']):
                        
                        # Check if within analysis period with proper timezone handling
                        timestamp_str = row.get('recommendation_timestamp', '')
                        if timestamp_str:
                            try:
                                # Parse timestamp and ensure it's timezone-aware
                                if timestamp_str.endswith('Z'):
                                    entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                elif '+' in timestamp_str or timestamp_str.endswith('UTC'):
                                    entry_time = datetime.fromisoformat(timestamp_str.replace(' UTC', '+00:00'))
                                else:
                                    # Assume UTC if no timezone info
                                    entry_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                                
                                # Compare timezone-aware datetimes
                                if entry_time < cutoff_date:
                                    continue
                                    
                            except (ValueError, TypeError) as e:
                                log_warning(f"Error parsing timestamp '{timestamp_str}': {e}")
                                continue
                        
                        ticker = row['ticker']
                        decision = row['decision']
                        
                        try:
                            entry_price = float(row['recommendation_price'])
                        except (ValueError, TypeError):
                            log_warning(f"Invalid entry price for {ticker}: {row.get('recommendation_price')}")
                            continue
                        
                        # Analyze performance at each configured checkpoint
                        for checkpoint in checkpoint_info:
                            field_prefix = checkpoint['field_prefix']
                            label = checkpoint['label']
                            
                            exit_price_str = row.get(field_prefix)
                            if exit_price_str:
                                try:
                                    exit_price = float(exit_price_str)
                                    
                                    # Calculate return based on position direction
                                    if decision == 'LONG':
                                        return_pct = ((exit_price - entry_price) / entry_price) * 100
                                    else:  # SHORT
                                        return_pct = ((entry_price - exit_price) / entry_price) * 100
                                    
                                    # Track overall metrics (use close price as primary)
                                    if field_prefix == 'price_close':
                                        metrics['total_trades'] += 1
                                        metrics['total_return_pct'] += return_pct
                                        
                                        if return_pct > 0:
                                            metrics['winning_trades'] += 1
                                            metrics['avg_win_pct'] += return_pct
                                            metrics['best_trade_pct'] = max(metrics['best_trade_pct'], return_pct)
                                        else:
                                            metrics['losing_trades'] += 1
                                            metrics['avg_loss_pct'] += abs(return_pct)
                                            metrics['worst_trade_pct'] = min(metrics['worst_trade_pct'], return_pct)
                                        
                                        metrics['trade_details'].append({
                                            'ticker': ticker,
                                            'decision': decision,
                                            'entry_price': entry_price,
                                            'exit_price': exit_price,
                                            'return_pct': return_pct,
                                            'timestamp': timestamp_str,
                                            'checkpoint': label
                                        })
                                    
                                    # Track checkpoint-specific performance
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
                                    
                                except (ValueError, ZeroDivisionError) as e:
                                    log_warning(f"Error calculating performance for {ticker} at {label}: {e}")
            
            # Calculate final metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                
                if metrics['winning_trades'] > 0:
                    metrics['avg_win_pct'] /= metrics['winning_trades']
                
                if metrics['losing_trades'] > 0:
                    metrics['avg_loss_pct'] /= metrics['losing_trades']
            
            # Calculate checkpoint-specific metrics
            for label, checkpoint_metrics in metrics['checkpoint_performance'].items():
                if checkpoint_metrics['total_trades'] > 0:
                    checkpoint_metrics['win_rate'] = checkpoint_metrics['winning_trades'] / checkpoint_metrics['total_trades']
                    checkpoint_metrics['avg_return_pct'] = checkpoint_metrics['total_return_pct'] / checkpoint_metrics['total_trades']
            
            log_info(f"📈 Performance Analysis Complete:")
            log_info(f"   Total Trades: {metrics['total_trades']}")
            log_info(f"   Win Rate: {metrics['win_rate']:.1%}")
            log_info(f"   Avg Win: {metrics['avg_win_pct']:.2f}%")
            log_info(f"   Avg Loss: {metrics['avg_loss_pct']:.2f}%")
            log_info(f"   Total Return: {metrics['total_return_pct']:.2f}%")
            
            # Log checkpoint performance
            for label, checkpoint_metrics in metrics['checkpoint_performance'].items():
                if checkpoint_metrics['total_trades'] > 0:
                    log_info(f"   {label.upper()} Performance: {checkpoint_metrics['win_rate']:.1%} win rate, {checkpoint_metrics['avg_return_pct']:.2f}% avg return")
            
            return metrics
            
        except Exception as e:
            log_error(f"Error calculating performance metrics: {e}")
            return metrics


# Enhanced standalone functions with configurable interval support
def backfill_missing_prices(csv_path: str, fmp_api_key: str, dry_run: bool = True):
    """Standalone function to backfill missing entry prices with configurable interval awareness"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    # Analyze what needs backfilling
    analysis = backfill_utility.analyze_missing_prices()
    log_info(f"Analysis: {analysis['missing_entry_prices']} entries need backfilling")
    log_info(f"Configuration: {analysis.get('configuration', {})}")
    
    # Perform backfill
    if analysis['missing_entry_prices'] > 0:
        results = backfill_utility.backfill_missing_entry_prices(dry_run=dry_run)
        log_info(f"Backfill results: {results}")
        return results
    else:
        log_info("No entries need backfilling")
        return {}

def backfill_checkpoint_prices(csv_path: str, fmp_api_key: str, dry_run: bool = True):
    """Standalone function to backfill missing checkpoint prices"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    return backfill_utility.backfill_missing_checkpoint_prices(dry_run=dry_run)

def validate_price_accuracy(csv_path: str, fmp_api_key: str, sample_size: int = 10):
    """Standalone function to validate existing price accuracy"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    return backfill_utility.validate_existing_prices(sample_size)

def analyze_trading_performance(csv_path: str, fmp_api_key: str, days: int = 30):
    """Standalone function to analyze trading performance with configurable intervals"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    analyzer = HistoricalPerformanceAnalyzer(csv_path, fmp_loader)
    
    return analyzer.calculate_performance_metrics(days)


def diagnose_missing_prices(csv_path: str, fmp_api_key: str, limit: int = 10):
    """Diagnose why prices are missing with detailed market timing analysis"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    log_info(f"🔍 Diagnosing missing prices for up to {limit} entries...")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            missing_entries = []
            for row in reader:
                ticker = row.get('ticker', '')
                decision = row.get('decision', '')
                entry_price = row.get('recommendation_price', '')
                timestamp_str = row.get('recommendation_timestamp', '')
                
                # Find entries with missing price data
                if (decision in ['LONG', 'SHORT'] and entry_price and timestamp_str and 
                    not backfill_utility._has_any_checkpoint_price(row)):
                    missing_entries.append(row)
                    if len(missing_entries) >= limit:
                        break
            
            log_info(f"Found {len(missing_entries)} entries with missing price data")
            
            for i, row in enumerate(missing_entries, 1):
                ticker = row['ticker']
                timestamp_str = row['recommendation_timestamp']
                
                try:
                    # Parse timestamp
                    target_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if target_timestamp.tzinfo is None:
                        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
                    
                    # Convert to EST for analysis
                    est_tz = pytz.timezone('US/Eastern')
                    target_est = target_timestamp.astimezone(est_tz)
                    
                    log_info(f"\n📊 {i}. {ticker} - {target_est.strftime('%Y-%m-%d %H:%M EST (%A)')}")
                    
                    # Analyze market timing
                    is_weekend = target_est.weekday() >= 5
                    is_market_hours = (9.5 <= target_est.hour + target_est.minute/60 <= 16) and not is_weekend
                    
                    log_info(f"   Decision time: {target_est.strftime('%H:%M EST')} on {target_est.strftime('%A')}")
                    log_info(f"   Market status: {'✅ Open' if is_market_hours else '❌ Closed'}")
                    
                    if is_weekend:
                        log_info(f"   Issue: Weekend - no trading data available")
                    elif target_est.hour < 9 or (target_est.hour == 9 and target_est.minute < 30):
                        log_info(f"   Issue: Before market open (9:30 AM)")
                    elif target_est.hour >= 16:
                        log_info(f"   Issue: After market close (4:00 PM)")
                    
                    # Check what checkpoint times would be
                    checkpoint_info = Config.get_checkpoint_info()
                    for checkpoint in checkpoint_info:
                        if checkpoint.get('minutes'):
                            check_time = target_timestamp + timedelta(minutes=checkpoint['minutes'])
                            check_est = check_time.astimezone(est_tz)
                            log_info(f"   {checkpoint['label']} target: {check_est.strftime('%H:%M EST on %A')}")
                        else:
                            close_time = target_est.replace(hour=Config.CLOSE_PRICE_HOUR, minute=Config.CLOSE_PRICE_MINUTE)
                            log_info(f"   {checkpoint['label']} target: {close_time.strftime('%H:%M EST on %A')}")
                    
                except Exception as e:
                    log_error(f"Error analyzing {ticker}: {e}")
            
    except Exception as e:
        log_error(f"Error in diagnosis: {e}")


if __name__ == "__main__":
    """Example usage with enhanced diagnostic tools"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    
    if not Config.FMP_API_KEY:
        print("FMP_API_KEY is required")
        sys.exit(1)
    
    csv_path = "output/trading_decisions.csv"
    
    print("🔄 Enhanced Historical Price Utilities with Diagnostic Tools")
    print(f"Current configuration: {Config.get_price_check_labels()}")
    
    print("\n1. Analyzing missing prices...")
    analysis = backfill_missing_prices(csv_path, Config.FMP_API_KEY, dry_run=True)
    
    print("\n2. Diagnosing market timing issues...")
    diagnose_missing_prices(csv_path, Config.FMP_API_KEY, limit=5)
    
    print("\n3. Backfilling missing checkpoint prices...")
    checkpoint_results = backfill_checkpoint_prices(csv_path, Config.FMP_API_KEY, dry_run=True)
    
    print("\n4. Validating existing prices...")
    validation = validate_price_accuracy(csv_path, Config.FMP_API_KEY, sample_size=5)
    
    print("\n5. Analyzing performance...")
    performance = analyze_trading_performance(csv_path, Config.FMP_API_KEY, days=30)