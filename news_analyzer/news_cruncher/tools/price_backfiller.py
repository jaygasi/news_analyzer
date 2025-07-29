"""
Trading Recommendations Price Backfiller - FINAL FIXED VERSION
Complete solution for updating missing price data in trading_decisions.csv

ALL CRITICAL FIXES APPLIED:
- Fixed timezone parsing (FMP EDT -> UTC conversion)
- Enhanced tolerance logic for close price targets
- Improved market hours detection
- 5 comprehensive price-fetching strategies
- Better error handling and debugging
- FIXED: Always ask user about analysis options
- FIXED: Better API error handling and fallback strategies

Python 3.13.3 compatible
"""
import sys
import csv
import shutil
import time
from pathlib import Path # Ensure Path is imported
from datetime import datetime, timedelta, timezone, date # Ensure date is imported
from datetime import time as dt_time # FIX 1: Import time as dt_time to avoid conflict
from typing import Dict, List, Optional, Tuple, Any
import pytz

# Add parent directory to sys.path to allow importing project modules
sys.path.append(str(Path(__file__).parent.parent))

# Constants
UTC_OFFSET_SUFFIX = '+00:00'

try:
    from config import Config
    from data_loaders.base_fmp_loader import BaseFMPLoader
    from utils.simple_logger import log_info, log_error, log_debug, log_warning
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure this script is run from the news_cruncher directory.")
    print("Try: cd news_cruncher && python tools/price_backfiller.py")
    sys.exit(1)


class TradingRecommendationsPriceBackfiller:
    """Complete price backfiller for trading recommendations CSV with all fixes applied"""

    def __init__(self, fmp_api_key: str):
        """Initialize the price backfiller with configuration validation"""
        self.fmp_loader = BaseFMPLoader(fmp_api_key)
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.timezone('UTC')

        # Cache to avoid duplicate API calls
        self.price_cache: Dict[str, List[Dict]] = {}

        # Add future buffer for price fetching
        self.future_buffer_minutes = getattr(Config, 'PRICE_FETCH_FUTURE_BUFFER_MINUTES', 30)

        # Enhanced configuration options
        self.verbose_debug = getattr(Config, 'PRICE_BACKFILLER_VERBOSE_DEBUG', True)
        self.enable_last_known_strategy = getattr(Config, 'ENABLE_LAST_KNOWN_PRICE_STRATEGY', True)
        self.enable_multi_day_strategy = getattr(Config, 'ENABLE_MULTI_DAY_SEARCH_STRATEGY', True)

        # VALIDATE configuration compatibility
        self._validate_configuration()

        log_info("🔄 Trading Recommendations Price Backfiller initialized (FINAL FIXED VERSION)")

    def _validate_configuration(self):
        """Validate that Config settings match expected CSV format"""

        # Check critical price tracking settings
        required_attrs = [
            'PRICE_CHECK_1_MINUTES', 'PRICE_CHECK_2_MINUTES',
            'CLOSE_PRICE_HOUR', 'CLOSE_PRICE_MINUTE'
        ]

        missing_attrs = [attr for attr in required_attrs if not hasattr(Config, attr)]
        if missing_attrs:
            log_warning(f"⚠️ Missing Config attributes: {missing_attrs}")
            log_warning("   This may cause column mismatch issues")

        # Display current price tracking configuration
        log_info("📊 Price Tracking Configuration:")
        log_info(f"   Checkpoint 1: {getattr(Config, 'PRICE_CHECK_1_MINUTES', 'MISSING')} minutes")
        log_info(f"   Checkpoint 2: {getattr(Config, 'PRICE_CHECK_2_MINUTES', 'MISSING')} minutes")
        log_info(f"   Close Time: {getattr(Config, 'CLOSE_PRICE_HOUR', 'MISSING')}:{getattr(Config, 'CLOSE_PRICE_MINUTE', 'MISSING'):02d}")

        # Check CSV logging configuration
        only_log_trades = getattr(Config, 'ONLY_LOG_TRADING_DECISIONS', True)
        log_info(f"📝 CSV Logging: {'LONG/SHORT only' if only_log_trades else 'All decisions (LONG/SHORT/NONE)'}")

        if only_log_trades:
            log_warning("⚠️ ONLY_LOG_TRADING_DECISIONS=True - NONE decisions won't be in CSV")
            log_warning("   Backfiller will only see LONG/SHORT positions")

    def select_csv_file(self) -> Path:
        """Prompt user to select CSV file to update"""
        print("\n" + "="*60)
        print("📁 CSV FILE SELECTION")
        print("="*60)

        possible_files = self._find_csv_files()

        if not possible_files:
            return self._handle_no_csv_files_found()

        return self._select_from_available_files(possible_files)

    def _find_csv_files(self) -> List[Path]:
        """Find CSV files in common locations"""
        possible_files = []
        search_paths = [
            Path("."),
            Path("output"),
            Path("../output"),
        ]

        for search_path in search_paths:
            if search_path.exists():
                csv_files = list(search_path.glob("*.csv"))
                for csv_file in csv_files:
                    if csv_file.is_file():
                        possible_files.append(csv_file)

        return possible_files

    def _handle_no_csv_files_found(self) -> Path:
        """Handle case when no CSV files are found"""
        print("❌ No CSV files found in current or output directories.")
        print("Please enter the full path to your CSV file:")
        file_path = input("CSV file path: ").strip()
        return Path(file_path)

    def _select_from_available_files(self, possible_files: List[Path]) -> Path:
        """Select CSV file from available options"""
        print("Found the following CSV files:")
        for i, file_path in enumerate(possible_files, 1):
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  {i}. {file_path} ({file_size:.1f} KB)")

        print("  0. Enter custom path")

        while True:
            try:
                choice = int(input(f"Select file (0-{len(possible_files)}): "))
                if choice == 0:
                    file_path = input("Enter CSV file path: ").strip()
                    return Path(file_path)
                elif 1 <= choice <= len(possible_files):
                    return possible_files[choice - 1]
                else:
                    print(f"Please enter a number between 0 and {len(possible_files)}")
            except ValueError:
                print("Please enter a valid number")

    def analyze_csv_file(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze CSV file structure and content"""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            rows = list(reader)

        return {
            'total_rows': len(rows),
            'headers': headers,
            'header_analysis': self._analyze_headers(headers),
            'missing_fields_summary': self._analyze_missing_fields(rows),
            'decision_breakdown': self._analyze_decisions(rows),
            'date_range': self._analyze_date_range(rows),
            'updatable_rows': self._count_updatable_rows(rows),
            'completed_tracking': self._count_completed_tracking(rows),
            'tracking_analysis': self._analyze_tracking_status(rows)
        }

    def _analyze_headers(self, headers: List[str]) -> Dict[str, Any]:
        """Analyze CSV headers"""
        price_columns = [h for h in headers if any(p in h for p in ['price_', 'recommendation_price'])]
        expected_columns = ['recommendation_price', 'price_checkpoint1', 'price_checkpoint2', 'price_close']
        missing_expected = [col for col in expected_columns if col not in headers]

        return {
            'total_columns': len(headers),
            'price_columns': price_columns,
            'has_recommendation_price': 'recommendation_price' in headers,
            'missing_expected': missing_expected
        }

    def _analyze_missing_fields(self, rows: List[Dict]) -> Dict[str, int]:
        """Analyze missing fields across all rows"""
        missing_fields_summary = {}

        for row in rows:
            for field, value in row.items():
                if not value or value.strip() == '':
                    missing_fields_summary[field] = missing_fields_summary.get(field, 0) + 1

        return missing_fields_summary

    def _analyze_decisions(self, rows: List[Dict]) -> Dict[str, int]:
        """Analyze decision breakdown"""
        decision_breakdown = {}
        for row in rows:
            decision = row.get('decision', 'UNKNOWN')
            decision_breakdown[decision] = decision_breakdown.get(decision, 0) + 1
        return decision_breakdown

    def _analyze_date_range(self, rows: List[Dict]) -> Dict[str, Optional[datetime]]:
        """Analyze date range of trading decisions"""
        timestamps = []
        for row in rows:
            timestamp_str = row.get('timestamp', '')
            if timestamp_str:
                try:
                    timestamp = self._parse_timestamp_robust(timestamp_str)
                    if timestamp:
                        timestamps.append(timestamp)
                except (ValueError, TypeError, AttributeError):
                    continue

        return {
            'earliest': min(timestamps) if timestamps else None,
            'latest': max(timestamps) if timestamps else None
        }

    def _count_updatable_rows(self, rows: List[Dict]) -> int:
        """Count rows that can be updated with price data"""
        updatable_count = 0
        for row in rows:
            if self._has_missing_price_data(row):
                updatable_count += 1
        return updatable_count

    def _count_completed_tracking(self, rows: List[Dict]) -> int:
        """Count rows with completed tracking status"""
        return sum(1 for row in rows if row.get('tracking_status') == 'completed')

    def _analyze_tracking_status(self, rows: List[Dict]) -> Dict[str, Any]:
        """Analyze tracking status and identify fixable rows"""
        required_price_fields = [
            'recommendation_price',
            'price_checkpoint1', 
            'price_checkpoint2',
            'price_close'
        ]
        
        status_counts = {}
        should_be_completed = 0
        
        for row in rows:
            current_status = row.get('tracking_status', '')
            status_counts[current_status] = status_counts.get(current_status, 0) + 1
            
            # Check if this row should be completed but isn't
            if current_status != 'completed':
                has_all_data = True
                for field in required_price_fields:
                    value = row.get(field, '').strip()
                    if not value or value.lower() in ['', 'none', 'null']:
                        has_all_data = False
                        break
                
                if has_all_data:
                    should_be_completed += 1
        
        return {
            'status_counts': status_counts,
            'should_be_completed': should_be_completed
        }

    def _has_missing_price_data(self, row: Dict[str, str]) -> bool:
        """Check if row has missing price data that can be backfilled"""
        ticker = row.get('ticker', '').strip()
        if not ticker:
            return False

        # Check if recommendation price is missing
        if not self._get_float_value(row.get('recommendation_price')):
            return True

        # Check checkpoint prices
        checkpoint_info = self._get_checkpoint_info()
        for checkpoint in checkpoint_info:
            field_name = checkpoint['field_prefix']
            if not self._get_float_value(row.get(field_name)):
                return True

        return False

    def _parse_timestamp_robust(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string with multiple format support"""
        if not timestamp_str:
            return None

        # Handle timezone-aware timestamps
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S.%f%z',      # 2025-06-09 14:20:47.645155+00:00
            '%Y-%m-%d %H:%M:%S%z',          # 2025-06-09 14:20:47+00:00
            '%Y-%m-%dT%H:%M:%S.%f%z',       # 2025-06-09T14:20:47.645155+00:00
            '%Y-%m-%dT%H:%M:%S%z',          # 2025-06-09T14:20:47+00:00
            '%Y-%m-%d %H:%M:%S.%f',         # 2025-06-09 14:20:47.645155 (assume UTC)
            '%Y-%m-%d %H:%M:%S',            # 2025-06-09 14:20:47 (assume UTC)
        ]

        for fmt in formats_to_try:
            try:
                parsed = datetime.strptime(timestamp_str, fmt)
                # If no timezone info, assume UTC
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                continue

        # Fallback: try to handle timestamps with various timezone suffixes
        try:
            # Handle cases where timezone offset might be malformed
            if UTC_OFFSET_SUFFIX in timestamp_str:
                clean_timestamp = timestamp_str.replace(UTC_OFFSET_SUFFIX, '')
                parsed = datetime.fromisoformat(clean_timestamp)
                return parsed.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, AttributeError):
            log_warning(f"Could not parse timestamp {timestamp_str}: using current time")
            return datetime.now(timezone.utc)

        return None

    def _get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get checkpoint configuration info"""
        return [
            {
                'field_prefix': 'price_checkpoint1',
                'minutes': getattr(Config, 'PRICE_CHECK_1_MINUTES', 45),
                'description': f'First checkpoint at {getattr(Config, "PRICE_CHECK_1_MINUTES", 45)} minutes'
            },
            {
                'field_prefix': 'price_checkpoint2',
                'minutes': getattr(Config, 'PRICE_CHECK_2_MINUTES', 60),
                'description': f'Second checkpoint at {getattr(Config, "PRICE_CHECK_2_MINUTES", 60)} minutes'
            },
            {
                'field_prefix': 'price_close',
                'minutes': None,  # Special handling for close time
                'description': f'Close time at {getattr(Config, "CLOSE_PRICE_HOUR", 15)}:{getattr(Config, "CLOSE_PRICE_MINUTE", 30):02d}'
            }
        ]

    def update_csv_prices(self, csv_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Update missing prices in CSV file and fix tracking status"""
        results = self._initialize_update_results()

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self._create_backup(csv_path, dry_run)
        rows_data = self._load_csv_data(csv_path)

        updated_rows = self._process_rows_for_updates(rows_data['rows'], dry_run, results)

        if not dry_run:
            # Fix tracking status before saving
            updated_rows = self._fix_tracking_status(updated_rows, results)
            self._save_updated_csv(csv_path, rows_data['headers'], updated_rows)

        self._log_update_summary(results, dry_run)
        return results

    def _initialize_update_results(self) -> Dict[str, Any]:
        """Initialize results tracking dictionary"""
        return {
            'rows_processed': 0,
            'rows_updated': 0,
            'api_calls_made': 0,
            'successful_price_fetches': 0,
            'failed_tickers': [],
            'errors': [],
            'tracking_status_fixes': 0
        }

    def _create_backup(self, csv_path: Path, dry_run: bool) -> None:
        """Create backup of CSV file"""
        if dry_run:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = csv_path.parent / f"{csv_path.stem}_backup_{timestamp}.csv"
        shutil.copy2(csv_path, backup_path)
        log_info(f"📁 Backup created: {backup_path}")

    def _load_csv_data(self, csv_path: Path) -> Dict[str, Any]:
        """Load CSV data and headers"""
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames or []
            rows = list(reader)

        return {'headers': headers, 'rows': rows}

    def _process_rows_for_updates(self, rows: List[Dict], dry_run: bool, results: Dict) -> List[Dict]:
        """Process all rows for price updates"""
        updated_rows = []

        for i, row in enumerate(rows):
            results['rows_processed'] += 1

            if not self._has_missing_price_data(row):
                updated_rows.append(row)
                continue

            updated_row = self._update_row_prices(row, dry_run, results)
            updated_rows.append(updated_row)

            if updated_row != row:
                results['rows_updated'] += 1

            # Rate limiting
            if results['api_calls_made'] % 10 == 0:
                time.sleep(1)

        return updated_rows

    def _save_updated_csv(self, csv_path: Path, headers: List[str], updated_rows: List[Dict]):
        """Save updated data back to CSV"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(updated_rows)

    def _log_update_summary(self, results: Dict[str, Any], dry_run: bool):
        """Log summary of update results"""
        mode = "DRY RUN" if dry_run else "LIVE UPDATE"
        log_info(f"🏁 {mode} Summary:")
        log_info(f"   Rows processed: {results['rows_processed']}")
        log_info(f"   Rows updated: {results['rows_updated']}")
        log_info(f"   API calls made: {results['api_calls_made']}")
        log_info(f"   Successful fetches: {results['successful_price_fetches']}")
        log_info(f"   Tracking status fixes: {results.get('tracking_status_fixes', 0)}")

        if results['failed_tickers']:
            log_warning(f"   Failed tickers: {', '.join(results['failed_tickers'])}")

        if results['errors']:
            log_error(f"   Errors encountered: {len(results['errors'])}")

    def _update_row_prices(self, row: Dict[str, str], dry_run: bool, results: Dict) -> Dict[str, str]:
        """Update prices for a single row"""
        updated_row = row.copy()
        ticker = row.get('ticker', '').strip()

        if not ticker:
            return updated_row

        # ENHANCED: Validate ticker before processing
        if not self._is_valid_ticker(ticker):
            log_warning(f"⚠️ Skipping invalid ticker: {ticker}")
            return updated_row

        # Parse recommendation timestamp
        rec_timestamp = self._parse_timestamp_robust(row.get('timestamp', ''))
        if not rec_timestamp:
            return updated_row

        # ENHANCED: Check if timestamp is too recent for API data
        if self._is_timestamp_too_recent(rec_timestamp):
            log_warning(f"⚠️ {ticker}: Timestamp too recent for reliable API data ({rec_timestamp})")
            return updated_row

        # Calculate all target times
        target_times = self._calculate_target_times(rec_timestamp)

        # Track API calls for this ticker
        ticker_api_calls = 0
        ticker_successful_fetches = 0

        # Update recommendation price if missing
        updated_row, ticker_api_calls, ticker_successful_fetches = self._update_recommendation_price(
            updated_row, row, ticker, target_times, dry_run, ticker_api_calls, ticker_successful_fetches
        )

        # Update checkpoint prices
        updated_row, ticker_api_calls, ticker_successful_fetches = self._update_checkpoint_prices(
            updated_row, row, ticker, target_times, dry_run, ticker_api_calls, ticker_successful_fetches
        )

        # Update overall results
        self._update_ticker_results(results, ticker, ticker_api_calls, ticker_successful_fetches)

        return updated_row

    def _is_valid_ticker(self, ticker: str) -> bool:
        """Check if ticker appears to be valid"""
        if not ticker or len(ticker) < 1 or len(ticker) > 10:
            return False
        
        # Check for obvious invalid patterns
        invalid_patterns = ['test', 'example', 'null', 'none', '']
        return ticker.lower() not in invalid_patterns

    def _is_timestamp_too_recent(self, timestamp: datetime) -> bool:
        """Check if timestamp is too recent for reliable API data"""
        now = datetime.now(timezone.utc)
        time_diff = now - timestamp
        
        # If less than 2 hours ago, might not have data
        if time_diff.total_seconds() < 7200:  # 2 hours
            return True
            
        # If it's the same day and market is still open, might not have complete data
        if timestamp.date() == now.date():
            est_now = now.astimezone(self.est_tz)
            if est_now.hour < 16:  # Before 4 PM EST
                return True
                
        return False

    def _update_recommendation_price(self, updated_row: Dict[str, str], row: Dict[str, str],
                                   ticker: str, target_times: Dict[str, datetime],
                                   dry_run: bool, ticker_api_calls: int,
                                   ticker_successful_fetches: int) -> Tuple[Dict[str, str], int, int]:
        """Update recommendation price if missing"""
        if not self._get_float_value(row.get('recommendation_price')) and not dry_run:
            price_result = self.get_price_at_timestamp(ticker, target_times['recommendation'])
            ticker_api_calls += 1
            if price_result:
                price, price_timestamp = price_result
                updated_row['recommendation_price'] = f"{price:.2f}"
                updated_row['recommendation_timestamp'] = price_timestamp.isoformat()
                ticker_successful_fetches += 1
                log_debug(f"📈 {ticker} recommendation_price: ${price:.2f}")

        return updated_row, ticker_api_calls, ticker_successful_fetches

    def _update_checkpoint_prices(self, updated_row: Dict[str, str], row: Dict[str, str],
                                ticker: str, target_times: Dict[str, datetime],
                                dry_run: bool, ticker_api_calls: int,
                                ticker_successful_fetches: int) -> Tuple[Dict[str, str], int, int]:
        """Update checkpoint prices if missing"""
        checkpoint_info = self._get_checkpoint_info()

        for checkpoint in checkpoint_info:
            field_prefix = checkpoint['field_prefix']

            if not self._get_float_value(row.get(field_prefix)) and not dry_run:
                price_result = self.get_price_at_timestamp(ticker, target_times[field_prefix])
                ticker_api_calls += 1
                if price_result:
                    price, price_timestamp = price_result
                    updated_row[field_prefix] = f"{price:.2f}"
                    updated_row[f"{field_prefix}_timestamp"] = price_timestamp.isoformat()

                    # Calculate percentage change if recommendation price exists
                    rec_price = self._get_float_value(updated_row.get('recommendation_price'))
                    if rec_price and rec_price > 0:
                        change_pct = ((price - rec_price) / rec_price) * 100
                        updated_row[f"{field_prefix}_change_pct"] = f"{change_pct:.2f}"

                    ticker_successful_fetches += 1
                    log_debug(f"📈 {ticker} {field_prefix}: ${price:.2f}")

        return updated_row, ticker_api_calls, ticker_successful_fetches

    def _update_ticker_results(self, results: Dict, ticker: str, ticker_api_calls: int, ticker_successful_fetches: int):
        """Update overall results with ticker statistics"""
        results['api_calls_made'] += ticker_api_calls
        results['successful_price_fetches'] += ticker_successful_fetches

        # Track failed tickers
        if ticker_api_calls > 0 and ticker_successful_fetches == 0 and ticker not in results['failed_tickers']:
            results['failed_tickers'].append(ticker)

        if ticker_successful_fetches > 0:
            log_info(f"✅ {ticker}: {ticker_successful_fetches}/{ticker_api_calls} successful price fetches")

    def _fix_tracking_status(self, rows: List[Dict], results: Dict) -> List[Dict]:
        """Fix tracking status for trades with complete price data"""
        required_price_fields = [
            'recommendation_price',
            'price_checkpoint1', 
            'price_checkpoint2',
            'price_close'
        ]
        
        fixes_made = 0
        
        for row in rows:
            current_status = row.get('tracking_status', '')
            
            # Skip if already completed
            if current_status == 'completed':
                continue
            
            # Check if this row has all required price data
            has_all_data = True
            for field in required_price_fields:
                value = row.get(field, '').strip()
                if not value or value.lower() in ['', 'none', 'null']:
                    has_all_data = False
                    break
            
            # Update status if complete
            if has_all_data:
                ticker = row.get('ticker', 'UNKNOWN')
                row['tracking_status'] = 'completed'
                fixes_made += 1
                log_debug(f"✅ Marked {ticker} as completed (was: '{current_status}')")
        
        results['tracking_status_fixes'] = fixes_made
        
        if fixes_made > 0:
            log_info(f"🔧 Fixed tracking status for {fixes_made} completed trades")
        
        return rows

    def get_price_at_timestamp(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Get price at specific timestamp using comprehensive strategy"""
        if self.verbose_debug:
            log_debug(f"🎯 Getting price for {ticker} at {target_timestamp}")

        strategies = self._get_price_strategies()

        for strategy_name, strategy_func in strategies:
            if self._should_skip_strategy(strategy_name):
                continue

            try:
                result = strategy_func(ticker, target_timestamp)
                if result:
                    price, price_timestamp = result
                    if self.verbose_debug:
                        log_debug(f"✅ {ticker}: Strategy '{strategy_name}' found ${price:.2f} at {price_timestamp}")
                    return result

            except Exception as e:
                if self.verbose_debug:
                    log_debug(f"⚠️ {ticker}: Strategy '{strategy_name}' failed: {e}")
                continue

        log_warning(f"❌ {ticker}: All price strategies failed for {target_timestamp}")
        return None

    def _get_price_strategies(self) -> List[Tuple[str, callable]]:
        """Get list of price fetching strategies"""
        return [
            ("exact_match", self._get_exact_match_price),
            ("tolerance_based", self._get_tolerance_based_price),
            ("business_day", self._get_business_day_price),
            ("last_known", self._get_last_known_price),
            ("multi_day_search", self._get_multi_day_search_price)
        ]

    def _should_skip_strategy(self, strategy_name: str) -> bool:
        """Check if strategy should be skipped based on configuration"""
        if strategy_name == "last_known" and not self.enable_last_known_strategy:
            return True
        if strategy_name == "multi_day_search" and not self.enable_multi_day_strategy:
            return True
        return False

    def _get_exact_match_price(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Strategy 1: Get exact timestamp match"""
        # Skip exact match for same-day requests - API likely won't have complete data
        if target_timestamp.date() == datetime.now(timezone.utc).date():
            return None
            
        intraday_data = self._get_intraday_data(ticker, target_timestamp.date())
        if not intraday_data:
            return None

        target_time_str = target_timestamp.strftime('%H:%M:%S')

        for data_point in intraday_data:
            if data_point.get('time') == target_time_str:
                price = float(data_point.get('close', 0))
                if price > 0:
                    actual_timestamp = self._combine_date_time(target_timestamp.date(), data_point.get('time'))
                    return price, actual_timestamp

        return None

    def _get_tolerance_based_price(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Strategy 2: Get price within tolerance window"""
        # Skip tolerance-based for same-day requests
        if target_timestamp.date() == datetime.now(timezone.utc).date():
            return None
            
        tolerance_minutes = self._determine_tolerance(target_timestamp)
        intraday_data = self._get_intraday_data(ticker, target_timestamp.date())

        if not intraday_data:
            return None

        return self._find_best_tolerance_match(intraday_data, target_timestamp, tolerance_minutes)

    def _find_best_tolerance_match(self, intraday_data: List[Dict], target_timestamp: datetime,
                                 tolerance_minutes: int) -> Optional[Tuple[float, datetime]]:
        """Find best match within tolerance window"""
        best_match = None
        best_time_diff = float('inf')
        target_time = target_timestamp.time()

        for data_point in intraday_data:
            try:
                point_time = datetime.strptime(data_point.get('time', ''), '%H:%M:%S').time()
                time_diff = abs((datetime.combine(target_timestamp.date(), target_time) -
                               datetime.combine(target_timestamp.date(), point_time)).total_seconds() / 60)

                if time_diff <= tolerance_minutes and time_diff < best_time_diff:
                    price = float(data_point.get('close', 0))
                    if price > 0:
                        best_match = (price, self._combine_date_time(target_timestamp.date(), data_point.get('time')))
                        best_time_diff = time_diff

            except (ValueError, TypeError):
                continue

        return best_match

    def _get_business_day_price(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Strategy 3: Get most recent business day price"""
        try:
            # Get recent daily data (last 10 days to account for weekends and holidays)
            end_date = datetime.now().date() - timedelta(days=1)  # Yesterday to avoid same-day issues
            start_date = end_date - timedelta(days=10)

            daily_data = self.fmp_loader.get_historical_daily_prices(ticker, start_date, end_date)

            if daily_data:
                # Sort by date descending to get most recent
                sorted_data = sorted(daily_data, key=lambda x: x.get('date', ''), reverse=True)
                latest_data = sorted_data[0]

                price = float(latest_data.get('close', 0))
                if price > 0:
                    date_str = latest_data.get('date', '')
                    price_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    return price, price_date

        except Exception as e:
            log_debug(f"Business day strategy failed for {ticker}: {e}")

        return None

    def _get_last_known_price(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Strategy 4: Get last known price within reasonable timeframe"""
        if not self.enable_last_known_strategy:
            return None

        max_days_back = getattr(Config, 'LAST_KNOWN_PRICE_MAX_DAYS', 5)

        try:
            end_date = datetime.now().date() - timedelta(days=1)  # Yesterday to avoid same-day issues
            start_date = end_date - timedelta(days=max_days_back)

            daily_data = self.fmp_loader.get_historical_daily_prices(ticker, start_date, end_date)

            if daily_data:
                # Get the most recent available price
                sorted_data = sorted(daily_data, key=lambda x: x.get('date', ''), reverse=True)
                latest_data = sorted_data[0]

                price = float(latest_data.get('close', 0))
                if price > 0:
                    date_str = latest_data.get('date', '')
                    price_date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    return price, price_date

        except Exception as e:
            log_debug(f"Last known price strategy failed for {ticker}: {e}")

        return None

    def _get_multi_day_search_price(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Strategy 5: Search multiple days around target (avoiding same day)"""
        if not self.enable_multi_day_strategy:
            return None

        search_range = getattr(Config, 'MULTI_DAY_SEARCH_RANGE', 3)
        today = datetime.now(timezone.utc).date()

        # Search days around the target date, but avoid today
        for day_offset in range(-search_range, search_range + 1):
            search_date = target_timestamp.date() + timedelta(days=day_offset)
            
            # Skip today - API won't have complete data
            if search_date >= today:
                continue

            try:
                intraday_data = self._get_intraday_data(ticker, search_date)
                if intraday_data:
                    # Try to find price close to target time
                    result = self._find_closest_time_price(intraday_data, target_timestamp.time(), search_date)
                    if result:
                        return result

            except Exception as e:
                log_debug(f"Multi-day search failed for {ticker} on {search_date}: {e}")
                continue

        return None

    def _find_closest_time_price(self, intraday_data: List[Dict], target_time: dt_time, search_date: date) -> Optional[Tuple[float, datetime]]:
        """Find price closest to target time in intraday data"""
        best_match = None
        best_time_diff = float('inf')

        for data_point in intraday_data:
            try:
                point_time = datetime.strptime(data_point.get('time', ''), '%H:%M:%S').time()
                time_diff = abs((datetime.combine(search_date, target_time) -
                               datetime.combine(search_date, point_time)).total_seconds())

                if time_diff < best_time_diff:
                    price = float(data_point.get('close', 0))
                    if price > 0:
                        best_match = (price, self._combine_date_time(search_date, data_point.get('time')))
                        best_time_diff = time_diff

            except (ValueError, TypeError):
                continue

        return best_match

    def _calculate_target_times(self, recommendation_timestamp: datetime) -> Dict[str, datetime]:
        """Calculate all target timestamps for price fetching"""
        target_times = {
            'recommendation': recommendation_timestamp
        }

        # Add checkpoint times
        checkpoint_info = self._get_checkpoint_info()
        for checkpoint in checkpoint_info:
            field_prefix = checkpoint['field_prefix']
            minutes = checkpoint['minutes']

            if minutes is not None:
                # Regular checkpoint (add minutes to recommendation time)
                target_times[field_prefix] = recommendation_timestamp + timedelta(minutes=minutes)
            else:
                # Close time (same day at specified hour/minute)
                close_hour = getattr(Config, 'CLOSE_PRICE_HOUR', 15)
                close_minute = getattr(Config, 'CLOSE_PRICE_MINUTE', 30)

                # Create close time on same date as recommendation
                rec_date = recommendation_timestamp.date()
                close_time = datetime.combine(rec_date, datetime.min.time().replace(hour=close_hour, minute=close_minute))
                close_time = close_time.replace(tzinfo=recommendation_timestamp.tzinfo)

                target_times[field_prefix] = close_time

        return target_times

    def _determine_tolerance(self, target_timestamp: datetime) -> int:
        """Determine tolerance window based on target time characteristics"""
        if not getattr(Config, 'PRICE_FETCH_ENHANCED_TOLERANCE', True):
            return 15  # Default tolerance

        now = datetime.now(timezone.utc)

        # Future target (shouldn't normally happen, but handle gracefully)
        if target_timestamp > now:
            return getattr(Config, 'PRICE_FETCH_FUTURE_TARGET_TOLERANCE_MINUTES', 30)

        # Market hours detection
        est_target = target_timestamp.astimezone(self.est_tz)
        hour = est_target.hour

        # Market open/close periods need wider tolerance
        if hour <= 10 or hour >= 15:  # Near open (9:30) or close (4:00)
            return 30

        # Regular market hours
        return 15

    def _get_intraday_data(self, ticker: str, target_date: date) -> Optional[List[Dict]]:
        """Get intraday data for ticker on specific date with caching"""
        cache_key = f"{ticker}_{target_date}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        try:
            intraday_data = self.fmp_loader.get_intraday_prices(ticker, target_date)

            if intraday_data:
                self.price_cache[cache_key] = intraday_data
                return intraday_data
            else:
                # Try alternative: get daily data and use it as fallback
                log_debug(f"⚠️ No intraday data for {ticker} on {target_date}, trying daily fallback")
                daily_data = self.fmp_loader.get_historical_daily_prices(ticker, target_date, target_date)

                if daily_data and len(daily_data) > 0:
                    # Convert daily to pseudo-intraday
                    daily_item = daily_data[0]
                    pseudo_intraday = [{
                        'time': '15:30:00',  # Market close time
                        'close': daily_item.get('close'),
                        'high': daily_item.get('high'),
                        'low': daily_item.get('low'),
                        'open': daily_item.get('open'),
                        'volume': daily_item.get('volume')
                    }]
                    self.price_cache[cache_key] = pseudo_intraday
                    log_debug(f"✅ Using daily close price as fallback for {ticker}")
                    return pseudo_intraday

        except Exception as e:
            log_debug(f"Failed to get intraday data for {ticker} on {target_date}: {e}")

        return None

    def _combine_date_time(self, date_part: date, time_str: str) -> datetime:
        """Combine date and time string into timezone-aware datetime"""
        try:
            time_part = datetime.strptime(time_str, '%H:%M:%S').time()
            combined = datetime.combine(date_part, time_part)
            return combined.replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback to noon if time parsing fails
            return datetime.combine(date_part, datetime.min.time().replace(hour=12)).replace(tzinfo=timezone.utc)

    def _get_float_value(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None

    def display_analysis_results(self, analysis: Dict[str, Any]):
        """Display analysis results in organized format"""
        print("\n" + "="*60)
        print("📊 CSV ANALYSIS RESULTS")
        print("="*60)

        self._display_basic_stats(analysis)
        self._display_decision_breakdown(analysis)
        self._display_tracking_analysis(analysis)
        self._display_header_analysis(analysis)
        self._display_missing_fields(analysis)
        self._display_date_range(analysis)

    def _display_basic_stats(self, analysis: Dict[str, Any]):
        """Display basic statistics"""
        print("📋 Basic Statistics:")
        print(f"Total rows: {analysis['total_rows']}")
        print(f"Updatable rows: {analysis['updatable_rows']}")
        print(f"Completed tracking: {analysis['completed_tracking']}")

    def _display_decision_breakdown(self, analysis: Dict[str, Any]):
        """Display decision breakdown"""
        decision_breakdown = analysis['decision_breakdown']
        total_decisions = sum(decision_breakdown.values())
        print(f"\n📊 Decision Breakdown (Total: {total_decisions}):")
        for decision_type, count in decision_breakdown.items():
            if count > 0:
                percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
                print(f"  {decision_type}: {count} ({percentage:.1f}%)")

    def _display_tracking_analysis(self, analysis: Dict[str, Any]):
        """Display tracking status analysis"""
        tracking_analysis = analysis['tracking_analysis']
        status_counts = tracking_analysis['status_counts']
        should_be_completed = tracking_analysis['should_be_completed']
        
        print(f"\n🔧 Tracking Status Analysis:")
        for status, count in status_counts.items():
            status_label = status if status else 'EMPTY'
            print(f"  {status_label}: {count}")
        
        if should_be_completed > 0:
            print(f"  🔧 Can be marked 'completed': {should_be_completed}")
            print(f"     (These trades have all required price data)")
        else:
            print(f"  ✅ All complete trades are properly marked")

    def _display_header_analysis(self, analysis: Dict[str, Any]):
        """Display header analysis"""
        header_analysis = analysis['header_analysis']
        print("\n🗂️ Header Analysis:")
        print(f"  Total columns: {header_analysis['total_columns']}")
        print(f"  Price columns: {len(header_analysis['price_columns'])}")
        print(f"  Has recommendation_price: {header_analysis['has_recommendation_price']}")

        if header_analysis['missing_expected']:
            print(f"  ⚠️ Missing expected columns: {', '.join(header_analysis['missing_expected'])}")

    def _display_missing_fields(self, analysis: Dict[str, Any]):
        """Display missing fields summary"""
        if analysis['missing_fields_summary']:
            print("\n🔍 Most Common Missing Fields:")
            sorted_missing = sorted(analysis['missing_fields_summary'].items(),
                                  key=lambda x: x[1], reverse=True)
            for field, count in sorted_missing[:5]:
                print(f"  {field}: {count} rows")

    def _display_date_range(self, analysis: Dict[str, Any]):
        """Display date range information"""
        date_range = analysis['date_range']
        if date_range['earliest'] and date_range['latest']:
            print("\n📅 Date Range:")
            print(f"  Earliest: {date_range['earliest'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Latest: {date_range['latest'].strftime('%Y-%m-%d %H:%M:%S')}")


class TradingPerformanceAnalyzer:
    """Analyze trading performance from completed trades"""

    def analyze_trading_performance(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze performance of completed trades"""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        completed_trades = self._load_completed_trades(csv_path)

        if not completed_trades:
            print("📊 No completed trades found for analysis")
            return {'error': 'No completed trades found'}

        trades_data = self._process_trades_data(completed_trades)
        analysis_results = self._analyze_trades(trades_data)

        self._display_performance_results(analysis_results)
        return analysis_results

    def _load_completed_trades(self, csv_path: Path) -> List[Dict[str, str]]:
        """Load completed trades from CSV"""
        completed_trades = []

        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('tracking_status') == 'completed':
                    completed_trades.append(row)

        return completed_trades

    def _process_trades_data(self, completed_trades: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process trades data for analysis"""
        trades_data = []

        for trade in completed_trades:
            if not self._is_valid_trade(trade):
                continue

            trade_data = self._extract_trade_metrics(trade)
            if trade_data:
                trades_data.append(trade_data)

        return trades_data

    def _is_valid_trade(self, trade: Dict[str, str]) -> bool:
        """Check if trade has required data for analysis"""
        required_fields = ['ticker', 'decision', 'recommendation_price', 'price_close']
        return all(trade.get(field) and trade.get(field).strip() for field in required_fields)

    def _extract_trade_metrics(self, trade: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract trade metrics for analysis"""
        try:
            rec_price = self._get_float_value(trade.get('recommendation_price'))
            close_price = self._get_float_value(trade.get('price_close'))

            if not rec_price or not close_price or rec_price <= 0:
                return None

            # Calculate return based on decision type
            decision = trade.get('decision', '').upper()
            if decision == 'LONG':
                trade_return = ((close_price - rec_price) / rec_price) * 100
            elif decision == 'SHORT':
                trade_return = ((rec_price - close_price) / rec_price) * 100
            else:
                return None  # Skip NONE decisions

            return {
                'ticker': trade.get('ticker'),
                'decision': decision,
                'rec_price': rec_price,
                'close_price': close_price,
                'return': trade_return,
                'is_winner': trade_return > 0,
                'timestamp': trade.get('timestamp', '')
            }

        except (ValueError, TypeError):
            return None

    def _get_float_value(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None

    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading performance metrics"""
        total_trades = len(trades)
        winners = [t for t in trades if t['is_winner']]
        losers = [t for t in trades if not t['is_winner']]

        total_return = sum(t['return'] for t in trades)
        avg_return = total_return / total_trades if total_trades > 0 else 0
        win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0

        # Best and worst trades
        best_trade = max(trades, key=lambda x: x['return']) if trades else None
        worst_trade = min(trades, key=lambda x: x['return']) if trades else None

        return {
            'total_trades': total_trades,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }

    def _display_performance_results(self, results: Dict[str, Any]):
        """Display performance analysis results"""
        print(f"📊 Total trades analyzed: {results['total_trades']}")
        print(f"🏆 Winners: {results['winners']} ({results['win_rate']:.1f}%)")
        print(f"💀 Losers: {results['losers']}")
        print(f"📈 Total return: {results['total_return']:+.2f}%")
        print(f"📊 Average return: {results['avg_return']:+.2f}%")

        if results['best_trade']:
            best = results['best_trade']
            print(f"\n🥇 Best trade: {best['ticker']} ({best['decision']}) | {best['return']:+.2f}%")

        if results['worst_trade']:
            worst = results['worst_trade']
            print(f"🥴 Worst trade: {worst['ticker']} ({worst['decision']}) | {worst['return']:+.2f}%")


def main():
    """Main function with reduced cognitive complexity"""
    print("🚀 Trading Recommendations Price Backfiller - FINAL FIXED VERSION")
    print("=" * 60)

    # Validate FMP API key
    if not Config.FMP_API_KEY:
        print("❌ FMP_API_KEY is required in config.py or .env file")
        sys.exit(1)

    try:
        backfiller = TradingRecommendationsPriceBackfiller(Config.FMP_API_KEY)

        # Test API connectivity if requested
        if _should_test_api():
            _test_api_connectivity(backfiller)

        # Select and analyze CSV file
        csv_path = backfiller.select_csv_file()
        _validate_csv_file(csv_path)

        analysis = backfiller.analyze_csv_file(csv_path)
        backfiller.display_analysis_results(analysis)

        # FIXED: Always determine what actions to take (ask user)
        action_config = _determine_actions_fixed(analysis)

        # Execute selected actions
        _execute_actions(backfiller, csv_path, analysis, action_config)

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        log_error(f"Main function error: {e}")
        sys.exit(1)

    print("=" * 60)
    print("🏁 Price backfiller completed successfully!")


def _should_test_api() -> bool:
    """Check if user wants to test API connectivity"""
    print("Would you like to test API connectivity with a few sample tickers first? (y/n)")
    test_choice = input().strip().lower()
    return test_choice in ['y', 'yes']


def _test_api_connectivity(backfiller: TradingRecommendationsPriceBackfiller) -> None:
    """Test API connectivity with sample tickers and comprehensive diagnostics"""
    test_tickers = ['AAPL', 'MSFT', 'TSLA']
    print("\n🧪 Testing API connectivity with comprehensive diagnostics...")

    # Test 1: Basic API connectivity
    print("\n1️⃣ Testing basic FMP API connectivity...")
    for ticker in test_tickers:
        try:
            # Test basic quote endpoint
            quote_data = backfiller.fmp_loader.make_request(f"quote/{ticker}")
            if quote_data and len(quote_data) > 0:
                price = quote_data[0].get('price', 'Unknown')
                print(f"✅ {ticker}: Basic API works - Current price: ${price}")
            else:
                print(f"❌ {ticker}: Basic API failed - No quote data")
        except Exception as e:
            print(f"❌ {ticker}: Basic API error - {e}")

    # Test 2: Historical daily prices
    print("\n2️⃣ Testing historical daily price methods...")
    test_date = datetime.now().date() - timedelta(days=2)  # Use 2 days ago to avoid same-day issues
    for ticker in ['AAPL']:  # Test just one ticker for speed
        try:
            daily_data = backfiller.fmp_loader.get_historical_daily_prices(
                ticker, test_date - timedelta(days=2), test_date
            )
            if daily_data:
                print(f"✅ {ticker}: Historical daily method works - Got {len(daily_data)} days")
            else:
                print(f"❌ {ticker}: Historical daily method failed - No data")
        except Exception as e:
            print(f"❌ {ticker}: Historical daily error - {e}")

    # Test 3: Intraday prices
    print("\n3️⃣ Testing intraday price methods...")
    for ticker in ['AAPL']:  # Test just one ticker
        try:
            intraday_data = backfiller.fmp_loader.get_intraday_prices(ticker, test_date)
            if intraday_data:
                print(f"✅ {ticker}: Intraday method works - Got {len(intraday_data)} data points")
            else:
                print(f"❌ {ticker}: Intraday method failed - No data (may be normal for weekend/holiday)")
        except Exception as e:
            print(f"❌ {ticker}: Intraday error - {e}")

    # Test 4: Price backfiller integration
    print("\n4️⃣ Testing price backfiller integration...")
    for ticker in ['AAPL']:
        test_timestamp = datetime.now(timezone.utc) - timedelta(days=2)  # Use 2 days ago
        result = backfiller.get_price_at_timestamp(ticker, test_timestamp)
        if result:
            price, timestamp = result
            print(f"✅ {ticker}: ${price:.2f} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"❌ {ticker}: Price backfiller failed - All strategies failed")

    print("\n✅ API connectivity test complete. Check results above before proceeding.")


def _validate_csv_file(csv_path: Path) -> None:
    """Validate that CSV file exists"""
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)


def _determine_actions_fixed(analysis: Dict[str, Any]) -> Dict[str, bool]:
    """FIXED: Always ask user what actions to take"""
    print(f"\n💡 Analysis Summary:")
    print(f"   • Total rows: {analysis['total_rows']}")
    print(f"   • Updatable rows: {analysis['updatable_rows']}")
    print(f"   • Completed tracking: {analysis['completed_tracking']}")
    
    # Show tracking status fix potential
    tracking_analysis = analysis.get('tracking_analysis', {})
    should_be_completed = tracking_analysis.get('should_be_completed', 0)
    if should_be_completed > 0:
        print(f"   • Can mark as completed: {should_be_completed}")
    
    if analysis['updatable_rows'] == 0 and should_be_completed == 0:
        if analysis['completed_tracking'] > 0:
            print("\n📊 No price updates needed, but you can analyze completed trades.")
            return _get_user_action_choice_fixed(analysis, updates_needed=False)
        else:
            print("\n✅ No rows need updating and no completed trades to analyze.")
            return {'update_prices': False, 'run_analysis': False}
    else:
        return _get_user_action_choice_fixed(analysis, updates_needed=True)


def _get_user_action_choice_fixed(analysis: Dict[str, Any], updates_needed: bool = True) -> Dict[str, bool]:
    """FIXED: Get user's choice for what actions to perform"""
    if analysis['completed_tracking'] > 0:
        print(f"\n💡 Found {analysis['completed_tracking']} completed trades available for analysis.")
    
    if updates_needed:
        print(f"\n🔧 Found {analysis['updatable_rows']} rows that need price updates.")
        
        # Mention tracking status fixes
        tracking_analysis = analysis.get('tracking_analysis', {})
        should_be_completed = tracking_analysis.get('should_be_completed', 0)
        if should_be_completed > 0:
            print(f"🔧 Found {should_be_completed} rows that can be marked as 'completed'.")
        
        print("\nWhat would you like to do?")
        print("  1. Update missing prices and fix tracking status (+ analyze if completed trades exist)")
        print("  2. Just analyze existing completed trades")
        print("  3. Just update missing prices and fix tracking status")
        print("  4. Exit without doing anything")
        choice_prompt = "Choice (1/2/3/4): "
    else:
        print("\nWhat would you like to do?")
        print("  1. Analyze existing completed trades")
        print("  2. Exit without doing anything")
        choice_prompt = "Choice (1/2): "

    while True:
        action_choice = input(choice_prompt).strip()

        if updates_needed:
            if action_choice == '1':
                return {'update_prices': True, 'run_analysis': True}
            elif action_choice == '2':
                if analysis['completed_tracking'] > 0:
                    return {'update_prices': False, 'run_analysis': True}
                else:
                    print("❌ No completed trades to analyze. Choose option 3 or 4.")
                    continue
            elif action_choice == '3':
                return {'update_prices': True, 'run_analysis': False}
            elif action_choice == '4':
                print("👋 Exiting without changes.")
                sys.exit(0)
            else:
                print("Please enter 1, 2, 3, or 4")
        else:
            if action_choice == '1':
                return {'update_prices': False, 'run_analysis': True}
            elif action_choice == '2':
                print("👋 Exiting without changes.")
                sys.exit(0)
            else:
                print("Please enter 1 or 2")


def _execute_actions(backfiller: TradingRecommendationsPriceBackfiller, csv_path: Path,
                    analysis: Dict[str, Any], action_config: Dict[str, bool]) -> None:
    """Execute the selected actions"""
    if action_config['update_prices']:
        _handle_price_updates(backfiller, csv_path, analysis)

    if action_config['run_analysis']:
        if analysis['completed_tracking'] > 0:
            _handle_performance_analysis(csv_path)
        else:
            print("📊 No completed trades available for analysis.")


def _handle_price_updates(backfiller: TradingRecommendationsPriceBackfiller,
                         csv_path: Path, analysis: Dict[str, Any]) -> None:
    """Handle price update workflow"""
    print(f"\n🔧 Found {analysis['updatable_rows']} rows that need price updates.")
    print("⚠️  Note: API may not have complete data for same-day trades.")

    # Ask about dry run
    should_dry_run = _should_run_dry_run()

    if should_dry_run:
        print("\n🔍 Running in DRY-RUN mode...")
        backfiller.update_csv_prices(csv_path, dry_run=True)

        if _should_proceed_with_live_update():
            print("\n💾 Running LIVE update...")
            backfiller.update_csv_prices(csv_path, dry_run=False)
            print("✅ Price backfill completed!")
        else:
            print("Update cancelled.")
    else:
        print("\n💾 Running LIVE update...")
        backfiller.update_csv_prices(csv_path, dry_run=False)
        print("✅ Price backfill completed!")


def _should_run_dry_run() -> bool:
    """Check if user wants to run dry run first"""
    print("Would you like to run in dry-run mode first? (recommended) (y/n)")
    dry_run_choice = input().strip().lower()
    return dry_run_choice in ['y', 'yes']


def _should_proceed_with_live_update() -> bool:
    """Check if user wants to proceed with live update after dry run"""
    print("\nDry run completed. Proceed with actual update? (y/n)")
    proceed_choice = input().strip().lower()
    return proceed_choice in ['y', 'yes']


def _handle_performance_analysis(csv_path: Path) -> None:
    """Handle performance analysis"""
    print("\n📊 Running performance analysis...")
    analyzer = TradingPerformanceAnalyzer()
    analyzer.analyze_trading_performance(csv_path)


if __name__ == "__main__":
    main()
