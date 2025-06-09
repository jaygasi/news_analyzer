"""
Trading Recommendations Price Backfiller - UPDATED VERSION
Complete solution for updating missing price data in trading_decisions.csv

ENHANCEMENTS:
- Dynamic CSV header compatibility 
- Processes ALL rows (LONG/SHORT/NONE)
- Enhanced diagnostics for None values
- Configuration validation
- Better error handling and recovery
- Decision type breakdown reporting

Python 3.13.3 compatible
"""
import sys
import csv
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import pytz

# Add parent directory to sys.path to allow importing project modules
sys.path.append(str(Path(__file__).parent.parent)) 

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
    """Complete price backfiller for trading recommendations CSV with dynamic header support"""
    
    def __init__(self, fmp_api_key: str):
        """Initialize the price backfiller with configuration validation"""
        self.fmp_loader = BaseFMPLoader(fmp_api_key)
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.timezone('UTC')
        
        # Cache to avoid duplicate API calls
        self.price_cache: Dict[str, List[Dict]] = {}
        
        # Add future buffer for price fetching
        self.future_buffer_minutes = getattr(Config, 'PRICE_FETCH_FUTURE_BUFFER_MINUTES', 30)
        
        # VALIDATE configuration compatibility
        self._validate_configuration()
        
        log_info("🔄 Trading Recommendations Price Backfiller initialized")

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
        log_info(f"📊 Price Tracking Configuration:")
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
        
        # Look for CSV files in common locations
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
        
        if not possible_files:
            print("❌ No CSV files found in current or output directories.")
            print("Please enter the full path to your CSV file:")
            file_path = input("CSV file path: ").strip()
            return Path(file_path)
        
        print("Found the following CSV files:")
        for i, file_path in enumerate(possible_files, 1):
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  {i}. {file_path} ({file_size:.1f} KB)")
        
        print("  0. Enter custom path")
        
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(possible_files)}) or 0 for custom: ").strip()
                
                if choice == "0":
                    file_path = input("Enter full path to CSV file: ").strip()
                    return Path(file_path)
                
                choice_int = int(choice)
                if 1 <= choice_int <= len(possible_files):
                    selected_file = possible_files[choice_int - 1]
                    print(f"✅ Selected: {selected_file}")
                    return selected_file
                else:
                    print(f"❌ Please enter a number between 1 and {len(possible_files)}")
            
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled by user")
                sys.exit(0)
    
    def analyze_csv_file(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze the CSV file to understand what needs to be updated - DYNAMIC VERSION"""
        print(f"\n📊 Analyzing {csv_path.name}...")
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        analysis = {
            'total_rows': 0,
            'missing_recommendation_price': 0,
            'missing_checkpoint1': 0,
            'missing_checkpoint2': 0,
            'missing_close': 0,
            'completed_tracking': 0,
            'pending_tracking': 0,
            'updatable_rows': 0,
            'sample_rows': [],
            # NEW: Decision type tracking
            'decision_types': {'LONG': 0, 'SHORT': 0, 'NONE': 0, 'OTHER': 0},
            'updatable_by_decision': {'LONG': 0, 'SHORT': 0, 'NONE': 0, 'OTHER': 0}
        }
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                
                if not headers:
                    raise ValueError("CSV file appears to be empty or invalid")
                
                # Build dynamic required columns based on Config
                base_required_columns = [
                    'ticker', 'recommendation_timestamp', 'tracking_status', 'decision',
                    'recommendation_price'
                ]
                
                # Add dynamic checkpoint columns based on configuration
                checkpoint_info = self._get_checkpoint_info()
                dynamic_columns = []
                for checkpoint in checkpoint_info:
                    field_prefix = checkpoint['field_prefix']
                    dynamic_columns.extend([
                        field_prefix,
                        f"{field_prefix}_timestamp", 
                        f"{field_prefix}_change_pct"
                    ])
                
                required_columns = base_required_columns + dynamic_columns
                
                missing_columns = [col for col in required_columns if col not in headers]
                if missing_columns:
                    print(f"⚠️ Missing expected columns: {missing_columns}")
                    print(f"📋 Available columns: {list(headers)}")
                    log_warning(f"Some expected columns missing: {missing_columns}")
                    print("   Continuing with available columns...")
                
                for row_num, row in enumerate(reader):
                    analysis['total_rows'] += 1
                    
                    # Track decision types
                    decision_type = str(row.get('decision', 'OTHER')).strip().upper()
                    if decision_type not in analysis['decision_types']:
                        decision_type = 'OTHER'
                    analysis['decision_types'][decision_type] += 1
                    
                    # Check tracking status
                    tracking_status = str(row.get('tracking_status', '')).strip().lower()
                    if tracking_status == 'completed':
                        analysis['completed_tracking'] += 1
                        continue
                    elif tracking_status == 'pending':
                        analysis['pending_tracking'] += 1
                    
                    # Check for missing prices - DYNAMIC VERSION
                    rec_price = str(row.get('recommendation_price', '')).strip()
                    has_missing = False
                    
                    if not rec_price or rec_price.lower() in ['none', '', '0', '0.0', '0.00']:
                        analysis['missing_recommendation_price'] += 1
                        has_missing = True
                    
                    # Dynamic checkpoint processing
                    checkpoint_info = self._get_checkpoint_info()
                    missing_checkpoints = []
                    for checkpoint in checkpoint_info:
                        field_name = checkpoint['field_prefix']
                        field_value = str(row.get(field_name, '')).strip()
                        
                        if not field_value or field_value.lower() in ['none', '', '0', '0.0', '0.00']:
                            missing_checkpoints.append(field_name)
                            has_missing = True
                            
                            # Update analysis counters dynamically
                            if field_name == 'price_checkpoint1':
                                analysis['missing_checkpoint1'] += 1
                            elif field_name == 'price_checkpoint2':
                                analysis['missing_checkpoint2'] += 1
                            elif field_name == 'price_close':
                                analysis['missing_close'] += 1
                    
                    if has_missing:
                        analysis['updatable_rows'] += 1
                        
                        # Track updatable rows by decision type
                        analysis['updatable_by_decision'][decision_type] += 1
                        
                        # Store sample for display
                        if len(analysis['sample_rows']) < 5:
                            analysis['sample_rows'].append({
                                'row': row_num + 2,  # +2 for header and 0-indexing
                                'ticker': row.get('ticker', 'N/A'),
                                'decision': decision_type,
                                'timestamp': row.get('recommendation_timestamp', 'N/A'),
                                'rec_price': rec_price or 'MISSING',
                                'missing_fields': missing_checkpoints
                            })
        
        except Exception as e:
            log_error(f"Error analyzing CSV: {e}")
            raise
        
        return analysis
    
    def _get_checkpoint_info(self) -> List[Dict[str, str]]:
        """Get checkpoint information dynamically from Config or fallback"""
        try:
            if hasattr(Config, 'get_checkpoint_info'):
                return Config.get_checkpoint_info()
        except:
            pass
        
        # Fallback to standard checkpoints
        return [
            {'field_prefix': 'price_checkpoint1', 'label': 'Checkpoint 1'}, 
            {'field_prefix': 'price_checkpoint2', 'label': 'Checkpoint 2'}, 
            {'field_prefix': 'price_close', 'label': 'Close Price'}
        ]
    
    def display_analysis_results(self, analysis: Dict[str, Any]):
        """Display analysis results to user with decision type breakdown"""
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        
        print(f"📄 Total rows: {analysis['total_rows']}")
        
        # Decision type breakdown
        print(f"\n📈 Decision Type Breakdown:")
        for decision_type, count in analysis['decision_types'].items():
            if count > 0:
                print(f"  {decision_type}: {count} decisions")
        
        print(f"\n📊 Price Data Status:")
        print(f"  Missing recommendation price: {analysis['missing_recommendation_price']}")
        print(f"  Missing checkpoint 1: {analysis['missing_checkpoint1']}")
        print(f"  Missing checkpoint 2: {analysis['missing_checkpoint2']}")
        print(f"  Missing close price: {analysis['missing_close']}")
        
        print(f"\n🔄 Tracking Status:")
        print(f"  Completed tracking: {analysis['completed_tracking']}")
        print(f"  Pending tracking: {analysis['pending_tracking']}")
        
        print(f"\n🎯 Updatable Rows: {analysis['updatable_rows']}")
        
        # Updatable breakdown by decision type
        if analysis.get('updatable_by_decision'):
            print(f"  Updatable by decision type:")
            for decision_type, count in analysis['updatable_by_decision'].items():
                if count > 0:
                    print(f"    {decision_type}: {count} rows need price updates")
        
        if analysis['sample_rows']:
            print(f"\nSample rows needing updates:")
            for sample in analysis['sample_rows']:
                missing_str = ', '.join(sample['missing_fields']) if sample['missing_fields'] else 'Rec price only'
                print(f"  Row {sample['row']}: {sample['ticker']} ({sample['decision']}) - Missing: {missing_str}")
        
        print("\n" + "="*60)
    
    def get_price_at_timestamp(self, ticker: str, target_timestamp: datetime, 
                              interval: str = '5min') -> Optional[Tuple[float, datetime]]:
        """Get historical price at specific timestamp with enhanced error recovery"""
        try:
            # ADD ticker validation
            if not ticker or len(ticker.strip()) == 0:
                log_warning(f"❌ Invalid ticker provided: '{ticker}'")
                return None
                
            # Clean ticker symbol
            ticker = ticker.strip().upper()
            
            # Skip OTC tickers that commonly fail
            if ticker.endswith(('F', 'FF')):
                log_debug(f"⚠️ OTC ticker {ticker} - may have limited data")
            
            # Skip fetching if target is in the future beyond buffer
            now_utc = datetime.now(timezone.utc)
            if target_timestamp > now_utc + timedelta(minutes=self.future_buffer_minutes):
                log_debug(f"⏰ Skipping {ticker} price fetch - target is in future")
                return None
            
            target_date = target_timestamp.date()
            
            # Strategy 1: Try intraday data
            result = self._try_intraday_data(ticker, target_timestamp, target_date, interval)
            if result:
                return result
            
            # Strategy 2: Try daily historical data
            result = self._try_daily_data(ticker, target_timestamp, target_date)
            if result:
                return result
            
            # Strategy 3: Try current/recent quote (for very recent timestamps)
            if (datetime.now(timezone.utc) - target_timestamp).days <= 1:
                result = self._try_current_quote(ticker, target_timestamp)
                if result:
                    return result
            
            # ADD enhanced error logging
            log_error(f"💥 All price fetch strategies failed for {ticker}")
            log_debug(f"   Target: {target_timestamp}")
            log_debug(f"   This will create 'None' value in CSV")
            log_debug(f"   Suggestions: Check ticker symbol, verify market hours, confirm FMP access")
            
            return None
            
        except Exception as e:
            log_error(f"💥 Exception in price fetch for {ticker}: {e}")
            return None
    
    def _try_intraday_data(self, ticker: str, target_timestamp: datetime, 
                          target_date, interval: str) -> Optional[Tuple[float, datetime]]:
        """Try to get intraday price data"""
        try:
            # Use cache if available
            cache_key = f"{ticker}_{target_date}_{interval}"
            if cache_key not in self.price_cache:
                date_str = target_date.strftime('%Y-%m-%d')
                endpoint = f"historical-chart/{interval}/{ticker}"
                params = {"from": date_str, "to": date_str}
                
                data = self.fmp_loader.make_request(endpoint, params)
                
                if data and isinstance(data, list):
                    # Parse and store in cache
                    parsed_data = []
                    for item in data:
                        if item.get('date') and item.get('close'):
                            try:
                                # Parse datetime
                                dt_str = item['date']
                                if 'T' in dt_str:
                                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                                else:
                                    dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                                    dt = dt.replace(tzinfo=timezone.utc)
                                
                                parsed_data.append({
                                    'timestamp': dt,
                                    'price': float(item['close'])
                                })
                            except (ValueError, TypeError) as e:
                                log_debug(f"Error parsing intraday data for {ticker}: {e}")
                                continue
                    
                    self.price_cache[cache_key] = parsed_data
                else:
                    self.price_cache[cache_key] = []
            
            historical_data = self.price_cache[cache_key]
            closest_data = self._find_closest_price(historical_data, target_timestamp)
            
            if closest_data:
                return closest_data['price'], closest_data['timestamp']
            
            return None
            
        except Exception as e:
            log_error(f"Error getting intraday data for {ticker}: {e}")
            return None
    
    def _try_daily_data(self, ticker: str, target_timestamp: datetime, target_date) -> Optional[Tuple[float, datetime]]:
        """Try to get daily historical data"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Try historical-price-full endpoint
            endpoint = f"historical-price-full/{ticker}"
            params = {"from": date_str, "to": date_str}
            
            data = self.fmp_loader.make_request(endpoint, params)
            
            if data and isinstance(data, dict) and 'historical' in data:
                historical = data['historical']
                if historical and isinstance(historical, list):
                    for item in historical:
                        if item.get('date') == date_str:
                            close_price = float(item.get('close', 0))
                            if close_price > 0:
                                # Use close time for daily data
                                close_dt = datetime.strptime(f"{date_str} 16:00:00", '%Y-%m-%d %H:%M:%S')
                                close_dt = self.est_tz.localize(close_dt).astimezone(timezone.utc)
                                log_debug(f"Found daily price for {ticker}: ${close_price:.2f}")
                                return close_price, close_dt
            
            # Fallback: try simple historical endpoint
            simple_endpoint = f"historical-price/{ticker}"
            simple_data = self.fmp_loader.make_request(simple_endpoint)
            
            if simple_data and isinstance(simple_data, list) and simple_data:
                historical_item = simple_data[0]
                if historical_item.get('date') == date_str:
                    close_price = float(historical_item.get('close', 0))
                    if close_price > 0:
                        close_dt = datetime.strptime(f"{date_str} 16:00:00", '%Y-%m-%d %H:%M:%S')
                        close_dt = self.est_tz.localize(close_dt).astimezone(timezone.utc)
                        log_debug(f"Found historical price for {ticker}: ${close_price:.2f}")
                        return close_price, close_dt
            
            return None
            
        except Exception as e:
            log_error(f"Error getting daily price for {ticker}: {e}")
            return None
    
    def _try_current_quote(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Try to get current quote as last resort"""
        try:
            # Try quote endpoint
            endpoint = f"quote/{ticker}"
            data = self.fmp_loader.make_request(endpoint)
            
            if data and isinstance(data, list) and data:
                quote = data[0]
                price = float(quote.get('price', 0))
                if price > 0:
                    log_debug(f"Found current quote for {ticker}: ${price:.2f}")
                    return price, datetime.now(timezone.utc)
            
            # Fallback: try quote-short endpoint
            short_endpoint = f"quote-short/{ticker}"
            short_data = self.fmp_loader.make_request(short_endpoint)
            
            if short_data and isinstance(short_data, list) and short_data:
                quote = short_data[0]
                price = float(quote.get('price', 0))
                if price > 0:
                    log_debug(f"Found short quote for {ticker}: ${price:.2f}")
                    return price, datetime.now(timezone.utc)
            
            return None
            
        except Exception as e:
            log_error(f"Error getting current quote for {ticker}: {e}")
            return None
    
    def _find_closest_price(self, historical_data: List[Dict], target_timestamp: datetime) -> Optional[Dict]:
        """Find the closest price point to target timestamp"""
        if not historical_data:
            return None
        
        closest_data = None
        min_time_diff = float('inf')
        
        for data_point in historical_data:
            time_diff = abs((data_point['timestamp'] - target_timestamp).total_seconds())
            
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_data = data_point
        
        # Determine tolerance based on market hours
        target_est = target_timestamp.astimezone(self.est_tz)
        
        if self._is_during_market_hours(target_est):
            max_tolerance = 2700  # 45 minutes during market hours
        else:
            max_tolerance = 7200  # 2 hours outside market hours
        
        if min_time_diff <= max_tolerance:
            log_debug(f"Found price {min_time_diff/60:.1f} minutes from target")
            return closest_data
        else:
            log_debug(f"Closest price {min_time_diff/60:.1f}min away, exceeds tolerance")
            return None
    
    def _is_during_market_hours(self, dt_est: datetime) -> bool:
        """Check if datetime (EST) is during market hours"""
        if dt_est.weekday() >= 5:  # Weekend
            return False
        
        # Market hours: 9:30 AM - 4:00 PM EST
        return (dt_est.hour == 9 and dt_est.minute >= 30) or \
               (10 <= dt_est.hour <= 15) or \
               (dt_est.hour == 16 and dt_est.minute == 0)
    
    def _calculate_target_times(self, recommendation_timestamp) -> Dict[str, datetime]:
        """Calculate target times for all checkpoints"""
        # Parse recommendation timestamp
        if isinstance(recommendation_timestamp, str):
            rec_time = datetime.fromisoformat(recommendation_timestamp.replace('Z', '+00:00'))
        else:
            rec_time = recommendation_timestamp
        
        if rec_time.tzinfo is None:
            rec_time = rec_time.replace(tzinfo=timezone.utc)
        
        targets = {
            'recommendation': rec_time,
            'checkpoint1': rec_time + timedelta(minutes=Config.PRICE_CHECK_1_MINUTES),
            'checkpoint2': rec_time + timedelta(minutes=Config.PRICE_CHECK_2_MINUTES)
        }
        
        # Calculate close time
        rec_time_est = rec_time.astimezone(self.est_tz)
        
        if self._is_during_market_hours(rec_time_est):
            # Same day close
            close_time_est = rec_time_est.replace(
                hour=Config.CLOSE_PRICE_HOUR, 
                minute=Config.CLOSE_PRICE_MINUTE, 
                second=0, 
                microsecond=0
            )
        else:
            # Next trading day close
            next_day = rec_time_est + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)
            
            close_time_est = next_day.replace(
                hour=Config.CLOSE_PRICE_HOUR, 
                minute=Config.CLOSE_PRICE_MINUTE, 
                second=0, 
                microsecond=0
            )
        
        targets['close'] = close_time_est.astimezone(timezone.utc)
        
        return targets
    
    def update_csv_file(self, csv_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Update the CSV file with missing prices"""
        results = {
            'processed_rows': 0,
            'updated_rows': 0,
            'completed_rows': 0,
            'skipped_rows': 0,
            'error_rows': 0,
            'api_calls_made': 0,
            'successful_price_fetches': 0,
            'failed_tickers': []
        }
        
        if not dry_run:
            # Create backup
            backup_path = csv_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            shutil.copy2(csv_path, backup_path)
            log_info(f"📋 Created backup: {backup_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                rows = list(reader)
            
            for i, row in enumerate(rows):
                results['processed_rows'] += 1
                
                # Skip completed rows
                if str(row.get('tracking_status', '')).strip().lower() == 'completed':
                    results['skipped_rows'] += 1
                    continue
                
                try:
                    updated_row = self._update_row_prices(row, results, dry_run)
                    rows[i] = updated_row
                    
                    # Check if row is now complete
                    if self._is_row_complete(updated_row):
                        updated_row['tracking_status'] = 'completed'
                        results['completed_rows'] += 1
                    
                    results['updated_rows'] += 1
                    
                except Exception as e:
                    log_error(f"Error updating row {i+1}: {e}")
                    results['error_rows'] += 1
            
            # Write updated CSV if not dry run
            if not dry_run and results['updated_rows'] > 0:
                with open(csv_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                
                log_info(f"💾 Updated CSV saved: {csv_path}")
        
        except Exception as e:
            log_error(f"Error updating CSV: {e}")
            raise
        
        return results
    
    def _update_row_prices(self, row: Dict[str, str], results: Dict[str, Any], dry_run: bool = False) -> Dict[str, str]:
        """Update missing prices for a single row with enhanced diagnostics"""
        updated_row = row.copy()
        ticker = row.get('ticker', '')
        
        if not ticker:
            return updated_row
        
        log_info(f"🔄 Processing {ticker}...")
        
        # Calculate target times
        try:
            recommendation_timestamp = row.get('recommendation_timestamp', '')
            if not recommendation_timestamp:
                log_warning(f"No recommendation timestamp for {ticker}")
                return updated_row
            
            targets = self._calculate_target_times(recommendation_timestamp)
            
        except Exception as e:
            log_error(f"Error calculating target times for {ticker}: {e}")
            return updated_row
        
        # Track API calls for this ticker
        ticker_api_calls = 0
        ticker_successful_fetches = 0
        
        # Update recommendation price
        rec_price = str(row.get('recommendation_price', '')).strip()
        if not rec_price or rec_price.lower() in ['none', '', '0', '0.0', '0.00']:
            if not dry_run:
                price_data = self.get_price_at_timestamp(ticker, targets['recommendation'])
                ticker_api_calls += 1
                if price_data:
                    price, actual_timestamp = price_data
                    updated_row['recommendation_price'] = f"{price:.2f}"
                    updated_row['recommendation_timestamp'] = actual_timestamp.isoformat()
                    ticker_successful_fetches += 1
                    log_info(f"📈 {ticker} recommendation price: ${price:.2f}")
                else:
                    log_warning(f"❌ {ticker} recommendation price fetch failed")
                    log_debug(f"   Target time: {targets['recommendation']}")
                    log_debug(f"   This will result in 'None' in CSV")
            else:
                log_info(f"🔍 DRY RUN: Would update {ticker} recommendation price")
        
        # Update dynamic checkpoints
        checkpoint_info = self._get_checkpoint_info()
        for checkpoint in checkpoint_info:
            field_name = checkpoint['field_prefix']
            target_key = None
            
            # Map field names to target keys
            if field_name == 'price_checkpoint1':
                target_key = 'checkpoint1'
            elif field_name == 'price_checkpoint2':
                target_key = 'checkpoint2'
            elif field_name == 'price_close':
                target_key = 'close'
            
            if target_key and target_key in targets:
                current_value = str(row.get(field_name, '')).strip()
                if not current_value or current_value.lower() in ['none', '', '0', '0.0', '0.00']:
                    if not dry_run:
                        price_data = self.get_price_at_timestamp(ticker, targets[target_key])
                        ticker_api_calls += 1
                        if price_data:
                            price, actual_timestamp = price_data
                            updated_row[field_name] = f"{price:.2f}"
                            updated_row[f"{field_name}_timestamp"] = actual_timestamp.isoformat()
                            
                            # Calculate percentage change
                            rec_price_val = self._get_float_value(updated_row.get('recommendation_price', ''))
                            if rec_price_val and rec_price_val > 0:
                                change_pct = ((price - rec_price_val) / rec_price_val) * 100
                                updated_row[f"{field_name}_change_pct"] = f"{change_pct:.2f}"
                            
                            ticker_successful_fetches += 1
                            log_info(f"📈 {ticker} {field_name}: ${price:.2f}")
                        else:
                            log_warning(f"❌ {ticker} {field_name} fetch failed")
                    else:
                        log_info(f"🔍 DRY RUN: Would update {ticker} {field_name}")
        
        # Update results tracking
        results['api_calls_made'] += ticker_api_calls
        results['successful_price_fetches'] += ticker_successful_fetches
        
        if ticker_successful_fetches == 0 and ticker_api_calls > 0:
            results['failed_tickers'].append(ticker)
            log_warning(f"❌ No prices found for {ticker} despite {ticker_api_calls} API attempts")
        elif ticker_successful_fetches > 0:
            log_info(f"✅ {ticker}: {ticker_successful_fetches}/{ticker_api_calls} successful price fetches")
        
        # Small delay to respect API rate limits
        if ticker_api_calls > 0 and not dry_run:
            time.sleep(0.2)
        
        return updated_row
    
    def _get_float_value(self, value_str: str) -> Optional[float]:
        """Safely convert string to float"""
        try:
            if not value_str or value_str.strip().lower() in ['none', '', '0', '0.0', '0.00']:
                return None
            return float(value_str.strip())
        except (ValueError, TypeError):
            return None
    
    def _is_row_complete(self, row: Dict[str, str]) -> bool:
        """Check if all required price fields are populated"""
        required_fields = ['recommendation_price']
        
        # Add dynamic checkpoint fields
        checkpoint_info = self._get_checkpoint_info()
        for checkpoint in checkpoint_info:
            required_fields.append(checkpoint['field_prefix'])
        
        for field in required_fields:
            value = self._get_float_value(row.get(field, ''))
            if value is None or value <= 0:
                return False
        
        return True
    
    def display_results(self, results: Dict[str, Any]):
        """Display update results with enhanced diagnostics"""
        print("\n" + "="*60)
        print("📊 UPDATE RESULTS")
        print("="*60)
        print(f"Processed rows: {results['processed_rows']}")
        print(f"Updated rows: {results['updated_rows']}")
        print(f"Completed rows: {results['completed_rows']}")
        print(f"Skipped rows: {results['skipped_rows']}")
        print(f"Error rows: {results['error_rows']}")
        print(f"API calls made: {results['api_calls_made']}")
        print(f"Successful price fetches: {results['successful_price_fetches']}")
        
        if results['api_calls_made'] > 0:
            success_rate = (results['successful_price_fetches'] / results['api_calls_made']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # Show failed tickers if any
        if results.get('failed_tickers'):
            print(f"\n💭 Failed tickers ({len(results['failed_tickers'])}):")
            for ticker in results['failed_tickers'][:10]:  # Show first 10
                print(f"  • {ticker}")
            if len(results['failed_tickers']) > 10:
                print(f"  ... and {len(results['failed_tickers']) - 10} more")


class TradingPerformanceAnalyzer:
    """Trading performance analysis functionality"""
    
    def analyze_trading_performance(self, csv_path: Path) -> Dict[str, Any]:
        """Analyze trading performance from completed trades"""
        print("\n" + "="*60)
        print("🎯 TRADING PERFORMANCE ANALYSIS")
        print("="*60)
        
        completed_trades = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if str(row.get('tracking_status', '')).strip().lower() == 'completed':
                        try:
                            trade = self._parse_completed_trade(row)
                            if trade:
                                completed_trades.append(trade)
                        except Exception as e:
                            log_debug(f"Error parsing trade: {e}")
            
            if not completed_trades:
                print("❌ No completed trades found for analysis")
                return {}
            
            # Perform analysis
            results = self._analyze_trades(completed_trades)
            self._display_performance_results(results)
            
            return results
            
        except Exception as e:
            log_error(f"Error analyzing performance: {e}")
            return {}
    
    def _parse_completed_trade(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Parse a completed trade row into analysis format"""
        try:
            ticker = row.get('ticker', '')
            decision = row.get('decision', '')
            
            # Get prices
            entry_price = float(row.get('recommendation_price', 0) or 0)
            close_price = float(row.get('price_close', 0) or 0)
            
            if entry_price <= 0 or close_price <= 0:
                return None
            
            # Calculate return
            if decision == 'LONG':
                return_pct = ((close_price - entry_price) / entry_price) * 100
            elif decision == 'SHORT':
                return_pct = ((entry_price - close_price) / entry_price) * 100
            else:
                return None
            
            return {
                'ticker': ticker,
                'decision': decision,
                'entry_price': entry_price,
                'close_price': close_price,
                'return': return_pct,
                'timestamp': row.get('recommendation_timestamp', ''),
                'is_winner': return_pct > 0
            }
            
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
    """Main function"""
    print("🚀 Trading Recommendations Price Backfiller - ENHANCED VERSION")
    print("=" * 60)
    
    # Validate FMP API key
    if not Config.FMP_API_KEY:
        print("❌ FMP_API_KEY is required in config.py or .env file")
        sys.exit(1)
    
    try:
        # Initialize backfiller
        backfiller = TradingRecommendationsPriceBackfiller(Config.FMP_API_KEY)
        
        # Initialize flow control variables
        update_prices = True
        run_analysis = False
        dry_run = False
        
        # Check if user wants to test API connectivity first
        print("Would you like to test API connectivity with a few sample tickers first? (y/n)")
        test_choice = input().strip().lower()
        
        if test_choice in ['y', 'yes']:
            test_tickers = ['AAPL', 'MSFT', 'TSLA']  # Known good tickers
            print(f"\n🧪 Testing API connectivity with {', '.join(test_tickers)}...")
            
            for ticker in test_tickers:
                test_timestamp = datetime.now(timezone.utc) - timedelta(hours=24)
                result = backfiller.get_price_at_timestamp(ticker, test_timestamp)
                if result:
                    price, timestamp = result
                    print(f"✅ {ticker}: ${price:.2f} at {timestamp.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print(f"❌ {ticker}: No price data found")
            
            print("\nAPI test complete. Proceeding with file selection...\n")
        
        # Select CSV file
        csv_path = backfiller.select_csv_file()
        
        if not csv_path.exists():
            print(f"❌ File not found: {csv_path}")
            sys.exit(1)
        
        # Analyze file
        analysis = backfiller.analyze_csv_file(csv_path)
        backfiller.display_analysis_results(analysis)
        
        # Check if user just wants to run performance analysis
        if analysis['completed_tracking'] > 0:
            print(f"\n💡 Found {analysis['completed_tracking']} completed trades for analysis.")
            
            while True:
                action_choice = input("What would you like to do?\n  1. Update missing prices and analyze performance\n  2. Just analyze existing completed trades\n  3. Just update missing prices\nChoice (1/2/3): ").strip()
                
                if action_choice == '1':
                    update_prices = True
                    run_analysis = True
                    break
                elif action_choice == '2':
                    print("Running performance analysis only...")
                    analyzer = TradingPerformanceAnalyzer()
                    performance_results = analyzer.analyze_trading_performance(csv_path)
                    return
                elif action_choice == '3':
                    update_prices = True
                    run_analysis = False
                    break
                else:
                    print("Please enter 1, 2, or 3")
        else:
            update_prices = True
            run_analysis = True
        
        if analysis['updatable_rows'] == 0 and not update_prices:
            print("✅ No rows need updating. All data appears to be complete!")
            return
        
        if not update_prices:
            return
        
        # Show sample of problematic tickers for user awareness
        if analysis.get('sample_rows'):
            sample_tickers = [row['ticker'] for row in analysis['sample_rows'][:3]]
            print(f"\n🔍 Sample tickers to be processed: {', '.join(sample_tickers)}")
            print("💡 Note: OTC stocks (ending in F) may have limited historical data")
        
        # Confirm with user for price updates
        if update_prices and analysis['updatable_rows'] > 0:
            print(f"\nThis will attempt to update {analysis['updatable_rows']} rows.")
            print("⚠️  This will make API calls to FMP and may incur costs.")
            est_calls = analysis['updatable_rows'] * 3  # Rough estimate
            print(f"📊 Estimated API calls: ~{est_calls}")
            
            while True:
                choice = input("\nProceed with price updates? (y/n/d for dry-run): ").strip().lower()
                if choice in ['y', 'yes']:
                    dry_run = False
                    break
                elif choice in ['n', 'no']:
                    print("❌ Operation cancelled")
                    return
                elif choice in ['d', 'dry', 'dry-run']:
                    dry_run = True
                    break
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'd' for dry-run")
        else:
            dry_run = False
        
        # Update file if requested
        results = {}
        if update_prices:
            start_time = datetime.now()
            results = backfiller.update_csv_file(csv_path, dry_run=dry_run)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            backfiller.display_results(results)
            
            print(f"\n⏱️  Processing time: {processing_time:.1f} seconds")
            
            if not dry_run and results['updated_rows'] > 0:
                print(f"\n✅ Successfully updated {csv_path}")
                print("💡 A backup was created before making changes")
                
                if results.get('failed_tickers'):
                    print(f"\n💭 Suggestions for failed tickers:")
                    print(f"  1. Check if tickers are correctly formatted")
                    print(f"  2. Verify tickers are actively traded")
                    print(f"  3. Consider using different exchanges (e.g., .TO for Toronto)")
                    print(f"  4. Some OTC stocks may not have historical data available")
        
        # Run performance analysis if requested and we have completed trades
        if run_analysis or (update_prices and not dry_run and results.get('completed_rows', 0) > 0):
            # Check total completed trades after updates
            total_completed = 0
            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if str(row.get('tracking_status', '')).strip().lower() == 'completed':
                            total_completed += 1
            except:
                pass
            
            if total_completed > 0:
                if not run_analysis:  # Only ask if we haven't already decided
                    print(f"\n🎯 Performance Analysis Available!")
                    print(f"Found {total_completed} completed trades total.")
                    
                    while True:
                        analyze_choice = input("Run performance analysis? (y/n): ").strip().lower()
                        if analyze_choice in ['y', 'yes']:
                            run_analysis = True
                            break
                        elif analyze_choice in ['n', 'no']:
                            break
                        else:
                            print("Please enter 'y' for yes or 'n' for no")
                
                if run_analysis:
                    analyzer = TradingPerformanceAnalyzer()
                    analyzer.analyze_trading_performance(csv_path)
            else:
                print(f"\n📊 No completed trades available for performance analysis yet.")
                print(f"Trades will be marked 'completed' once all price checkpoints are filled.")
        
        print(f"\n✅ Price backfiller operation completed!")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        log_error(f"Main function error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
