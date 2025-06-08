"""
Trading Recommendations Price Backfiller
Complete solution for updating missing price data in trading_recommendations.csv

This script will:
1. Prompt user to select CSV file to update
2. Fill in missing recommendation_price, price_checkpoint1, price_checkpoint2, price_close
3. Calculate percentage changes for each checkpoint
4. Mark tracking_status as 'completed' when all fields are populated
5. Handle market hours edge cases and skip rows that already have values
6. Create backups before making changes
7. Provide comprehensive trading performance analysis

Python 3.13.3 compatible
"""
import sys
import csv
import shutil
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
    sys.exit(1)


class TradingRecommendationsPriceBackfiller:
    """Complete price backfiller for trading recommendations CSV"""
    
    def __init__(self, fmp_api_key: str):
        """Initialize the price backfiller"""
        self.fmp_loader = BaseFMPLoader(fmp_api_key)
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.timezone('UTC')
        
        # Cache to avoid duplicate API calls
        self.price_cache: Dict[str, List[Dict]] = {}
        
        log_info("🔄 Trading Recommendations Price Backfiller initialized")
    
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
        """Analyze the CSV file to understand what needs to be updated"""
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
            'sample_rows': []
        }
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                
                if not headers:
                    raise ValueError("CSV file appears to be empty or invalid")
                
                # Verify required columns exist
                required_columns = [
                    'ticker', 'recommendation_timestamp', 'tracking_status',
                    'recommendation_price', 'price_checkpoint1', 'price_checkpoint2', 'price_close'
                ]
                
                missing_columns = [col for col in required_columns if col not in headers]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                for row_num, row in enumerate(reader):
                    analysis['total_rows'] += 1
                    
                    # Check tracking status
                    tracking_status = str(row.get('tracking_status', '')).strip().lower()
                    if tracking_status == 'completed':
                        analysis['completed_tracking'] += 1
                        continue
                    elif tracking_status == 'pending':
                        analysis['pending_tracking'] += 1
                    
                    # Check for missing prices
                    rec_price = str(row.get('recommendation_price', '')).strip()
                    checkpoint1 = str(row.get('price_checkpoint1', '')).strip()
                    checkpoint2 = str(row.get('price_checkpoint2', '')).strip()
                    close_price = str(row.get('price_close', '')).strip()
                    
                    has_missing = False
                    
                    if not rec_price or rec_price.lower() in ['none', '', '0', '0.0', '0.00']:
                        analysis['missing_recommendation_price'] += 1
                        has_missing = True
                    
                    if not checkpoint1 or checkpoint1.lower() in ['none', '', '0', '0.0', '0.00']:
                        analysis['missing_checkpoint1'] += 1
                        has_missing = True
                    
                    if not checkpoint2 or checkpoint2.lower() in ['none', '', '0', '0.0', '0.00']:
                        analysis['missing_checkpoint2'] += 1
                        has_missing = True
                    
                    if not close_price or close_price.lower() in ['none', '', '0', '0.0', '0.00']:
                        analysis['missing_close'] += 1
                        has_missing = True
                    
                    if has_missing:
                        analysis['updatable_rows'] += 1
                        
                        # Store sample for display
                        if len(analysis['sample_rows']) < 5:
                            analysis['sample_rows'].append({
                                'row': row_num + 2,  # +2 for header and 0-indexing
                                'ticker': row.get('ticker', 'N/A'),
                                'timestamp': row.get('recommendation_timestamp', 'N/A'),
                                'rec_price': rec_price or 'MISSING',
                                'checkpoint1': checkpoint1 or 'MISSING',
                                'checkpoint2': checkpoint2 or 'MISSING',
                                'close': close_price or 'MISSING'
                            })
        
        except Exception as e:
            log_error(f"Error analyzing CSV: {e}")
            raise
        
        return analysis
    
    def display_analysis_results(self, analysis: Dict[str, Any]):
        """Display analysis results to user"""
        print("\n" + "="*60)
        print("📊 ANALYSIS RESULTS")
        print("="*60)
        print(f"Total rows: {analysis['total_rows']}")
        print(f"Already completed: {analysis['completed_tracking']}")
        print(f"Pending tracking: {analysis['pending_tracking']}")
        print(f"Rows needing updates: {analysis['updatable_rows']}")
        print()
        print("Missing data breakdown:")
        print(f"  • Recommendation prices: {analysis['missing_recommendation_price']}")
        print(f"  • Checkpoint 1 ({Config.PRICE_CHECK_1_MINUTES}m): {analysis['missing_checkpoint1']}")
        print(f"  • Checkpoint 2 ({Config.PRICE_CHECK_2_MINUTES}m): {analysis['missing_checkpoint2']}")
        print(f"  • Close prices ({Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d}): {analysis['missing_close']}")
        
        if analysis['sample_rows']:
            print(f"\nSample rows needing updates:")
            for sample in analysis['sample_rows']:
                print(f"  Row {sample['row']}: {sample['ticker']} - "
                      f"Rec:{sample['rec_price']}, "
                      f"C1:{sample['checkpoint1']}, "
                      f"C2:{sample['checkpoint2']}, "
                      f"Close:{sample['close']}")
        
        print("\n" + "="*60)
    
    def get_price_at_timestamp(self, ticker: str, target_timestamp: datetime, 
                              interval: str = '5min') -> Optional[Tuple[float, datetime]]:
        """Get historical price at specific timestamp with fallback strategies"""
        try:
            # Skip fetching if target is in the future beyond buffer
            now_utc = datetime.now(timezone.utc)
            if target_timestamp > now_utc + timedelta(minutes=Config.PRICE_FETCH_FUTURE_BUFFER_MINUTES):
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
            
            log_warning(f"❌ No price data available for {ticker} on {target_date} (tried all strategies)")
            return None
            
        except Exception as e:
            log_error(f"Error getting price for {ticker}: {e}")
            return None
    
    def _try_intraday_data(self, ticker: str, target_timestamp: datetime, 
                          target_date: datetime.date, interval: str) -> Optional[Tuple[float, datetime]]:
        """Try to get intraday historical data"""
        cache_key = f"{ticker}_{target_date}_{interval}_intraday"
        
        if cache_key not in self.price_cache:
            log_debug(f"🔍 Trying intraday data for {ticker}")
            historical_data = self._get_intraday_data(ticker, target_date, interval)
            self.price_cache[cache_key] = historical_data or []
        
        historical_data = self.price_cache[cache_key]
        
        if historical_data:
            closest_price_data = self._find_closest_price(historical_data, target_timestamp)
            if closest_price_data:
                log_info(f"✅ Found intraday price for {ticker}: ${closest_price_data['close']:.2f}")
                return closest_price_data['close'], closest_price_data['timestamp']
        
        return None
    
    def _try_daily_data(self, ticker: str, target_timestamp: datetime, 
                       target_date: datetime.date) -> Optional[Tuple[float, datetime]]:
        """Try to get daily historical data as fallback"""
        cache_key = f"{ticker}_{target_date}_daily"
        
        if cache_key not in self.price_cache:
            log_debug(f"🔍 Trying daily data for {ticker}")
            price = self._get_daily_price(ticker, target_date)
            self.price_cache[cache_key] = price
        
        price = self.price_cache[cache_key]
        
        if price and price > 0:
            # For daily data, use the target date at market close as timestamp
            close_time = self.est_tz.localize(
                datetime.combine(target_date, datetime.min.time().replace(hour=16))
            ).astimezone(timezone.utc)
            log_info(f"✅ Found daily price for {ticker}: ${price:.2f}")
            return price, close_time
        
        return None
    
    def _try_current_quote(self, ticker: str, target_timestamp: datetime) -> Optional[Tuple[float, datetime]]:
        """Try to get current/recent quote for very recent timestamps"""
        cache_key = f"{ticker}_current_quote"
        
        if cache_key not in self.price_cache:
            log_debug(f"🔍 Trying current quote for {ticker}")
            price = self._get_current_quote(ticker)
            self.price_cache[cache_key] = price
        
        price = self.price_cache[cache_key]
        
        if price and price > 0:
            log_info(f"✅ Found current quote for {ticker}: ${price:.2f}")
            return price, datetime.now(timezone.utc)
        
        return None
    
    def _get_intraday_data(self, ticker: str, date: datetime.date, interval: str = '5min') -> List[Dict]:
        """Get intraday historical data from FMP"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            endpoint = f"historical-chart/{interval}/{ticker}"
            params = {
                'from': date_str,
                'to': date_str
            }
            
            data = self.fmp_loader.make_request(endpoint, params)
            
            if not data or not isinstance(data, list):
                return []
            
            # Parse timestamps and sort
            processed_data = []
            for item in data:
                try:
                    timestamp_str = item.get('date', '')
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    est_dt = self.est_tz.localize(dt)
                    utc_dt = est_dt.astimezone(self.utc_tz)
                    
                    processed_data.append({
                        'timestamp': utc_dt,
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'close': float(item.get('close', 0)),
                        'volume': int(item.get('volume', 0))
                    })
                except (ValueError, TypeError) as e:
                    log_debug(f"Error parsing data point for {ticker}: {e}")
                    continue
            
            # Sort by timestamp
            processed_data.sort(key=lambda x: x['timestamp'])
            
            log_debug(f"Fetched {len(processed_data)} price points for {ticker} on {date_str}")
            return processed_data
            
        except Exception as e:
            log_error(f"Error fetching intraday data for {ticker}: {e}")
            return []
    
    def _get_daily_price(self, ticker: str, date: datetime.date) -> Optional[float]:
        """Get daily closing price as fallback"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            
            # Try historical-price-full endpoint
            endpoint = f"historical-price-full/{ticker}"
            params = {
                'from': date_str,
                'to': date_str
            }
            
            data = self.fmp_loader.make_request(endpoint, params)
            
            if data and 'historical' in data and data['historical']:
                close_price = float(data['historical'][0].get('close', 0))
                if close_price > 0:
                    log_debug(f"Found daily close price for {ticker}: ${close_price:.2f}")
                    return close_price
            
            # Fallback: try simple historical endpoint
            simple_endpoint = f"historical-price/{ticker}"
            simple_data = self.fmp_loader.make_request(simple_endpoint)
            
            if simple_data and isinstance(simple_data, list) and simple_data:
                historical_item = simple_data[0]
                if historical_item.get('date') == date_str:
                    close_price = float(historical_item.get('close', 0))
                    if close_price > 0:
                        log_debug(f"Found historical price for {ticker}: ${close_price:.2f}")
                        return close_price
            
            return None
            
        except Exception as e:
            log_error(f"Error getting daily price for {ticker}: {e}")
            return None
    
    def _get_current_quote(self, ticker: str) -> Optional[float]:
        """Get current quote as last resort"""
        try:
            # Try quote endpoint
            endpoint = f"quote/{ticker}"
            data = self.fmp_loader.make_request(endpoint)
            
            if data and isinstance(data, list) and data:
                quote = data[0]
                price = float(quote.get('price', 0))
                if price > 0:
                    log_debug(f"Found current quote for {ticker}: ${price:.2f}")
                    return price
            
            # Fallback: try quote-short endpoint
            short_endpoint = f"quote-short/{ticker}"
            short_data = self.fmp_loader.make_request(short_endpoint)
            
            if short_data and isinstance(short_data, list) and short_data:
                quote = short_data[0]
                price = float(quote.get('price', 0))
                if price > 0:
                    log_debug(f"Found short quote for {ticker}: ${price:.2f}")
                    return price
            
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
    
    def _calculate_target_times(self, recommendation_timestamp: datetime) -> Dict[str, datetime]:
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
        """Update the CSV file with missing price data"""
        if dry_run:
            print("\n🧪 DRY RUN MODE - No changes will be made")
        else:
            print(f"\n💾 UPDATING {csv_path.name}")
        
        print("="*60)
        
        results = {
            'processed_rows': 0,
            'updated_rows': 0,
            'completed_rows': 0,
            'skipped_rows': 0,
            'error_rows': 0,
            'api_calls_made': 0,
            'successful_price_fetches': 0,
            'failed_tickers': [],
            'strategy_success': {
                'intraday': 0,
                'daily': 0,
                'current_quote': 0
            }
        }
        
        # Create backup
        if not dry_run:
            backup_path = csv_path.parent / f"{csv_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            shutil.copy2(csv_path, backup_path)
            log_info(f"📋 Created backup: {backup_path}")
        
        # Read and process CSV
        rows = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                
                for row_num, row in enumerate(reader, 1):
                    results['processed_rows'] += 1
                    
                    # Skip if already completed
                    if str(row.get('tracking_status', '')).strip().lower() == 'completed':
                        results['skipped_rows'] += 1
                        rows.append(row)
                        continue
                    
                    try:
                        updated_row = self._update_row_prices(row, results)
                        
                        # Check if all prices are now populated
                        if self._is_row_complete(updated_row):
                            updated_row['tracking_status'] = 'completed'
                            results['completed_rows'] += 1
                            log_info(f"✅ Completed: {updated_row['ticker']}")
                        
                        results['updated_rows'] += 1
                        rows.append(updated_row)
                        
                        # Progress indicator every 10 rows
                        if row_num % 10 == 0:
                            print(f"📊 Processed {row_num}/{results['processed_rows']} rows...")
                        
                    except Exception as e:
                        log_error(f"Error processing row {row_num} ({row.get('ticker', 'Unknown')}): {e}")
                        results['error_rows'] += 1
                        rows.append(row)  # Keep original row
                
                # Write updated CSV
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
    
    def _update_row_prices(self, row: Dict[str, str], results: Dict[str, Any]) -> Dict[str, str]:
        """Update missing prices for a single row"""
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
            price_data = self.get_price_at_timestamp(ticker, targets['recommendation'])
            ticker_api_calls += 1
            if price_data:
                price, actual_timestamp = price_data
                updated_row['recommendation_price'] = f"{price:.2f}"
                updated_row['recommendation_timestamp'] = actual_timestamp.isoformat()
                ticker_successful_fetches += 1
                log_info(f"📈 {ticker} recommendation price: ${price:.2f}")
        
        # Update checkpoint 1
        checkpoint1 = str(row.get('price_checkpoint1', '')).strip()
        if not checkpoint1 or checkpoint1.lower() in ['none', '', '0', '0.0', '0.00']:
            price_data = self.get_price_at_timestamp(ticker, targets['checkpoint1'])
            ticker_api_calls += 1
            if price_data:
                price, actual_timestamp = price_data
                updated_row['price_checkpoint1'] = f"{price:.2f}"
                updated_row['price_checkpoint1_timestamp'] = actual_timestamp.isoformat()
                
                # Calculate percentage change
                rec_price_val = self._get_float_value(updated_row.get('recommendation_price', ''))
                if rec_price_val and rec_price_val > 0:
                    change_pct = ((price - rec_price_val) / rec_price_val) * 100
                    updated_row['price_checkpoint1_change_pct'] = f"{change_pct:.2f}"
                
                ticker_successful_fetches += 1
                log_info(f"📈 {ticker} checkpoint1 ({Config.PRICE_CHECK_1_MINUTES}m): ${price:.2f}")
        
        # Update checkpoint 2
        checkpoint2 = str(row.get('price_checkpoint2', '')).strip()
        if not checkpoint2 or checkpoint2.lower() in ['none', '', '0', '0.0', '0.00']:
            price_data = self.get_price_at_timestamp(ticker, targets['checkpoint2'])
            ticker_api_calls += 1
            if price_data:
                price, actual_timestamp = price_data
                updated_row['price_checkpoint2'] = f"{price:.2f}"
                updated_row['price_checkpoint2_timestamp'] = actual_timestamp.isoformat()
                
                # Calculate percentage change
                rec_price_val = self._get_float_value(updated_row.get('recommendation_price', ''))
                if rec_price_val and rec_price_val > 0:
                    change_pct = ((price - rec_price_val) / rec_price_val) * 100
                    updated_row['price_checkpoint2_change_pct'] = f"{change_pct:.2f}"
                
                ticker_successful_fetches += 1
                log_info(f"📈 {ticker} checkpoint2 ({Config.PRICE_CHECK_2_MINUTES}m): ${price:.2f}")
        
        # Update close price
        close_price = str(row.get('price_close', '')).strip()
        if not close_price or close_price.lower() in ['none', '', '0', '0.0', '0.00']:
            price_data = self.get_price_at_timestamp(ticker, targets['close'])
            ticker_api_calls += 1
            if price_data:
                price, actual_timestamp = price_data
                updated_row['price_close'] = f"{price:.2f}"
                updated_row['price_close_timestamp'] = actual_timestamp.isoformat()
                
                # Calculate percentage change
                rec_price_val = self._get_float_value(updated_row.get('recommendation_price', ''))
                if rec_price_val and rec_price_val > 0:
                    change_pct = ((price - rec_price_val) / rec_price_val) * 100
                    updated_row['price_close_change_pct'] = f"{change_pct:.2f}"
                
                ticker_successful_fetches += 1
                log_info(f"📈 {ticker} close ({Config.CLOSE_PRICE_HOUR:02d}:{Config.CLOSE_PRICE_MINUTE:02d}): ${price:.2f}")
        
        # Update results tracking
        results['api_calls_made'] += ticker_api_calls
        results['successful_price_fetches'] += ticker_successful_fetches
        
        if ticker_successful_fetches == 0 and ticker_api_calls > 0:
            results['failed_tickers'].append(ticker)
            log_warning(f"❌ No prices found for {ticker} despite {ticker_api_calls} API attempts")
        elif ticker_successful_fetches > 0:
            log_info(f"✅ {ticker}: {ticker_successful_fetches}/{ticker_api_calls} successful price fetches")
        
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
        required_fields = [
            'recommendation_price',
            'price_checkpoint1', 
            'price_checkpoint2',
            'price_close'
        ]
        
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
        
        # Show strategy breakdown if available
        if 'strategy_success' in results:
            strategy_stats = results['strategy_success']
            total_strategy_success = sum(strategy_stats.values())
            if total_strategy_success > 0:
                print(f"\nStrategy breakdown:")
                for strategy, count in strategy_stats.items():
                    if count > 0:
                        print(f"  • {strategy}: {count} successes")
        
        # Show failed tickers if any
        if results.get('failed_tickers'):
            failed_count = len(results['failed_tickers'])
            print(f"\n❌ Tickers with no price data ({failed_count}):")
            for ticker in results['failed_tickers'][:10]:  # Show first 10
                print(f"  • {ticker}")
            
            if failed_count > 10:
                print(f"  • ... and {failed_count - 10} more")
            
            print(f"\n💡 Failed tickers may be:")
            print(f"  • OTC/Pink Sheet stocks (limited data)")
            print(f"  • Foreign stocks with different symbols")
            print(f"  • Recently delisted or inactive stocks")
            print(f"  • Stocks with trading suspensions")
        
        print("="*60)


class TradingPerformanceAnalyzer:
    """Comprehensive trading performance analysis for completed trades"""
    
    def __init__(self):
        """Initialize the performance analyzer"""
        self.est_tz = pytz.timezone('US/Eastern')
        
    def analyze_trading_performance(self, csv_path: Path) -> Dict[str, Any]:
        """Perform comprehensive analysis of trading performance"""
        print(f"\n📊 ANALYZING TRADING PERFORMANCE")
        print("="*60)
        
        trades = self._load_completed_trades(csv_path)
        
        if not trades:
            print("❌ No completed trades found for analysis")
            return {}
        
        print(f"📈 Analyzing {len(trades)} completed trades...")
        
        # Calculate all performance metrics
        results = {
            'total_trades': len(trades),
            'timeframe_analysis': self._analyze_by_timeframe(trades),
            'direction_analysis': self._analyze_by_direction(trades),
            'overall_metrics': self._calculate_overall_metrics(trades),
            'best_worst_trades': self._find_best_worst_trades(trades),
            'monthly_performance': self._analyze_monthly_performance(trades),
            'ticker_performance': self._analyze_ticker_performance(trades)
        }
        
        self._display_performance_results(results)
        return results
    
    def _load_completed_trades(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load only completed trades from CSV"""
        trades = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if str(row.get('tracking_status', '')).strip().lower() == 'completed':
                        # Convert to proper data types
                        trade = self._process_trade_row(row)
                        if trade:
                            trades.append(trade)
        
        except Exception as e:
            log_error(f"Error loading trades: {e}")
        
        return trades
    
    def _process_trade_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Process a single trade row into structured data"""
        try:
            # Extract basic info
            ticker = row.get('ticker', '')
            decision = row.get('decision', '').upper()
            
            if not ticker or decision not in ['LONG', 'SHORT']:
                return None
            
            # Extract prices and changes
            rec_price = self._safe_float(row.get('recommendation_price'))
            checkpoint1_change = self._safe_float(row.get('price_checkpoint1_change_pct'))
            checkpoint2_change = self._safe_float(row.get('price_checkpoint2_change_pct'))
            close_change = self._safe_float(row.get('price_close_change_pct'))
            
            if rec_price is None:
                return None
            
            # Parse timestamp
            timestamp_str = row.get('recommendation_timestamp', '')
            try:
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = None
            except:
                timestamp = None
            
            trade = {
                'ticker': ticker,
                'decision': decision,
                'recommendation_price': rec_price,
                'timestamp': timestamp,
                'confidence': self._safe_float(row.get('confidence', 0)),
                'checkpoint1_change_pct': checkpoint1_change,
                'checkpoint2_change_pct': checkpoint2_change,
                'close_change_pct': close_change,
                'raw_row': row
            }
            
            # Calculate directional returns (account for LONG vs SHORT)
            trade['checkpoint1_return'] = self._calculate_directional_return(
                checkpoint1_change, decision) if checkpoint1_change is not None else None
            trade['checkpoint2_return'] = self._calculate_directional_return(
                checkpoint2_change, decision) if checkpoint2_change is not None else None
            trade['close_return'] = self._calculate_directional_return(
                close_change, decision) if close_change is not None else None
            
            return trade
            
        except Exception as e:
            log_debug(f"Error processing trade row for {row.get('ticker', 'Unknown')}: {e}")
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or str(value).strip().lower() in ['', 'none', 'nan']:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _calculate_directional_return(self, price_change_pct: float, decision: str) -> float:
        """Calculate return accounting for LONG vs SHORT positions"""
        if decision == 'LONG':
            return price_change_pct  # Positive price change = positive return
        else:  # SHORT
            return -price_change_pct  # Positive price change = negative return for shorts
    
    def _analyze_by_timeframe(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by timeframe (checkpoint1, checkpoint2, close)"""
        timeframes = {
            f'checkpoint1_{Config.PRICE_CHECK_1_MINUTES}m': 'checkpoint1_return',
            f'checkpoint2_{Config.PRICE_CHECK_2_MINUTES}m': 'checkpoint2_return', 
            f'close_{Config.CLOSE_PRICE_HOUR:02d}h{Config.CLOSE_PRICE_MINUTE:02d}m': 'close_return'
        }
        
        results = {}
        
        for timeframe_name, return_field in timeframes.items():
            returns = [trade[return_field] for trade in trades if trade[return_field] is not None]
            
            if returns:
                wins = [r for r in returns if r > 0]
                losses = [r for r in returns if r < 0]
                neutral = [r for r in returns if r == 0]
                
                results[timeframe_name] = {
                    'total_trades': len(returns),
                    'wins': len(wins),
                    'losses': len(losses),
                    'neutral': len(neutral),
                    'win_rate': len(wins) / len(returns) * 100 if returns else 0,
                    'avg_return': sum(returns) / len(returns),
                    'avg_win': sum(wins) / len(wins) if wins else 0,
                    'avg_loss': sum(losses) / len(losses) if losses else 0,
                    'best_trade': max(returns) if returns else 0,
                    'worst_trade': min(returns) if returns else 0,
                    'total_return': sum(returns),
                    'std_dev': self._calculate_std_dev(returns)
                }
            else:
                results[timeframe_name] = {'total_trades': 0}
        
        return results
    
    def _analyze_by_direction(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by trade direction (LONG vs SHORT)"""
        long_trades = [t for t in trades if t['decision'] == 'LONG']
        short_trades = [t for t in trades if t['decision'] == 'SHORT']
        
        results = {}
        
        for direction, trade_list in [('LONG', long_trades), ('SHORT', short_trades)]:
            if trade_list:
                # Use close returns for overall direction analysis
                close_returns = [t['close_return'] for t in trade_list if t['close_return'] is not None]
                
                if close_returns:
                    wins = [r for r in close_returns if r > 0]
                    losses = [r for r in close_returns if r < 0]
                    
                    results[direction] = {
                        'total_trades': len(close_returns),
                        'wins': len(wins),
                        'losses': len(losses),
                        'win_rate': len(wins) / len(close_returns) * 100,
                        'avg_return': sum(close_returns) / len(close_returns),
                        'total_return': sum(close_returns),
                        'best_trade': max(close_returns),
                        'worst_trade': min(close_returns)
                    }
                else:
                    results[direction] = {'total_trades': 0}
            else:
                results[direction] = {'total_trades': 0}
        
        return results
    
    def _calculate_overall_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall portfolio metrics"""
        close_returns = [t['close_return'] for t in trades if t['close_return'] is not None]
        
        if not close_returns:
            return {}
        
        wins = [r for r in close_returns if r > 0]
        losses = [r for r in close_returns if r < 0]
        
        # Calculate additional metrics
        win_rate = len(wins) / len(close_returns) * 100
        avg_return = sum(close_returns) / len(close_returns)
        total_return = sum(close_returns)
        
        # Risk metrics
        std_dev = self._calculate_std_dev(close_returns)
        sharpe_ratio = (avg_return / std_dev) if std_dev > 0 else 0
        
        # Profit factor (total wins / total losses)
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        return {
            'total_trades': len(close_returns),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'std_dev': std_dev,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'best_trade': max(close_returns),
            'worst_trade': min(close_returns),
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0
        }
    
    def _find_best_worst_trades(self, trades: List[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
        """Find best and worst performing trades"""
        # Sort by close return
        trades_with_returns = [t for t in trades if t['close_return'] is not None]
        trades_with_returns.sort(key=lambda x: x['close_return'], reverse=True)
        
        best_trades = []
        worst_trades = []
        
        for trade in trades_with_returns[:top_n]:
            best_trades.append({
                'ticker': trade['ticker'],
                'decision': trade['decision'],
                'return': trade['close_return'],
                'timestamp': trade['timestamp'].strftime('%Y-%m-%d') if trade['timestamp'] else 'Unknown'
            })
        
        for trade in trades_with_returns[-top_n:]:
            worst_trades.append({
                'ticker': trade['ticker'],
                'decision': trade['decision'],
                'return': trade['close_return'],
                'timestamp': trade['timestamp'].strftime('%Y-%m-%d') if trade['timestamp'] else 'Unknown'
            })
        
        return {
            'best_trades': best_trades,
            'worst_trades': worst_trades[::-1]  # Reverse to show worst first
        }
    
    def _analyze_monthly_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by month"""
        monthly_data = {}
        
        for trade in trades:
            if trade['timestamp'] and trade['close_return'] is not None:
                month_key = trade['timestamp'].strftime('%Y-%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                
                monthly_data[month_key].append(trade['close_return'])
        
        monthly_results = {}
        for month, returns in monthly_data.items():
            if returns:
                wins = [r for r in returns if r > 0]
                monthly_results[month] = {
                    'total_trades': len(returns),
                    'wins': len(wins),
                    'win_rate': len(wins) / len(returns) * 100,
                    'total_return': sum(returns),
                    'avg_return': sum(returns) / len(returns)
                }
        
        return monthly_results
    
    def _analyze_ticker_performance(self, trades: List[Dict[str, Any]], min_trades: int = 2) -> Dict[str, Any]:
        """Analyze performance by individual ticker (only tickers with multiple trades)"""
        ticker_data = {}
        
        for trade in trades:
            if trade['close_return'] is not None:
                ticker = trade['ticker']
                if ticker not in ticker_data:
                    ticker_data[ticker] = []
                ticker_data[ticker].append(trade['close_return'])
        
        # Filter to tickers with minimum number of trades
        ticker_results = {}
        for ticker, returns in ticker_data.items():
            if len(returns) >= min_trades:
                wins = [r for r in returns if r > 0]
                ticker_results[ticker] = {
                    'total_trades': len(returns),
                    'wins': len(wins),
                    'win_rate': len(wins) / len(returns) * 100,
                    'total_return': sum(returns),
                    'avg_return': sum(returns) / len(returns)
                }
        
        return ticker_results
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _display_performance_results(self, results: Dict[str, Any]):
        """Display comprehensive performance analysis"""
        overall = results.get('overall_metrics', {})
        timeframes = results.get('timeframe_analysis', {})
        directions = results.get('direction_analysis', {})
        best_worst = results.get('best_worst_trades', {})
        
        print(f"\n🎯 OVERALL PERFORMANCE")
        print("-" * 40)
        if overall:
            print(f"Total Trades: {overall['total_trades']}")
            print(f"Win Rate: {overall['win_rate']:.1f}%")
            print(f"Average Return: {overall['avg_return']:+.2f}%")
            print(f"Total Return: {overall['total_return']:+.2f}%")
            print(f"Best Trade: {overall['best_trade']:+.2f}%")
            print(f"Worst Trade: {overall['worst_trade']:+.2f}%")
            print(f"Profit Factor: {overall['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        
        print(f"\n📊 PERFORMANCE BY TIMEFRAME")
        print("-" * 40)
        for timeframe, data in timeframes.items():
            if data.get('total_trades', 0) > 0:
                print(f"{timeframe}:")
                print(f"  Trades: {data['total_trades']} | Win Rate: {data['win_rate']:.1f}% | Avg Return: {data['avg_return']:+.2f}%")
        
        print(f"\n🎭 PERFORMANCE BY DIRECTION")
        print("-" * 40)
        for direction, data in directions.items():
            if data.get('total_trades', 0) > 0:
                print(f"{direction}:")
                print(f"  Trades: {data['total_trades']} | Win Rate: {data['win_rate']:.1f}% | Avg Return: {data['avg_return']:+.2f}%")
        
        if best_worst.get('best_trades'):
            print(f"\n🏆 TOP 5 BEST TRADES")
            print("-" * 40)
            for i, trade in enumerate(best_worst['best_trades'], 1):
                print(f"{i}. {trade['ticker']} ({trade['decision']}) | {trade['return']:+.2f}% | {trade['timestamp']}")
        
        if best_worst.get('worst_trades'):
            print(f"\n💀 TOP 5 WORST TRADES")
            print("-" * 40)
            for i, trade in enumerate(best_worst['worst_trades'], 1):
                print(f"{i}. {trade['ticker']} ({trade['decision']}) | {trade['return']:+.2f}% | {trade['timestamp']}")
        
        monthly = results.get('monthly_performance', {})
        if monthly:
            print(f"\n📅 MONTHLY PERFORMANCE")
            print("-" * 40)
            for month in sorted(monthly.keys()):
                data = monthly[month]
                print(f"{month}: {data['total_trades']} trades | {data['win_rate']:.1f}% win rate | {data['total_return']:+.2f}% return")
        
        ticker_perf = results.get('ticker_performance', {})
        if ticker_perf:
            print(f"\n🎯 TOP PERFORMING TICKERS (2+ trades)")
            print("-" * 40)
            # Sort by total return
            sorted_tickers = sorted(ticker_perf.items(), key=lambda x: x[1]['total_return'], reverse=True)
            for ticker, data in sorted_tickers[:10]:  # Top 10
                print(f"{ticker}: {data['total_trades']} trades | {data['win_rate']:.1f}% win rate | {data['total_return']:+.2f}% total")
        
        print("\n" + "="*60)


def main():
    """Main function"""
    print("🚀 Trading Recommendations Price Backfiller")
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
            print("📊 Estimated API calls: ~{} (4 calls per ticker with missing data)".format(analysis['updatable_rows'] * 2))
            
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
                        analysis_choice = input("Run trading performance analysis? (y/n): ").strip().lower()
                        if analysis_choice in ['y', 'yes']:
                            run_analysis = True
                            break
                        elif analysis_choice in ['n', 'no']:
                            print("Skipping performance analysis.")
                            break
                        else:
                            print("Please enter 'y' for yes or 'n' for no")
                
                if run_analysis:
                    analyzer = TradingPerformanceAnalyzer()
                    performance_results = analyzer.analyze_trading_performance(csv_path)
            else:
                print(f"\n📊 No completed trades available for performance analysis yet.")
                print(f"💡 Trades will be marked as 'completed' once all price checkpoints are filled.")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        print("💡 Try running with a smaller subset of data or check API key validity")
        sys.exit(1)


if __name__ == "__main__":
    main()
