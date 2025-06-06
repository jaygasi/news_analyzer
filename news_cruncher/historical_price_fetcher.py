"""
Historical price fetcher and backfill utilities for financial news analysis system
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
    """Fetch historical prices from FMP API for precise timestamps"""
    
    def __init__(self, fmp_loader: BaseFMPLoader):
        """Initialize historical price fetcher"""
        self.fmp_loader = fmp_loader
        self.est_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.timezone('UTC')
        
        # Cache to avoid duplicate API calls
        self.price_cache: Dict[str, List[HistoricalPrice]] = {}
        
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
        """Find the historical price closest to target timestamp"""
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
        
        # Only return if within reasonable time window (30 minutes)
        if min_time_diff <= 1800:  # 30 minutes
            return closest_price
        else:
            log_warning(f"Closest price is {min_time_diff/60:.1f} minutes away, too far from target")
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
    """Utility to backfill missing entry prices in existing CSV data"""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize backfill utility"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
        
    def analyze_missing_prices(self) -> Dict[str, Any]:
        """Analyze how many entries are missing price data"""
        try:
            analysis = {
                'total_rows': 0,
                'missing_entry_prices': 0,
                'missing_price_updates': 0,
                'completed_tracking': 0,
                'backfill_candidates': []
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
                    elif not row.get('price_45m') and entry_price:
                        analysis['missing_price_updates'] += 1
            
            return analysis
            
        except Exception as e:
            log_error(f"Error analyzing CSV: {e}")
            return {}
    
    def backfill_missing_entry_prices(self, dry_run: bool = True) -> Dict[str, Any]:
        """Backfill missing entry prices using historical data"""
        
        log_info("🔄 Starting entry price backfill process...")
        
        results = {
            'processed': 0,
            'successful_backfills': 0,
            'failed_backfills': 0,
            'backfilled_entries': []
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
    
    def validate_existing_prices(self, sample_size: int = 10) -> Dict[str, Any]:
        """Validate existing entry prices against historical data"""
        
        log_info(f"🔍 Validating {sample_size} existing entry prices...")
        
        validation_results = {
            'validated': 0,
            'accurate': 0,
            'inaccurate': 0,
            'errors': 0,
            'accuracy_details': []
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
    """Analyze historical performance of trading decisions"""
    
    def __init__(self, csv_path: str, fmp_loader: BaseFMPLoader):
        """Initialize performance analyzer"""
        self.csv_path = csv_path
        self.historical_fetcher = HistoricalPriceFetcher(fmp_loader)
    
    def calculate_performance_metrics(self, days_to_analyze: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics for completed trades"""
        
        log_info(f"📊 Analyzing performance for last {days_to_analyze} days...")
        
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
            'trade_details': []
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_analyze)
            
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Only analyze completed trades with entry and exit prices
                    if (row.get('recommendation_price') and 
                        row.get('price_close') and 
                        row.get('decision') in ['LONG', 'SHORT']):
                        
                        # Check if within analysis period
                        timestamp_str = row.get('recommendation_timestamp', '')
                        if timestamp_str:
                            entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if entry_time < cutoff_date:
                                continue
                        
                        # Calculate trade performance
                        ticker = row['ticker']
                        decision = row['decision']
                        entry_price = float(row['recommendation_price'])
                        exit_price = float(row['price_close'])
                        
                        # Calculate return based on position direction
                        if decision == 'LONG':
                            return_pct = ((exit_price - entry_price) / entry_price) * 100
                        else:  # SHORT
                            return_pct = ((entry_price - exit_price) / entry_price) * 100
                        
                        metrics['total_trades'] += 1
                        metrics['total_return_pct'] += return_pct
                        
                        trade_detail = {
                            'ticker': ticker,
                            'decision': decision,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return_pct': return_pct,
                            'timestamp': timestamp_str
                        }
                        
                        if return_pct > 0:
                            metrics['winning_trades'] += 1
                            metrics['avg_win_pct'] += return_pct
                            metrics['best_trade_pct'] = max(metrics['best_trade_pct'], return_pct)
                        else:
                            metrics['losing_trades'] += 1
                            metrics['avg_loss_pct'] += abs(return_pct)
                            metrics['worst_trade_pct'] = min(metrics['worst_trade_pct'], return_pct)
                        
                        metrics['trade_details'].append(trade_detail)
            
            # Calculate final metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                
                if metrics['winning_trades'] > 0:
                    metrics['avg_win_pct'] /= metrics['winning_trades']
                
                if metrics['losing_trades'] > 0:
                    metrics['avg_loss_pct'] /= metrics['losing_trades']
            
            log_info(f"📈 Performance Analysis Complete:")
            log_info(f"   Total Trades: {metrics['total_trades']}")
            log_info(f"   Win Rate: {metrics['win_rate']:.1%}")
            log_info(f"   Avg Win: {metrics['avg_win_pct']:.2f}%")
            log_info(f"   Avg Loss: {metrics['avg_loss_pct']:.2f}%")
            log_info(f"   Total Return: {metrics['total_return_pct']:.2f}%")
            
            return metrics
            
        except Exception as e:
            log_error(f"Error calculating performance metrics: {e}")
            return metrics


# Example usage functions
def backfill_missing_prices(csv_path: str, fmp_api_key: str, dry_run: bool = True):
    """Standalone function to backfill missing entry prices"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    # Analyze what needs backfilling
    analysis = backfill_utility.analyze_missing_prices()
    log_info(f"Analysis: {analysis['missing_entry_prices']} entries need backfilling")
    
    # Perform backfill
    if analysis['missing_entry_prices'] > 0:
        results = backfill_utility.backfill_missing_entry_prices(dry_run=dry_run)
        log_info(f"Backfill results: {results}")
        return results
    else:
        log_info("No entries need backfilling")
        return {}

def validate_price_accuracy(csv_path: str, fmp_api_key: str, sample_size: int = 10):
    """Standalone function to validate existing price accuracy"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    backfill_utility = CSVBackfillUtility(csv_path, fmp_loader)
    
    return backfill_utility.validate_existing_prices(sample_size)

def analyze_trading_performance(csv_path: str, fmp_api_key: str, days: int = 30):
    """Standalone function to analyze trading performance"""
    from data_loaders.base_fmp_loader import BaseFMPLoader
    
    fmp_loader = BaseFMPLoader(fmp_api_key)
    analyzer = HistoricalPerformanceAnalyzer(csv_path, fmp_loader)
    
    return analyzer.calculate_performance_metrics(days)


if __name__ == "__main__":
    """Example usage"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent))
    from config import Config
    
    if not Config.FMP_API_KEY:
        print("FMP_API_KEY is required")
        sys.exit(1)
    
    csv_path = "output/trading_decisions.csv"
    
    print("🔄 Historical Price Utilities")
    print("1. Analyzing missing prices...")
    analysis = backfill_missing_prices(csv_path, Config.FMP_API_KEY, dry_run=True)
    
    print("\n2. Validating existing prices...")
    validation = validate_price_accuracy(csv_path, Config.FMP_API_KEY, sample_size=5)
    
    print("\n3. Analyzing performance...")
    performance = analyze_trading_performance(csv_path, Config.FMP_API_KEY, days=30)
