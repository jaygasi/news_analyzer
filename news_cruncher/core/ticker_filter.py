"""
TickerFilterEngine for filtering tickers based on fundamental criteria
Python 3.13.3 compatible
"""
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_debug, log_warning, log_error
from config import Config


@dataclass
class FilterCriteria:
    """Criteria for filtering tickers based on fundamental analysis"""
    # All values should be passed from Config - no hardcoded defaults
    min_price: float  # Eliminates penny stocks
    max_price: float  # Avoids ultra-expensive stocks
    min_avg_volume: int  # Daily share volume (good liquidity)
    min_dollar_volume: int  # Daily dollar volume (institutional interest)
    min_market_cap: int  # Company size (established companies)
    max_volatility_beta: float  # Reasonable volatility
    require_options: bool  # Options = liquidity + institutional interest
    allowed_exchanges: List[str]  # Major exchanges only


class TickerFilterEngine:
    """Filter ticker buckets based on fundamental criteria"""
    
    def __init__(self, fmp_loader: BaseFMPLoader, filter_criteria: FilterCriteria) -> None:
        """Initialize ticker filter engine"""
        self.fmp_loader = fmp_loader
        self.criteria = filter_criteria
        self.fundamentals_cache = {}
        self.cache_db_path = Path('data/fundamentals_cache.db')
        
        # MODIFIED: Enable detailed debug temporarily for troubleshooting
        self.enable_detailed_debug = True  # CHANGED from False to True
        
        # NEW: Control for logging rejected stocks (ADD THESE LINES)
        self.enable_rejected_logging = getattr(Config, 'ENABLE_REJECTED_STOCK_LOGGING', True)
        self.max_rejected_to_log = getattr(Config, 'MAX_REJECTED_STOCKS_TO_LOG', 30)

        # Attributes for enhanced rate limiting
        self.last_request_time: float = 0.0
        self.min_request_interval: float = Config.FMP_MIN_REQUEST_INTERVAL
        self._consecutive_errors: int = 0
        
        # Initialize cache database
        self._init_cache_database()
        
        log_info(f"TickerFilterEngine initialized with criteria:")
        log_info(f"  Price range: ${self.criteria.min_price:.2f} - ${self.criteria.max_price:.2f}")
        log_info(f"  Min avg volume: {self.criteria.min_avg_volume:,} shares")
        log_info(f"  Min dollar volume: ${self.criteria.min_dollar_volume:,}")
        log_info(f"  Min market cap: ${self.criteria.min_market_cap:,}")
        log_info(f"  Max beta: {self.criteria.max_volatility_beta}")
        log_info(f"  Require options: {self.criteria.require_options}")
        log_info(f"  Allowed exchanges: {', '.join(self.criteria.allowed_exchanges)}")
        
        # Log debug settings
        log_info(f"Ticker filter debug mode: {'ENABLED' if self.enable_detailed_debug else 'DISABLED'}")
        log_info(f"Rejected stock logging: {'ENABLED' if self.enable_rejected_logging else 'DISABLED'} (max: {self.max_rejected_to_log})")
    
    def _init_cache_database(self) -> None:
        """Initialize SQLite cache for fundamental data"""
        try:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS fundamentals_cache (
                        ticker TEXT PRIMARY KEY,
                        price REAL,
                        market_cap INTEGER,
                        exchange TEXT,
                        beta REAL,
                        avg_volume INTEGER,
                        volume INTEGER,
                        has_options BOOLEAN,
                        cached_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_cached_at 
                    ON fundamentals_cache(cached_at)
                ''')
                
                conn.commit()
                
        except Exception as e:
            log_error(f"Error initializing fundamentals cache: {e}")
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable detailed debugging output"""
        self.enable_detailed_debug = enabled
        log_info(f"Ticker filter debug mode: {'ENABLED' if enabled else 'DISABLED'}")
    
    def set_rejected_logging(self, enabled: bool, max_to_log: int = 30) -> None:
        """Configure rejected stock logging"""
        self.enable_rejected_logging = enabled
        self.max_rejected_to_log = max_to_log
        log_info(f"Rejected stock logging: {'ENABLED' if enabled else 'DISABLED'} (max: {max_to_log})")
    
    def filter_ticker_buckets(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Filter ticker buckets based on fundamental criteria with enhanced logging"""
        log_info(f"🔍 Filtering {len(ticker_buckets)} tickers by fundamental criteria...")
        
        # Log initial tickers overview
        self._log_initial_tickers(ticker_buckets)
        
        filtered_buckets = {}
        filter_stats = {
            'passed': 0,
            'failed_price': 0,
            'failed_volume': 0,
            'failed_market_cap': 0,
            'failed_beta': 0,
            'failed_options': 0,
            'failed_exchange': 0,
            'failed_data_unavailable': 0
        }
        
        # ENHANCED: Always create rejected_stocks list for better diagnostics
        rejected_stocks = []
        tickers_to_process = list(ticker_buckets.keys())
        
        # Log API diagnostic info
        log_info(f"🔑 API Configuration Check:")
        log_info(f"   FMP API Key: {'✅ Configured' if self.fmp_loader.api_key else '❌ Missing'}")
        log_info(f"   Rate limit: {getattr(Config, 'FMP_REQUESTS_PER_MINUTE', 10)} req/min")
        log_info(f"   Min interval: {getattr(Config, 'FMP_MIN_REQUEST_INTERVAL', 0.2)}s")
        
        # OPTIMIZATION: Process in batches to show progress and avoid timeouts
        batch_size = 50  # Process 50 tickers at a time
        total_batches = (len(tickers_to_process) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(tickers_to_process))
            batch_tickers = tickers_to_process[start_idx:end_idx]
            
            log_info(f"🔄 Processing batch {batch_num + 1}/{total_batches} ({len(batch_tickers)} tickers)")
            
            # OPTIMIZATION: Batch API calls
            batch_fundamentals = self._get_batch_fundamentals(batch_tickers)
            
            for ticker in batch_tickers:
                try:
                    fundamentals = batch_fundamentals.get(ticker)
                    
                    if not fundamentals:
                        filter_stats['failed_data_unavailable'] += 1
                        failure_reason = "No fundamental data available from FMP API"
                        log_debug(f"❌ {ticker}: {failure_reason}")
                        
                        # ENHANCED: Always track data unavailable failures
                        rejected_stocks.append({
                            'ticker': ticker,
                            'reason': failure_reason,
                            'fundamentals': None
                        })
                        continue
                    
                    # Evaluate criteria
                    passed, reason = self._evaluate_criteria(fundamentals)
                    
                    if passed:
                        filtered_buckets[ticker] = ticker_buckets[ticker]
                        filter_stats['passed'] += 1
                        log_debug(f"✅ {ticker}: Passed all criteria")
                    else:
                        # Update filter stats
                        if 'price' in reason.lower():
                            filter_stats['failed_price'] += 1
                        elif 'volume' in reason.lower():
                            filter_stats['failed_volume'] += 1
                        elif 'market cap' in reason.lower():
                            filter_stats['failed_market_cap'] += 1
                        elif 'beta' in reason.lower():
                            filter_stats['failed_beta'] += 1
                        elif 'options' in reason.lower():
                            filter_stats['failed_options'] += 1
                        elif 'exchange' in reason.lower():
                            filter_stats['failed_exchange'] += 1
                        
                        log_debug(f"❌ {ticker}: {reason}")
                        
                        # Track rejected stock
                        rejected_stocks.append({
                            'ticker': ticker,
                            'reason': reason,
                            'fundamentals': fundamentals
                        })
                            
                except Exception as e:
                    filter_stats['failed_data_unavailable'] += 1
                    error_reason = f"Error getting fundamentals - {e}"
                    log_error(f"❌ {ticker}: {error_reason}")
                    
                    rejected_stocks.append({
                        'ticker': ticker,
                        'reason': error_reason,
                        'fundamentals': None
                    })
        
        # Log results
        total_filtered = len(ticker_buckets) - len(filtered_buckets)
        log_info(f"Fundamental filtering complete:")
        log_info(f"  ✅ Passed: {filter_stats['passed']} tickers")
        log_info(f"  ❌ Filtered out: {total_filtered} tickers")
        
        if total_filtered > 0:
            log_info(f"  Filter breakdown:")
            log_info(f"    Price range: {filter_stats['failed_price']}")
            log_info(f"    Volume requirements: {filter_stats['failed_volume']}")
            log_info(f"    Market cap: {filter_stats['failed_market_cap']}")
            log_info(f"    Beta (volatility): {filter_stats['failed_beta']}")
            log_info(f"    Options availability: {filter_stats['failed_options']}")
            log_info(f"    Exchange restrictions: {filter_stats['failed_exchange']}")
            log_info(f"    Data unavailable: {filter_stats['failed_data_unavailable']}")
        
        # ENHANCED: Always log rejected stocks sample (respecting config limits)
        max_to_log = self.max_rejected_to_log if self.enable_rejected_logging else 5
        if rejected_stocks:
            log_info(f"🚫 Sample of rejected stocks (first {min(len(rejected_stocks), max_to_log)}):")
            for i, rejected in enumerate(rejected_stocks[:max_to_log], 1):
                ticker = rejected['ticker']
                reason = rejected['reason']
                fund = rejected['fundamentals']
                
                if fund:
                    price = fund.get('price', 'N/A')
                    volume = fund.get('avg_volume', 'N/A')
                    market_cap = fund.get('market_cap', 'N/A')
                    exchange = fund.get('exchange', 'N/A')
                    
                    if isinstance(volume, (int, float)) and volume > 0:
                        volume_str = f"{volume:,}"
                    else:
                        volume_str = str(volume)
                        
                    if isinstance(market_cap, (int, float)) and market_cap > 0:
                        market_cap_str = f"${market_cap:,}"
                    else:
                        market_cap_str = str(market_cap)
                    
                    log_info(f"   {i:2d}. {ticker:<6} | ${price} | Vol: {volume_str} | Cap: {market_cap_str} | {exchange} | ❌ {reason}")
                else:
                    log_info(f"   {i:2d}. {ticker:<6} | No data available | ❌ {reason}")
        
        return filtered_buckets

    def _rate_limit(self):
        """Enhanced rate limiting with exponential backoff"""
        current_time = time.time()
        
        # Calculate time since last request
        time_since_last = current_time - self.last_request_time
        
        # Ensure minimum interval
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            log_debug(f"🔄 Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Track request for rate limiting
        self.last_request_time = time.time()
        
        # Enhanced: Add exponential backoff for consecutive errors
        if hasattr(self, '_consecutive_errors') and self._consecutive_errors > 3:
            backoff_time = min(30, 2 ** (self._consecutive_errors - 3)) # Exponential backoff, capped at 30s
            log_warning(f"⏳ Exponential backoff: sleeping {backoff_time:.2f}s due to {self._consecutive_errors} consecutive errors")
            time.sleep(backoff_time)

    def _get_batch_fundamentals(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get fundamentals for multiple tickers efficiently with caching"""
        results = {}
        tickers_to_fetch = []
        
        # First, check cache for all tickers
        for ticker in tickers:
            cached = self._get_cached_fundamentals(ticker)
            if cached:
                results[ticker] = cached
            else:
                tickers_to_fetch.append(ticker)
        
        if tickers_to_fetch:
            log_debug(f"📡 Fetching fresh data for {len(tickers_to_fetch)} tickers (cached: {len(results)})")
            
            # Batch fetch remaining tickers
            for ticker in tickers_to_fetch:
                self._rate_limit() # Apply rate limiting before each ticker's fundamental fetch

                try:
                    # Use existing _get_company_fundamentals but cache results
                    fundamentals = self._get_company_fundamentals(ticker)
                    if fundamentals:
                        results[ticker] = fundamentals
                        self._cache_fundamentals(ticker, fundamentals)
                        self._consecutive_errors = 0 # Reset errors on success
                    else:
                        # If _get_company_fundamentals returns None, it's a failure for this ticker
                        self._consecutive_errors += 1
                        results[ticker] = None # Ensure None is stored if fundamentals are not found

                except Exception as e:
                    log_debug(f"Error fetching {ticker}: {e}")
                    self._consecutive_errors += 1 # Increment errors on exception
                    results[ticker] = None
        
        return results

    def _log_initial_tickers(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log which tickers are being considered before filtering"""
        log_info(f"📊 Initial tickers before filtering ({len(ticker_buckets)}):")
        
        # Group by article count for better overview
        by_article_count = {}
        for ticker, articles in ticker_buckets.items():
            count = len(articles)
            if count not in by_article_count:
                by_article_count[count] = []
            by_article_count[count].append(ticker)
        
        for count in sorted(by_article_count.keys(), reverse=True):
            tickers = by_article_count[count]
            display_tickers = ', '.join(tickers[:15])  # Show first 15
            if len(tickers) > 15:
                display_tickers += f" (+{len(tickers)-15} more)"
            log_info(f"  📰 {count} articles: {display_tickers}")
        
        # Show some sample articles to understand data quality (only if debug enabled)
        if self.enable_detailed_debug:
            sample_tickers = list(ticker_buckets.keys())[:3]
            for ticker in sample_tickers:
                articles = ticker_buckets[ticker]
                log_info(f"  📄 Sample {ticker} articles:")
                for i, article in enumerate(articles[:2]):  # Show first 2 articles
                    title = article.get('title', 'No Title')[:60]
                    source = article.get('source', 'Unknown')
                    log_info(f"    {i+1}. [{source}] {title}...")
    
    def _get_company_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch company fundamental data with enhanced error handling and logging"""
        # Check memory cache first
        if ticker in self.fundamentals_cache:
            log_debug(f"📋 Using cached fundamentals for {ticker}")
            return self.fundamentals_cache[ticker]
        
        # Check database cache (24 hour expiry)
        cached_data = self._get_cached_fundamentals(ticker)
        if cached_data:
            self.fundamentals_cache[ticker] = cached_data
            log_debug(f"💾 Using database cached fundamentals for {ticker}")
            return cached_data
        
        # Fetch from API with enhanced error handling
        try:
            log_debug(f"🔍 Fetching fresh fundamentals for {ticker}")
            
            # Get company profile for basic info
            profile_data = self.fmp_loader.make_request(f"profile/{ticker}")
            if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
                log_warning(f"❌ No profile data for {ticker}: {profile_data}")
                return None
            
            profile = profile_data[0]
            
            # Get current quote for volume data
            quote_data = self.fmp_loader.make_request(f"quote/{ticker}")
            if not quote_data or not isinstance(quote_data, list) or len(quote_data) == 0:
                log_warning(f"❌ No quote data for {ticker}: {quote_data}")
                return None
            
            quote = quote_data[0]
            
            # Check if options are available (simplified check)
            has_options = self._check_options_availability(ticker)
            
            # Compile fundamental data
            fundamentals = {
                'price': float(profile.get('price', 0) or quote.get('price', 0)),
                'market_cap': int(profile.get('mktCap', 0) or 0),
                'exchange': str(profile.get('exchangeShortName', '') or profile.get('exchange', '')),
                'beta': float(profile.get('beta', 0) or 0),
                'avg_volume': int(profile.get('volAvg', 0) or quote.get('avgVolume', 0) or 0),
                'volume': int(quote.get('volume', 0) or 0),
                'has_options': bool(has_options)
            }
            
            # Validate critical data
            if fundamentals['price'] <= 0:
                log_warning(f"❌ {ticker}: Invalid price data - {fundamentals['price']}")
                return None
            
            if fundamentals['market_cap'] <= 0:
                log_warning(f"❌ {ticker}: Invalid market cap data - {fundamentals['market_cap']}")
                return None
            
            # Cache successful result
            self._cache_fundamentals(ticker, fundamentals)
            self.fundamentals_cache[ticker] = fundamentals
            
            log_debug(f"✅ {ticker}: Successfully compiled fundamentals - Price: ${fundamentals['price']}, Cap: ${fundamentals['market_cap']:,}")
            return fundamentals
            
        except Exception as e:
            log_error(f"❌ Error compiling fundamentals for {ticker}: {e}")
            return None
    
    def _check_options_availability(self, ticker: str) -> bool:
        """Check if options are available for the ticker"""
        if not self.criteria.require_options:
            return True
        
        try:
            # Try to get options chain - if it exists, options are available
            options_data = self.fmp_loader.make_request(f"options-chain/{ticker}")
            return bool(options_data and len(options_data) > 0)
        except:
            # If we can't check options, assume major stocks have them
            # This is a reasonable fallback for well-established companies
            return True
    
    def _evaluate_criteria(self, fundamentals: Dict[str, Any]) -> tuple[bool, str]:
        """Check if ticker meets all filter criteria"""
        try:
            # Price range check
            price = fundamentals.get('price', 0)
            if price < self.criteria.min_price:
                return False, f"Price ${price:.2f} below minimum ${self.criteria.min_price:.2f}"
            if price > self.criteria.max_price:
                return False, f"Price ${price:.2f} above maximum ${self.criteria.max_price:.2f}"
            
            # Market cap check
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap < self.criteria.min_market_cap:
                return False, f"Market cap ${market_cap:,} below minimum ${self.criteria.min_market_cap:,}"
            
            # Volume requirements
            avg_volume = fundamentals.get('avg_volume', 0)
            if avg_volume < self.criteria.min_avg_volume:
                return False, f"Avg volume {avg_volume:,} below minimum {self.criteria.min_avg_volume:,}"
            
            # Dollar volume check
            dollar_volume = price * avg_volume
            if dollar_volume < self.criteria.min_dollar_volume:
                return False, f"Dollar volume ${dollar_volume:,.0f} below minimum ${self.criteria.min_dollar_volume:,}"
            
            # Beta (volatility) check
            beta = fundamentals.get('beta', 0)
            if beta > self.criteria.max_volatility_beta:
                return False, f"Beta {beta:.2f} above maximum {self.criteria.max_volatility_beta:.2f}"
            
            # Exchange check
            exchange = fundamentals.get('exchange', '').strip()
            if exchange and exchange not in self.criteria.allowed_exchanges:
                return False, f"Exchange '{exchange}' not in allowed list {self.criteria.allowed_exchanges}"
            
            # Options availability check
            has_options = fundamentals.get('has_options', False)
            if self.criteria.require_options and not has_options:
                return False, f"Options not available but required"
            
            return True, "Passed all criteria"
            
        except Exception as e:
            return False, f"Error evaluating criteria: {e}"
    
    def _get_cached_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached fundamental data if still valid"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute('''
                    SELECT price, market_cap, exchange, beta, avg_volume, volume, has_options, cached_at
                    FROM fundamentals_cache 
                    WHERE ticker = ? AND cached_at > datetime('now', '-24 hours')
                ''', (ticker,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'price': row[0],
                        'market_cap': row[1],
                        'exchange': row[2],
                        'beta': row[3],
                        'avg_volume': row[4],
                        'volume': row[5],
                        'has_options': bool(row[6])
                    }
                    
        except Exception as e:
            log_debug(f"Error reading cache for {ticker}: {e}")
        
        return None
    
    def _cache_fundamentals(self, ticker: str, fundamentals: Dict[str, Any]) -> None:
        """Cache fundamental data"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO fundamentals_cache 
                    (ticker, price, market_cap, exchange, beta, avg_volume, volume, has_options, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (
                    ticker,
                    fundamentals.get('price', 0),
                    fundamentals.get('market_cap', 0),
                    fundamentals.get('exchange', ''),
                    fundamentals.get('beta', 0),
                    fundamentals.get('avg_volume', 0),
                    fundamentals.get('volume', 0),
                    fundamentals.get('has_options', False)
                ))
                conn.commit()
                
        except Exception as e:
            log_debug(f"Error caching fundamentals for {ticker}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging"""
        stats = {
            'memory_cache_size': len(self.fundamentals_cache),
            'memory_cached_tickers': list(self.fundamentals_cache.keys())[:10]  # First 10
        }
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM fundamentals_cache')
                total_cached = cursor.fetchone()[0]
                
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM fundamentals_cache 
                    WHERE cached_at > datetime('now', '-24 hours')
                ''')
                valid_cached = cursor.fetchone()[0]
                
                stats.update({
                    'total_entries': total_cached,
                    'valid_entries': valid_cached,
                    'cache_path': str(self.cache_db_path)
                })
                
            if hasattr(self.fmp_loader, 'get_rate_limit_status'):
                stats['rate_limit'] = self.fmp_loader.get_rate_limit_status()
        except Exception as e:
            stats['cache_error'] = str(e)
        
        return stats
    
    def clear_failed_ticker_cache(self) -> None:
        """Clear the failed ticker cache - useful for debugging"""
        try:
            if hasattr(self.fmp_loader, '_clear_failed_ticker_cache'):
                self.fmp_loader._clear_failed_ticker_cache()
                log_info("✅ Cleared failed ticker cache")
            else:
                log_warning("⚠️ Failed ticker cache clearing not available")
        except Exception as e:
            log_error(f"❌ Error clearing failed ticker cache: {e}")
