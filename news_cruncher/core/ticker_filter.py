"""
TickerFilterEngine for filtering tickers based on fundamental criteria
Python 3.13.3 compatible
"""
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_debug, log_warning, log_error


@dataclass
class FilterCriteria:
    """Criteria for filtering tickers based on fundamental analysis"""
    min_price: float = 10.00  # Eliminates penny stocks
    max_price: float = 300.00  # Avoids ultra-expensive stocks
    min_avg_volume: int = 500_000  # 500K shares daily (good liquidity)
    min_dollar_volume: int = 5_000_000  # $5M daily (institutional interest)
    min_market_cap: int = 500_000_000  # $500M+ (established companies)
    max_volatility_beta: float = 2.0  # Reasonable volatility
    require_options: bool = True  # Options = liquidity + institutional interest
    allowed_exchanges: List[str] = None  # Major exchanges only
    
    def __post_init__(self):
        if self.allowed_exchanges is None:
            self.allowed_exchanges = ['NASDAQ', 'NYSE', 'NYSEArca']


class TickerFilterEngine:
    """Filter ticker buckets based on fundamental criteria"""
    
    def __init__(self, fmp_loader: BaseFMPLoader, filter_criteria: FilterCriteria) -> None:
        """Initialize ticker filter engine"""
        self.fmp_loader = fmp_loader
        self.criteria = filter_criteria
        self.fundamentals_cache = {}
        self.cache_db_path = Path('data/fundamentals_cache.db')
        
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
    
    def _init_cache_database(self) -> None:
        """Initialize SQLite database for caching fundamental data"""
        try:
            # Ensure data directory exists
            self.cache_db_path.parent.mkdir(exist_ok=True)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS fundamentals_cache (
                        ticker TEXT PRIMARY KEY,
                        price REAL,
                        market_cap INTEGER,
                        exchange TEXT,
                        beta REAL,
                        avg_volume INTEGER,
                        has_options BOOLEAN,
                        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_cached_at ON fundamentals_cache(cached_at)
                ''')
                conn.commit()
                
        except Exception as e:
            log_error(f"Failed to initialize cache database: {e}")
    
    def filter_ticker_buckets(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Filter ticker buckets based on fundamental criteria"""
        if not ticker_buckets:
            return {}
        
        log_info(f"🔍 Filtering {len(ticker_buckets)} tickers by fundamental criteria...")
        
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
        
        for ticker, articles in ticker_buckets.items():
            try:
                fundamentals = self._get_company_fundamentals(ticker)
                
                if fundamentals is None:
                    filter_stats['failed_data_unavailable'] += 1
                    log_debug(f"❌ {ticker}: No fundamental data available")
                    continue
                
                passes_filter, reason = self._meets_criteria(ticker, fundamentals)
                
                if passes_filter:
                    filtered_buckets[ticker] = articles
                    filter_stats['passed'] += 1
                    log_debug(f"✅ {ticker}: Passed all criteria")
                else:
                    # Track specific failure reasons
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
                    
            except Exception as e:
                filter_stats['failed_data_unavailable'] += 1
                log_debug(f"❌ {ticker}: Error getting fundamentals - {e}")
                continue
        
        # Log filtering statistics
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
        
        return filtered_buckets
    
    def _get_company_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch company fundamental data with caching"""
        # Check memory cache first
        if ticker in self.fundamentals_cache:
            return self.fundamentals_cache[ticker]
        
        # Check database cache (24 hour expiry)
        cached_data = self._get_cached_fundamentals(ticker)
        if cached_data:
            self.fundamentals_cache[ticker] = cached_data
            return cached_data
        
        # Fetch from API
        try:
            # Get company profile for basic info
            profile_data = self.fmp_loader.make_request(f"profile/{ticker}")
            if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
                return None
            
            profile = profile_data[0]
            
            # Get current quote for volume data
            quote_data = self.fmp_loader.make_request(f"quote/{ticker}")
            if not quote_data or not isinstance(quote_data, list) or len(quote_data) == 0:
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
                'avg_volume': int(quote.get('avgVolume', 0) or quote.get('volume', 0)),
                'volume': int(quote.get('volume', 0) or 0),
                'has_options': has_options
            }
            
            # Cache the data
            self._cache_fundamentals(ticker, fundamentals)
            self.fundamentals_cache[ticker] = fundamentals
            
            return fundamentals
            
        except Exception as e:
            log_debug(f"Error fetching fundamentals for {ticker}: {e}")
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
    
    def _meets_criteria(self, ticker: str, fundamentals: Dict[str, Any]) -> tuple[bool, str]:
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
            
            # Volume checks
            avg_volume = fundamentals.get('avg_volume', 0)
            if avg_volume < self.criteria.min_avg_volume:
                return False, f"Avg volume {avg_volume:,} below minimum {self.criteria.min_avg_volume:,}"
            
            # Dollar volume check (price * avg_volume)
            dollar_volume = price * avg_volume
            if dollar_volume < self.criteria.min_dollar_volume:
                return False, f"Dollar volume ${dollar_volume:,.0f} below minimum ${self.criteria.min_dollar_volume:,}"
            
            # Beta (volatility) check
            beta = fundamentals.get('beta', 0)
            if beta > self.criteria.max_volatility_beta:
                return False, f"Beta {beta:.2f} above maximum {self.criteria.max_volatility_beta:.2f}"
            
            # Exchange check
            exchange = fundamentals.get('exchange', '').upper()
            if exchange not in [ex.upper() for ex in self.criteria.allowed_exchanges]:
                return False, f"Exchange '{exchange}' not in allowed list {self.criteria.allowed_exchanges}"
            
            # Options availability check
            if self.criteria.require_options:
                has_options = fundamentals.get('has_options', False)
                if not has_options:
                    return False, "Options not available"
            
            return True, "Passed all criteria"
            
        except Exception as e:
            return False, f"Error evaluating criteria: {e}"
    
    def _get_cached_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached fundamental data if still valid (24 hour expiry)"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    '''SELECT * FROM fundamentals_cache 
                       WHERE ticker = ? AND cached_at > datetime('now', '-24 hours')''',
                    (ticker,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'price': row['price'],
                        'market_cap': row['market_cap'],
                        'exchange': row['exchange'],
                        'beta': row['beta'],
                        'avg_volume': row['avg_volume'],
                        'has_options': bool(row['has_options'])
                    }
        except Exception as e:
            log_debug(f"Error reading cache for {ticker}: {e}")
        
        return None
    
    def _cache_fundamentals(self, ticker: str, fundamentals: Dict[str, Any]) -> None:
        """Cache fundamental data in database"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    '''INSERT OR REPLACE INTO fundamentals_cache 
                       (ticker, price, market_cap, exchange, beta, avg_volume, has_options, cached_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))''',
                    (
                        ticker,
                        fundamentals.get('price', 0),
                        fundamentals.get('market_cap', 0),
                        fundamentals.get('exchange', ''),
                        fundamentals.get('beta', 0),
                        fundamentals.get('avg_volume', 0),
                        fundamentals.get('has_options', False)
                    )
                )
                conn.commit()
        except Exception as e:
            log_debug(f"Error caching fundamentals for {ticker}: {e}")
    
    def get_filter_summary(self, original_count: int, filtered_count: int) -> Dict[str, Any]:
        """Get summary of filtering results"""
        return {
            'original_ticker_count': original_count,
            'filtered_ticker_count': filtered_count,
            'tickers_removed': original_count - filtered_count,
            'filter_efficiency': (original_count - filtered_count) / original_count if original_count > 0 else 0,
            'criteria_used': {
                'min_price': self.criteria.min_price,
                'max_price': self.criteria.max_price,
                'min_avg_volume': self.criteria.min_avg_volume,
                'min_dollar_volume': self.criteria.min_dollar_volume,
                'min_market_cap': self.criteria.min_market_cap,
                'max_volatility_beta': self.criteria.max_volatility_beta,
                'require_options': self.criteria.require_options,
                'allowed_exchanges': self.criteria.allowed_exchanges
            }
        }
