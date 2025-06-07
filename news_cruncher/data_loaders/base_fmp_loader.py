"""
Base FMP API loader with rate limiting
Python 3.13.3 compatible
"""
import time
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from config import Config
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.simple_logger import log_info, log_error, log_debug, log_warning

class BaseFMPLoader:
    """Base class for FMP API interactions with rate limiting"""
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"
    TIMEOUT = 30

    def __init__(self, api_key: str) -> None:
        """Initialize FMP API loader with rate limiting and failed ticker caching"""
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = getattr(Config, 'FMP_MIN_REQUEST_INTERVAL', 0.2)
        
        # ADD THESE IF MISSING:
        self.start_time = time.time()  # For rate limit tracking
        self.request_count = 0         # For rate limit tracking
    
        # NEW: Failed ticker cache
        self.enable_cache = getattr(Config, 'ENABLE_FAILED_TICKER_CACHE', True)
        self.cache_db_path = Path(getattr(Config, 'FAILED_TICKER_CACHE_DB', 'data/failed_tickers_cache.db'))
        self.cache_days = getattr(Config, 'FAILED_TICKER_CACHE_DAYS', 30)
        self.max_retries = getattr(Config, 'FAILED_TICKER_MAX_RETRIES', 3)
        
        # Initialize cache database
        if self.enable_cache:
            self._init_failed_ticker_cache()
    
    def _create_session(self) -> requests.Session:
        """Create optimized HTTP session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers
        session.headers.update({
            'User-Agent': 'FinancialNewsAnalyzer/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def _rate_limit(self) -> None:
        """Simple rate limiting to respect API limits"""
        current_time = time.time()
        elapsed_since_last = current_time - self.last_request_time
        
        # Reset counter every minute
        if current_time - self.start_time >= 60:
            self.request_count = 0
            self.start_time = current_time
        
        # Check rate limit
        if self.request_count >= Config.FMP_REQUESTS_PER_MINUTE:
            sleep_time = 60 - (current_time - self.start_time)
            if sleep_time > 0:
                log_warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.start_time = time.time()
        
        # Minimum delay between requests
        min_delay = 60.0 / Config.FMP_REQUESTS_PER_MINUTE
        if elapsed_since_last < min_delay:
            time.sleep(min_delay - elapsed_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and failed ticker caching to prevent 429 errors"""
        import time
        
        # Extract ticker from endpoint for cache checking
        ticker = self._extract_ticker_from_endpoint(endpoint)
        
        # NEW: Check failed ticker cache first
        if ticker and self._is_ticker_cached_as_failed(ticker):
            log_debug(f"⏭️ Skipping cached failed ticker {ticker} (endpoint: {endpoint})")
            return None
        
        # Rate limiting - ensure minimum interval between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        try:
            # Build URL
            if params is None:
                params = {}
            params['apikey'] = self.api_key
            
            url = f"{self.base_url}/{endpoint}"
            
            # Make request with timeout and retries
            response = requests.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            if response.status_code == 429:
                retry_delay = getattr(Config, 'FMP_RETRY_DELAY', 60)
                log_warning(f"Rate limit hit for {endpoint}, waiting {retry_delay} seconds...")
                
                # NEW: Cache ticker that causes 429 errors
                if ticker:
                    self._add_failed_ticker_to_cache(ticker, "rate_limit_429")
                    log_debug(f"🚫 Cached ticker {ticker} due to repeated 429 errors")
                
                time.sleep(retry_delay)
                # Retry once after rate limit
                response = requests.get(url, params=params, timeout=30)
                self.last_request_time = time.time()
            
            if response.status_code == 404:
                # NEW: Cache 404 not found tickers
                if ticker:
                    self._add_failed_ticker_to_cache(ticker, "not_found_404")
                    log_debug(f"🚫 Cached ticker {ticker} - not found (404)")
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            # NEW: Cache tickers that consistently fail
            if ticker and "429" in str(e):
                self._add_failed_ticker_to_cache(ticker, "request_error_429")
                log_debug(f"🚫 Cached ticker {ticker} due to request errors")
            
            log_error(f"Request error for {endpoint}: {e}")
            return None
        except Exception as e:
            log_error(f"Unexpected error for {endpoint}: {e}")
            return None
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        current_time = time.time()
        time_window = current_time - self.start_time
        
        return {
            'requests_made': self.request_count,
            'requests_limit': Config.FMP_REQUESTS_PER_MINUTE,
            'time_window_seconds': time_window,
            'requests_remaining': max(0, Config.FMP_REQUESTS_PER_MINUTE - self.request_count)
        }

    def _extract_ticker_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract ticker symbol from API endpoint for caching"""
        try:
            # Handle common FMP endpoints: profile/TICKER, quote/TICKER, etc.
            if '/' in endpoint:
                parts = endpoint.split('/')
                if len(parts) >= 2:
                    potential_ticker = parts[-1].upper()
                    # Basic validation - ticker should be 1-5 chars, letters only
                    if 1 <= len(potential_ticker) <= 6 and potential_ticker.isalpha():
                        return potential_ticker
            return None
        except Exception:
            return None
        

    def _init_failed_ticker_cache(self) -> None:
        """Initialize SQLite cache for failed tickers"""
        try:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS failed_tickers (
                        ticker TEXT PRIMARY KEY,
                        failure_reason TEXT NOT NULL,
                        failure_count INTEGER DEFAULT 1,
                        first_failed_at TIMESTAMP NOT NULL,
                        last_failed_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_expires_at 
                    ON failed_tickers(expires_at)
                ''')
                
                conn.commit()
                
        except Exception as e:
            log_error(f"Error initializing failed ticker cache: {e}")
            self.enable_cache = False  # Disable cache if can't initialize

    def _is_ticker_cached_as_failed(self, ticker: str) -> bool:
        """Check if ticker is in failed cache and not expired"""
        if not self.enable_cache:
            return False
            
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute('''
                    SELECT failure_reason, failure_count, expires_at 
                    FROM failed_tickers 
                    WHERE ticker = ? AND expires_at > datetime('now')
                ''', (ticker,))
                
                result = cursor.fetchone()
                return result is not None
                
        except Exception as e:
            log_error(f"Error checking failed ticker cache: {e}")
            return False

    def _add_failed_ticker_to_cache(self, ticker: str, reason: str) -> None:
        """Add or update failed ticker in cache"""
        if not self.enable_cache:
            return
            
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(days=self.cache_days)
            
            with sqlite3.connect(self.cache_db_path) as conn:
                # Check if ticker already exists
                cursor = conn.execute(
                    'SELECT failure_count FROM failed_tickers WHERE ticker = ?', 
                    (ticker,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    new_count = existing[0] + 1
                    conn.execute('''
                        UPDATE failed_tickers 
                        SET failure_count = ?, last_failed_at = ?, expires_at = ?, failure_reason = ?
                        WHERE ticker = ?
                    ''', (new_count, now, expires_at, reason, ticker))
                else:
                    # Insert new record
                    conn.execute('''
                        INSERT INTO failed_tickers 
                        (ticker, failure_reason, failure_count, first_failed_at, last_failed_at, expires_at)
                        VALUES (?, ?, 1, ?, ?, ?)
                    ''', (ticker, reason, now, now, expires_at))
                
                conn.commit()
                
        except Exception as e:
            log_error(f"Error adding failed ticker to cache: {e}")

    def _cleanup_expired_failed_tickers(self) -> int:
        """Remove expired entries from failed ticker cache"""
        if not self.enable_cache:
            return 0
            
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.execute('''
                    DELETE FROM failed_tickers 
                    WHERE expires_at < datetime('now')
                ''')
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    log_debug(f"Cleaned up {deleted_count} expired failed ticker entries")
                
                return deleted_count
                
        except Exception as e:
            log_error(f"Error cleaning up failed ticker cache: {e}")
            return 0
        

    def get_failed_ticker_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the failed ticker cache"""
        if not self.enable_cache:
            return {'cache_enabled': False}
        
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # Total cached tickers
                total_cursor = conn.execute('SELECT COUNT(*) FROM failed_tickers')
                total_cached = total_cursor.fetchone()[0]
                
                # Active (non-expired) tickers
                active_cursor = conn.execute('''
                    SELECT COUNT(*) FROM failed_tickers 
                    WHERE expires_at > datetime('now')
                ''')
                active_cached = active_cursor.fetchone()[0]
                
                # By failure reason
                reason_cursor = conn.execute('''
                    SELECT failure_reason, COUNT(*) 
                    FROM failed_tickers 
                    WHERE expires_at > datetime('now')
                    GROUP BY failure_reason
                ''')
                by_reason = dict(reason_cursor.fetchall())
                
                return {
                    'cache_enabled': True,
                    'total_cached_tickers': total_cached,
                    'active_cached_tickers': active_cached,
                    'expired_cached_tickers': total_cached - active_cached,
                    'failure_reasons': by_reason,
                    'cache_file': str(self.cache_db_path)
                }
                
        except Exception as e:
            log_error(f"Error getting cache stats: {e}")
            return {'cache_enabled': True, 'error': str(e)}

    def log_cache_stats(self) -> None:
        """Log cache statistics for monitoring"""
        stats = self.get_failed_ticker_cache_stats()
        
        if not stats.get('cache_enabled'):
            log_debug("Failed ticker cache is disabled")
            return
        
        if 'error' in stats:
            log_warning(f"Failed ticker cache error: {stats['error']}")
            return
        
        active = stats.get('active_cached_tickers', 0)
        if active > 0:
            log_info(f"🚫 Failed ticker cache: {active} active, {stats.get('expired_cached_tickers', 0)} expired")
            
            reasons = stats.get('failure_reasons', {})
            if reasons:
                reason_str = ", ".join([f"{reason}: {count}" for reason, count in reasons.items()])
                log_debug(f"   Breakdown: {reason_str}")
        else:
            log_debug("Failed ticker cache: empty (all tickers are working)")

    def periodic_cache_maintenance(self) -> None:
        """Perform periodic cache maintenance (cleanup expired entries)"""
        if not self.enable_cache:
            return
        
        try:
            # Clean up expired entries
            cleaned = self._cleanup_expired_failed_tickers()
            
            # Log stats periodically
            self.log_cache_stats()
            
        except Exception as e:
            log_error(f"Error in cache maintenance: {e}")