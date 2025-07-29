"""
Base FMP API loader with rate limiting
Python 3.13.3 compatible
"""
import time
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, date # Fix 1: Add date import
import pytz # Added for timestamp parsing
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
        """OPTIMIZED: Intelligent rate limiting that adapts to actual usage"""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.start_time >= 60:
            self.request_count = 0
            self.start_time = current_time
            log_debug(f"🔄 Rate limit counter reset - new minute window")

        # OPTIMIZATION: Use higher default rate limit
        max_requests = getattr(Config, 'FMP_REQUESTS_PER_MINUTE', 60)  # Increased default

        # Check if we're approaching the limit
        if self.request_count >= max_requests:
            sleep_time = 60 - (current_time - self.start_time)
            if sleep_time > 0:
                log_warning(f"⏳ Rate limit reached ({self.request_count}/{max_requests}), sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.start_time = time.time()

        # OPTIMIZATION: Reduced minimum delay for better throughput
        min_delay = getattr(Config, 'FMP_MIN_REQUEST_INTERVAL', 0.1)  # Reduced from 0.2s

        # Calculate time since last request
        elapsed_since_last = current_time - self.last_request_time

        # Only sleep if we haven't waited long enough
        if elapsed_since_last < min_delay:
            sleep_time = min_delay - elapsed_since_last
            log_debug(f"⏳ Request spacing: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        # Update tracking
        self.last_request_time = time.time()
        self.request_count += 1

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make API request with rate limiting and failed ticker caching to prevent 429 errors"""
        import time

        # Extract ticker from endpoint for cache checking
        ticker = self._extract_ticker_from_endpoint(endpoint)

        # ADD THIS DIAGNOSTIC BLOCK:
        if ticker:
            log_debug(f"🔍 API Request for {ticker}: endpoint={endpoint}")

            # Check if API key is available
            if not self.api_key or self.api_key.strip() == '':
                log_error(f"❌ Missing FMP API key for {ticker} request")
                return None

        # NEW: Check failed ticker cache first
        if ticker and self._is_ticker_cached_as_failed(ticker):
            # ADD ENHANCED LOGGING FOR CACHE HITS:
            log_warning(f"⏭️ Skipping cached failed ticker {ticker} (endpoint: {endpoint}) - Consider clearing cache if this seems wrong")
            return None

        # Use comprehensive rate limiting
        self._rate_limit()
        try:
            # Build URL
            if params is None:
                params = {}
            params['apikey'] = self.api_key

            url = f"{self.base_url}/{endpoint}"

            # ADD DIAGNOSTIC LOGGING:
            log_debug(f"📡 Making request to: {url[:100]}...")

            # Make request with timeout and retries
            response = requests.get(url, params=params, timeout=30)
            self.last_request_time = time.time()

            # ADD ENHANCED STATUS CODE LOGGING:
            if response.status_code != 200:
                log_warning(f"❌ API returned status {response.status_code} for {ticker or endpoint}")
                if response.status_code == 401:
                    log_error("🔑 AUTHENTICATION ERROR: Check your FMP_API_KEY in .env file")
                elif response.status_code == 403:
                    log_error("🚫 FORBIDDEN: Your FMP API key may have insufficient permissions")

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
            data = response.json()

            # ADD SUCCESS LOGGING:
            if ticker and data:
                log_debug(f"✅ Successfully fetched data for {ticker}")

            return data

        except requests.exceptions.ConnectionError as e:
            # Specific handling for DNS failures, refused connections, etc.
            log_error(f"❌ Network connection error for {endpoint} (ticker: {ticker}): {e}. Check network/DNS.")
            if ticker:
                self._add_failed_ticker_to_cache(ticker, "connection_error")
            return None
        except requests.exceptions.Timeout as e:
            log_error(f"❌ Request timed out for {endpoint} (ticker: {ticker}): {e}")
            if ticker:
                self._add_failed_ticker_to_cache(ticker, "timeout_error")
            return None
        except requests.exceptions.RequestException as e: # General catch-all for other request issues
            # ENHANCED ERROR LOGGING:
            error_msg = str(e)
            log_error(f"❌ Request error for {endpoint} (ticker: {ticker}): {error_msg}")

            # NEW: Cache tickers that consistently fail
            if ticker:
                self._add_failed_ticker_to_cache(ticker, f"request_error_{type(e).__name__}")
            return None
        except Exception as e:
            log_error(f"❌ Unexpected error for {endpoint} (ticker: {ticker}): {e}")
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

    def parse_fmp_timestamp(self, timestamp_str: str) -> datetime:
        """Parse FMP timestamp with proper timezone handling"""
        if not timestamp_str:
            return datetime.now(pytz.utc)

        try:
            if 'T' in timestamp_str:
                # ISO format with timezone
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # FMP format: assume Eastern Time and convert to UTC
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                est_tz = pytz.timezone('US/Eastern')
                return est_tz.localize(dt).astimezone(pytz.utc)
        except Exception as e:
            log_warning(f"Error parsing timestamp '{timestamp_str}': {e}")
            return datetime.now(pytz.utc)

    # Add these methods starting at line ~400 (end of class, before last })

    def get_intraday_prices(self, ticker: str, target_date: date) -> Optional[List[Dict]]:
        """Get intraday price data for specific date"""
        try:
            # FMP intraday endpoint format: historical-chart/1min/TICKER?from=DATE&to=DATE
            date_str = target_date.strftime('%Y-%m-%d')
            endpoint = f"historical-chart/1min/{ticker}"

            params = {
                'from': date_str,
                'to': date_str
            }

            data = self.make_request(endpoint, params)

            if data and isinstance(data, list):
                # Convert FMP format to expected format
                intraday_data = []
                for item in data:
                    if 'date' in item and 'close' in item:
                        # Extract time from datetime string
                        datetime_str = item.get('date', '')
                        try:
                            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')

                            intraday_data.append({
                                'time': time_str,
                                'close': item.get('close'),
                                'high': item.get('high'),
                                'low': item.get('low'),
                                'open': item.get('open'),
                                'volume': item.get('volume')
                            })
                        except (ValueError, TypeError):
                            continue

                log_debug(f"✅ Got {len(intraday_data)} intraday points for {ticker} on {date_str}")
                return intraday_data

            log_debug(f"❌ No intraday data for {ticker} on {date_str}")
            return None

        except Exception as e:
            log_debug(f"❌ Intraday request failed for {ticker}: {e}")
            return None

    def get_historical_daily_prices(self, ticker: str, start_date: date, end_date: date) -> Optional[List[Dict]]:
        """Get historical daily price data for date range"""
        try:
            # FMP historical endpoint format: historical-price-full/TICKER?from=START&to=END
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            endpoint = f"historical-price-full/{ticker}"

            params = {
                'from': start_str,
                'to': end_str
            }

            data = self.make_request(endpoint, params)

            if data and 'historical' in data:
                historical_data = data['historical']

                if historical_data and isinstance(historical_data, list):
                    log_debug(f"✅ Got {len(historical_data)} daily prices for {ticker} ({start_str} to {end_str})")
                    return historical_data

            log_debug(f"❌ No historical data for {ticker} ({start_str} to {end_str})")
            return None

        except Exception as e:
            log_debug(f"❌ Historical request failed for {ticker}: {e}")
            return None