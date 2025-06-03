"""
Base FMP data loader with core functionality and enhanced rate limiting
"""
import requests
import time
from typing import Optional, Dict, Any, Union, List
from utils.simple_logger import log_error, log_debug, log_warning


class BaseFMPLoader:
    """Base FMP loader with enhanced rate limiting and session management."""
    
    # Class-level constants - More conservative for rate limiting
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"
    TIMEOUT = 30
    MIN_REQUEST_INTERVAL = 0.5  # Increased from 0.3 to 0.5 seconds
    MAX_RETRIES = 2  # Reduced from 3 to 2
    
    def __init__(self, api_key: str) -> None:
        """Initialize base FMP loader with validated API key."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self._setup_session()
        self._setup_enhanced_rate_limiting()
    
    def _setup_session(self) -> None:
        """Setup optimized HTTP session."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # More conservative connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=3,  # Reduced from 5
            pool_maxsize=6,      # Reduced from 10
            max_retries=self.MAX_RETRIES
        )
        self.session.mount('https://', adapter)
    
    def _setup_enhanced_rate_limiting(self) -> None:
        """Initialize enhanced rate limiting."""
        self.last_request_time = 0.0
        self.request_count = 0
        self.rate_limit_start = time.time()
        self.requests_per_minute = 0
        self.max_requests_per_minute = 100  # Conservative limit
        
        # Track 429 errors
        self.rate_limit_errors = 0
        self.last_429_time = 0.0
    
    def _enhanced_rate_limit(self) -> None:
        """Enhanced rate limiting with 429 error detection."""
        current_time = time.time()
        
        # Reset per-minute counter every minute
        if current_time - self.rate_limit_start >= 60:
            self.requests_per_minute = 0
            self.rate_limit_start = current_time
        
        # Check if we're approaching per-minute limit
        if self.requests_per_minute >= self.max_requests_per_minute - 5:
            wait_time = 60 - (current_time - self.rate_limit_start)
            if wait_time > 0:
                log_warning(f"Approaching rate limit, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.rate_limit_start = time.time()
        
        # Enforce minimum interval
        elapsed = current_time - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            sleep_time = self.MIN_REQUEST_INTERVAL - elapsed
            time.sleep(sleep_time)
        
        # Extra delay if we recently hit a 429 error
        if current_time - self.last_429_time < 10:  # 10 seconds after 429
            time.sleep(1.0)  # Extra 1 second delay
        
        self.last_request_time = time.time()
        self.requests_per_minute += 1
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     use_v4: bool = False) -> Optional[Union[Dict, List]]:
        """Make API request with enhanced error handling and rate limiting."""
        self._enhanced_rate_limit()
        
        request_params = params or {}
        request_params['apikey'] = self.api_key
        
        base_url = self.BASE_URL_V4 if use_v4 else self.BASE_URL
        url = f"{base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=request_params, timeout=self.TIMEOUT)
            
            # Handle 429 specifically
            if response.status_code == 429:
                self.rate_limit_errors += 1
                self.last_429_time = time.time()
                log_warning(f"Rate limit hit (429) for {endpoint}. Total 429 errors: {self.rate_limit_errors}")
                
                # Exponential backoff for 429 errors
                backoff_time = min(2 ** min(self.rate_limit_errors, 5), 30)  # Max 30 seconds
                log_warning(f"Backing off for {backoff_time}s due to rate limit")
                time.sleep(backoff_time)
                return None
            
            response.raise_for_status()
            
            data = response.json()
            if not data:
                log_debug(f"Empty response from {endpoint}")
                return None
            
            # Reset 429 counter on success
            if self.rate_limit_errors > 0:
                self.rate_limit_errors = max(0, self.rate_limit_errors - 1)
            
            return data
            
        except requests.exceptions.Timeout:
            log_error(f"Timeout requesting {endpoint}")
        except requests.exceptions.ConnectionError:
            log_error(f"Connection error for {endpoint}")
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response.status_code == 429:
                # Already handled above
                pass
            else:
                log_error(f"HTTP error {e.response.status_code if hasattr(e, 'response') else 'unknown'} for {endpoint}")
        except requests.exceptions.RequestException as e:
            log_error(f"Request error for {endpoint}: {e}")
        except ValueError as e:
            log_error(f"JSON decode error for {endpoint}: {e}")
        except Exception as e:
            log_error(f"Unexpected error for {endpoint}: {e}")
        
        return None
    
    def _validate_response(self, data: Any, endpoint_name: str) -> bool:
        """Validate API response."""
        if not data:
            log_error(f"No data from {endpoint_name}")
            return False
        
        if not isinstance(data, list):
            log_error(f"Expected list from {endpoint_name}, got: {type(data)}")
            return False
        
        log_debug(f"{endpoint_name} returned {len(data)} results")
        return True
    
    def _transform_article(self, article: Dict, source: str) -> Dict[str, Any]:
        """Transform article to standard format."""
        return {
            'symbol': article.get('symbol', ''),
            'title': article.get('title', ''),
            'text': article.get('text', ''),
            'url': article.get('url', ''),
            'publishedDate': article.get('publishedDate', ''),
            'site': article.get('site', ''),
            'source': source
        }
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        return {
            'requests_per_minute': self.requests_per_minute,
            'max_requests_per_minute': self.max_requests_per_minute,
            'rate_limit_errors': self.rate_limit_errors,
            'last_429_time': self.last_429_time,
            'min_request_interval': self.MIN_REQUEST_INTERVAL
        }
    
    def __del__(self):
        """Clean up session resources."""
        if hasattr(self, 'session'):
            self.session.close()