"""
Base FMP API loader with rate limiting
Python 3.13.3 compatible
"""
import requests
import time
from typing import Dict, Any, Optional, Union, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.simple_logger import log_error, log_debug, log_warning
from config import Config


class BaseFMPLoader:
    """Base class for FMP API interactions with rate limiting"""
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"
    TIMEOUT = 30
    
    def __init__(self, api_key: str) -> None:
        """Initialize FMP loader with API key"""
        if not api_key:
            raise ValueError("FMP API key is required")
        
        self.api_key = api_key
        self.session = self._create_session()
        self.last_request_time = 0.0
        self.request_count = 0
        self.start_time = time.time()
    
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
    
    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                    use_v4: bool = False) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Make API request with rate limiting and error handling"""
        self._rate_limit()
        
        # Prepare parameters
        request_params = params or {}
        request_params['apikey'] = self.api_key
        
        # Build URL
        base_url = self.BASE_URL_V4 if use_v4 else self.BASE_URL
        url = f"{base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=request_params, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                log_debug(f"Empty response from {endpoint}")
                return None
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                log_warning(f"Rate limit hit for {endpoint}, backing off")
                time.sleep(60)  # Wait 1 minute for rate limit reset
                return None
            else:
                log_error(f"HTTP error {e.response.status_code if e.response else 'unknown'} for {endpoint}")
                
        except requests.exceptions.Timeout:
            log_error(f"Timeout for {endpoint}")
            
        except requests.exceptions.ConnectionError:
            log_error(f"Connection error for {endpoint}")
            
        except requests.exceptions.RequestException as e:
            log_error(f"Request error for {endpoint}: {e}")
            
        except ValueError as e:
            log_error(f"JSON decode error for {endpoint}: {e}")
            
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