"""
Base FMP data loader with core functionality and session management
"""
import requests
import time
from typing import Optional, Dict, Any, Union, List
from utils.simple_logger import log_error, log_debug


class BaseFMPLoader:
    """Base FMP loader with core functionality and session management."""
    
    # Class-level constants
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"
    TIMEOUT = 30
    MIN_REQUEST_INTERVAL = 0.3
    
    def __init__(self, api_key: str) -> None:
        """Initialize base FMP loader with validated API key."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self._setup_session()
        self._setup_rate_limiting()
    
    def _setup_session(self) -> None:
        """Setup optimized HTTP session."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('https://', adapter)
    
    def _setup_rate_limiting(self) -> None:
        """Initialize rate limiting."""
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     use_v4: bool = False) -> Optional[Union[Dict, List]]:
        """Make API request with comprehensive error handling."""
        self._rate_limit()
        
        request_params = params or {}
        request_params['apikey'] = self.api_key
        
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
            
        except requests.exceptions.Timeout:
            log_error(f"Timeout requesting {endpoint}")
        except requests.exceptions.ConnectionError:
            log_error(f"Connection error for {endpoint}")
        except requests.exceptions.HTTPError as e:
            log_error(f"HTTP error {e.response.status_code} for {endpoint}")
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
    
    def __del__(self):
        """Clean up session resources."""
        if hasattr(self, 'session'):
            self.session.close()