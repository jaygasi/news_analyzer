import requests
import time
from typing import List, Dict, Any, Union # Import Union for type hinting
from config import FMP_API_KEY, FMP_BASE_URL, FMP_ENDPOINTS
from utils.utils import get_logger, retry_api_call
from utils.exceptions import FMPAPIError, MissingConfigError

logger = get_logger(__name__)

class BaseFMPLoader:
    """
    Base class for interacting with the Financial Modeling Prep (FMP) API.
    Handles API key, base URL, rate limiting, and error handling.
    """
    def __init__(self, api_key: str = FMP_API_KEY, base_url: str = FMP_BASE_URL):
        if not api_key or api_key == "YOUR_FMP_API_KEY":
            raise MissingConfigError("FMP API Key not set. Please set FMP_API_KEY in config.py or as an environment variable.")
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session() # Use a session for better performance with multiple requests
        self.last_request_time = 0
        self.rate_limit_delay = 0.1 # Minimum delay between requests to avoid hitting limits (e.g., 100 requests/second for free tier)

    def _build_url(self, endpoint: str, **kwargs: Any) -> str:
        """Constructs the full API URL for a given endpoint."""
        url = f"{self.base_url}{endpoint}"
        params: Dict[str, Union[str, int]] = {k: v for k, v in kwargs.items() if v is not None} # Explicitly type params dict
        params["apikey"] = self.api_key
        
        # Replace path parameters if any (e.g., {symbol} in /historical-price/{symbol})
        for key, value in list(params.items()):
            placeholder = "{" + key + "}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))
                del params[key] # Remove it from query params

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{url}?{query_string}" if query_string else url

    @retry_api_call
    def _make_request(self, endpoint: str, **kwargs: Any) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """
        Makes a GET request to the FMP API.
        Includes rate limiting and basic error handling.
        Returns a dict, a list of dicts, or None.
        """
        url = self._build_url(endpoint, **kwargs)
        
        # Implement rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

        try:
            response = self.session.get(url, timeout=10) # 10-second timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            if not data:
                logger.warning(f"No data returned for endpoint {endpoint} with params {kwargs}. URL: {url}")
                return None
            return data
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e} - Response: {e.response.text}")
            raise FMPAPIError(f"FMP API HTTP error: {e.response.status_code} - {e.response.text}") from e
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise FMPAPIError(f"FMP API connection error: {e}") from e
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {url}: {e}")
            raise FMPAPIError(f"FMP API timeout error: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"An unexpected request error occurred for {url}: {e}")
            raise FMPAPIError(f"FMP API request error: {e}") from e
        except ValueError as e: # JSONDecodeError inherits from ValueError
            logger.error(f"JSON decoding error for {url}: {e} - Response: {response.text}")
            raise FMPAPIError(f"FMP API JSON decoding error: {e}") from e

    def get_stock_list(self) -> List[Dict[str, Any]]:
        """Fetches a list of all supported stocks from FMP."""
        logger.info("Fetching stock list from FMP...")
        data = self._make_request(FMP_ENDPOINTS["stock_list"])
        if isinstance(data, list): # Ensure it's a list
            logger.info(f"Fetched {len(data)} stocks.")
            return data
        logger.warning(f"Unexpected data type for stock list: {type(data)}. Expected list.")
        return []

    def get_historical_price(self, symbol: str, from_date: str | None = None, to_date: str | None = None) -> List[Dict[str, Any]]:
        """
        Fetches historical price data for a given symbol.
        Removed unused 'interval' parameter.
        """
        endpoint = FMP_ENDPOINTS["historical_price"].format(symbol=symbol)
        params: Dict[str, Any] = {"serietype": "line"} # FMP default, 'line' means close price
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        logger.info(f"Fetching historical price for {symbol} from {from_date} to {to_date}...")
        data = self._make_request(endpoint, **params)
        if isinstance(data, list): # Ensure it's a list
            return data
        logger.warning(f"Unexpected data type for historical price: {type(data)}. Expected list.")
        return []

    def get_quote(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetches real-time quote data for a given symbol."""
        endpoint = FMP_ENDPOINTS["quote"].format(symbol=symbol)
        logger.info(f"Fetching quote for {symbol}...")
        data = self._make_request(endpoint)
        if isinstance(data, list): # Ensure it's a list
            return data
        logger.warning(f"Unexpected data type for quote: {type(data)}. Expected list.")
        return []
    
    def get_news(self, endpoint_key: str, tickers: str | None = None, limit: int = 50, from_date: str | None = None, to_date: str | None = None) -> List[Dict[str, Any]]:
        """
        Fetches news from a specific FMP news endpoint.
        `tickers` should be a comma-separated string (e.g., "AAPL,MSFT").
        """
        endpoint = FMP_ENDPOINTS.get(endpoint_key)
        if not endpoint:
            logger.error(f"Unknown FMP news endpoint key: {endpoint_key}")
            return []
        
        params: Dict[str, Any] = {"limit": limit}
        if tickers:
            params["tickers"] = tickers
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        logger.debug(f"Fetching news from {endpoint_key} for tickers: {tickers if tickers else 'ALL'}...")
        data = self._make_request(endpoint, **params)
        if isinstance(data, list): # Ensure it's a list
            return data
        logger.warning(f"Unexpected data type for news: {type(data)}. Expected list.")
        return []