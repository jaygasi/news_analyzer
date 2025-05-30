"""
Optimized FMP client with improved error handling and request management
"""
import requests
from typing import Optional, Dict, Any, Union
import pandas as pd
from urllib.parse import urlencode
import time
from functools import wraps

from utils.log_utils import logd, logw, loge
from utils.string_utils import clean_string


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, max_calls: int = 300, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limits."""
        now = time.time()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        return len(self.calls) < self.max_calls
    
    def record_request(self) -> None:
        """Record a request."""
        self.calls.append(time.time())


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry requests on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    break
                except Exception as e:
                    # Don't retry on non-request exceptions
                    raise e
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


class OptimizedFmpClient:
    """
    Optimized FMP client with rate limiting and improved error handling.
    """
    
    def __init__(self, fmp_api_key: str, rate_limit: int = 300, timeout: int = 30):
        if not fmp_api_key:
            raise ValueError("FMP API key is required")
        
        self._api_key = fmp_api_key.strip()
        self.base_url = "https://financialmodelingprep.com/api"
        self.timeout = timeout
        self.rate_limiter = RateLimiter(max_calls=rate_limit)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FMP-Python-Client/1.0',
            'Accept': 'application/json',
            'Cache-Control': 'no-cache'
        })
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            JSON response or None if failed
        """
        if not self.rate_limiter.can_make_request():
            logw("Rate limit reached, waiting...")
            time.sleep(1)
            if not self.rate_limiter.can_make_request():
                loge("Rate limit still exceeded")
                return None
        
        if params is None:
            params = {}
        
        params['apikey'] = self._api_key
        
        try:
            self.rate_limiter.record_request()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            return data if data else None
            
        except requests.exceptions.Timeout:
            loge(f"Request timeout for URL: {url}")
            return None
        except requests.exceptions.RequestException as e:
            loge(f"Request failed for URL {url}: {e}")
            return None
        except ValueError as e:
            loge(f"Invalid JSON response from {url}: {e}")
            return None
    
    @retry_on_failure(max_retries=3)
    def fetch_stock_screener_results(self, 
                                   exchange_list: str = "nyse,nasdaq,amex",
                                   market_cap_more_than: int = 2000000000,
                                   price_more_than: float = 10.0,
                                   volume_more_than: int = 100000,
                                   beta_lower_than: float = 1.0,
                                   country: str = 'US',
                                   limit: int = 1000,
                                   **kwargs) -> Optional[pd.DataFrame]:
        """
        Fetch stock screener results with improved parameter handling.
        
        Args:
            exchange_list: Comma-separated list of exchanges
            market_cap_more_than: Minimum market cap
            price_more_than: Minimum price
            volume_more_than: Minimum volume
            beta_lower_than: Maximum beta
            country: Country filter
            limit: Maximum results
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with screener results or None
        """
        url = f"{self.base_url}/v3/stock-screener"
        
        params = {
            'exchange': exchange_list,
            'limit': limit,
            'marketCapMoreThan': market_cap_more_than,
            'betaLowerThan': beta_lower_than,
            'volumeMoreThan': volume_more_than,
            'country': country,
            'priceMoreThan': price_more_than,
            'isActivelyTrading': True,
            'isFund': False,
            'isEtf': False,
            **kwargs
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        logd(f"Fetching stock screener: {len(params)} parameters")
        
        data = self._make_request(url, params)
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                logw("Stock screener returned empty results")
                return None
            
            logd(f"Stock screener returned {len(df)} results")
            return df
            
        except Exception as e:
            loge(f"Error processing stock screener data: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def fetch_daily_prices(self, 
                          symbol: str, 
                          start_date_str: str, 
                          end_date_str: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical daily prices for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date_str: Start date (YYYY-MM-DD)
            end_date_str: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data or None
        """
        if not symbol or not symbol.strip():
            loge("Symbol is required for price data")
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/historical-price-full/{symbol}"
        
        params = {
            'from': start_date_str,
            'to': end_date_str,
            'serietype': 'line'
        }
        
        data = self._make_request(url, params)
        if not data or 'historical' not in data:
            return None
        
        historical_data = data['historical']
        if not historical_data:
            logw(f"No historical data for {symbol}")
            return None
        
        try:
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            logd(f"Fetched {len(df)} price records for {symbol}")
            return df
            
        except Exception as e:
            loge(f"Error processing price data for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def fetch_tradable_list(self) -> Optional[pd.DataFrame]:
        """
        Fetch list of tradable securities.
        
        Returns:
            DataFrame with tradable securities or None
        """
        url = f"{self.base_url}/v3/available-traded/list"
        
        data = self._make_request(url)
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            logd(f"Fetched {len(df)} tradable securities")
            return df
        except Exception as e:
            loge(f"Error processing tradable list: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_analyst_ratings(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get analyst ratings for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with analyst ratings or None
        """
        if not symbol or not symbol.strip():
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/grade/{symbol}"
        
        data = self._make_request(url)
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            logd(f"Fetched {len(df)} analyst ratings for {symbol}")
            return df
            
        except Exception as e:
            loge(f"Error processing analyst ratings for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_income_growth(self, symbol: str, period: str = 'annual') -> Optional[pd.DataFrame]:
        """
        Get income growth data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            
        Returns:
            DataFrame with income growth data or None
        """
        if not symbol or not symbol.strip():
            return None
        
        if period not in ['annual', 'quarterly']:
            loge(f"Invalid period: {period}")
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/income-statement-growth/{symbol}"
        
        params = {'period': period}
        data = self._make_request(url, params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            logd(f"Fetched income growth data for {symbol} ({period})")
            return df
            
        except Exception as e:
            loge(f"Error processing income growth for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_financial_ratios(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Get financial ratios for a symbol.
        
        Args:
            symbol: Stock symbol
            period: 'annual' or 'quarterly'
            
        Returns:
            DataFrame with financial ratios or None
        """
        if not symbol or not symbol.strip():
            return None
        
        if period not in ['annual', 'quarterly']:
            loge(f"Invalid period: {period}")
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/ratios/{symbol}"
        
        params = {'period': period}
        data = self._make_request(url, params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            logd(f"Fetched financial ratios for {symbol} ({period})")
            return df
            
        except Exception as e:
            loge(f"Error processing financial ratios for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_social_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get social sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with social sentiment or None
        """
        if not symbol or not symbol.strip():
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v4/historical/social-sentiment"
        
        params = {'symbol': symbol}
        data = self._make_request(url, params)
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            
            logd(f"Fetched social sentiment for {symbol}")
            return df
            
        except Exception as e:
            loge(f"Error processing social sentiment for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_stock_news(self, symbol: str, limit: int = 50) -> Optional[pd.DataFrame]:
        """
        Get recent news for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Number of news articles
            
        Returns:
            DataFrame with news articles or None
        """
        if not symbol or not symbol.strip():
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/stock_news"
        
        params = {
            'tickers': symbol,
            'limit': min(limit, 1000)  # Cap at reasonable limit
        }
        
        data = self._make_request(url, params)
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Clean and process news data
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                df = df.dropna(subset=['publishedDate'])
            
            # Clean text fields
            for col in ['title', 'text']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: clean_string(str(x)) if x else "")
            
            logd(f"Fetched {len(df)} news articles for {symbol}")
            return df
            
        except Exception as e:
            loge(f"Error processing news for {symbol}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def fetch_all_prices(self) -> Optional[pd.DataFrame]:
        """
        Fetch all real-time prices.
        
        Returns:
            DataFrame with all current prices or None
        """
        url = f"{self.base_url}/v3/stock/full/real-time-price"
        
        data = self._make_request(url)
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Add timestamp processing
            if 'lastSaleTime' in df.columns:
                df['date'] = pd.to_datetime(df['lastSaleTime'], unit='ms', errors='coerce')
            
            logd(f"Fetched real-time prices for {len(df)} symbols")
            return df
            
        except Exception as e:
            loge(f"Error processing real-time prices: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def fetch_dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch dividend history for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with dividend data or None
        """
        if not symbol or not symbol.strip():
            return None
        
        symbol = symbol.strip().upper()
        url = f"{self.base_url}/v3/historical-price-full/stock_dividend/{symbol}"
        
        data = self._make_request(url)
        if not data or 'historical' not in data:
            return None
        
        historical_data = data['historical']
        if not historical_data:
            return None
        
        try:
            df = pd.DataFrame(historical_data)
            
            # Process dates
            for date_col in ['paymentDate', 'declarationDate']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            if 'paymentDate' in df.columns:
                df = df.dropna(subset=['paymentDate'])
                df.set_index('paymentDate', inplace=True)
            
            logd(f"Fetched dividend history for {symbol}")
            return df
            
        except Exception as e:
            loge(f"Error processing dividends for {symbol}: {e}")
            return None
    
    def close(self) -> None:
        """Close the HTTP session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Maintain backward compatibility
FmpClient = OptimizedFmpClient