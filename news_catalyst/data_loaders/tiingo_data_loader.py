"""
Optimized Tiingo data loader with improved error handling and performance
"""
import os
import requests
import pandas as pd
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from functools import lru_cache
import time

from utils.string_utils import clean_string, join_items
from utils.log_utils import logd, logw, loge


class TiingoIntradayInterval(Enum):
    """Enum for different Tiingo intraday resample intervals."""
    MIN_1 = '1min'
    MIN_5 = '5min'
    MIN_15 = '15min'
    MIN_30 = '30min'
    HOUR_1 = '1hour'
    HOUR_4 = '4hour'
    DAY_1 = '1day'


class TiingoDailyInterval(Enum):
    """Enum for different daily Tiingo resample intervals."""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    ANNUALLY = 'annually'


@lru_cache(maxsize=1000)
def create_md5_hash(my_string: str) -> str:
    """
    Create MD5 hash with caching for repeated strings.
    
    Args:
        my_string: String to hash
        
    Returns:
        MD5 hash string
    """
    if not my_string:
        return ""
    
    return hashlib.md5(my_string.encode('utf-8')).hexdigest()


class TiingoRateLimiter:
    """Rate limiter for Tiingo API requests."""
    
    def __init__(self, max_calls: int = 500, time_window: int = 3600):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limits."""
        now = time.time()
        # Remove old calls
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        return len(self.calls) < self.max_calls
    
    def record_request(self) -> None:
        """Record a request."""
        if len(self.calls) >= self.max_calls:
            # Remove oldest call
            self.calls.pop(0)
        self.calls.append(time.time())


class OptimizedTiingoDataLoader:
    """
    Enhanced TiingoDataLoader with improved performance and error handling.
    """

    def __init__(self, api_key: str, rate_limit: int = 500):
        if not api_key or not api_key.strip():
            raise ValueError("Tiingo API key is required")
        
        self.api_key = api_key.strip()
        self.base_url = "https://api.tiingo.com"
        self.rate_limiter = TiingoRateLimiter(max_calls=rate_limit)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Tiingo-Python-Client/1.0'
        })
    
    def _make_request(self, url: str, headers: Optional[Dict] = None, 
                     params: Optional[Dict] = None, timeout: int = 30) -> Optional[Dict]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            headers: Additional headers
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            JSON response or None if failed
        """
        if not self.rate_limiter.can_make_request():
            logw("Tiingo rate limit reached")
            return None
        
        request_headers = {'Accept': 'application/json'}
        if headers:
            request_headers.update(headers)
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            self.rate_limiter.record_request()
            response = self.session.get(url, headers=request_headers, 
                                      params=params, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            return data if data else None
            
        except requests.exceptions.Timeout:
            loge(f"Tiingo request timeout: {url}")
            return None
        except requests.exceptions.RequestException as e:
            loge(f"Tiingo request failed: {e}")
            return None
        except ValueError as e:
            loge(f"Invalid JSON from Tiingo: {e}")
            return None
    
    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate and clean symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Cleaned symbol
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        cleaned = symbol.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty after cleaning")
        
        # Basic symbol validation
        if len(cleaned) > 10 or not cleaned.replace('.', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return cleaned
    
    def _validate_date_range(self, start_date: str, end_date: str) -> None:
        """
        Validate date range.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Raises:
            ValueError: If dates are invalid
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if start > end:
                raise ValueError("Start date must be before end date")
            
            # Don't allow future dates
            if end > pd.Timestamp.now():
                raise ValueError("End date cannot be in the future")
                
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def _process_price_data(self, data: List[Dict], cache_data: bool = False, 
                          cache_dir: str = "cache", file_name: str = None) -> Optional[pd.DataFrame]:
        """
        Process price data into DataFrame.
        
        Args:
            data: Raw price data from API
            cache_data: Whether to cache the data
            cache_dir: Cache directory
            file_name: Cache file name
            
        Returns:
            Processed DataFrame or None
        """
        if not data:
            return None
        
        try:
            # Convert to DataFrame efficiently
            df = pd.DataFrame(data)
            
            if df.empty:
                return None
            
            # Process date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if df.empty:
                return None
            
            # Set index and sort
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            # Cache if requested
            if cache_data and file_name:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    cache_path = os.path.join(cache_dir, file_name)
                    df.to_csv(cache_path)
                    logd(f"Cached price data to {cache_path}")
                except Exception as e:
                    logw(f"Failed to cache data: {e}")
            
            return df
            
        except Exception as e:
            loge(f"Error processing price data: {e}")
            return None
    
    def fetch_intraday_prices(self, symbol: str, start_date_str: str, end_date_str: str,
                            interval: TiingoIntradayInterval, cache_data: bool = False,
                            cache_dir: str = "cache") -> Optional[pd.DataFrame]:
        """
        Fetch intraday stock prices with improved error handling.
        
        Args:
            symbol: Stock symbol
            start_date_str: Start date in 'YYYY-MM-DD' format
            end_date_str: End date in 'YYYY-MM-DD' format
            interval: Data interval
            cache_data: Whether to cache the data
            cache_dir: Directory to save cached data
            
        Returns:
            DataFrame with stock prices or None
        """
        try:
            symbol = self._validate_symbol(symbol)
            self._validate_date_range(start_date_str, end_date_str)
            
            file_name = f"{symbol}_{interval.value}_{start_date_str}_{end_date_str}.csv"
            cache_path = os.path.join(cache_dir, file_name)
            
            # Check cache first
            if cache_data and os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['date'])
                    df.set_index('date', inplace=True)
                    logd(f"Loaded cached data for {symbol}")
                    return df
                except Exception as e:
                    logw(f"Failed to load cached data: {e}")
            
            # Fetch from API
            url = f"{self.base_url}/iex/{symbol}/prices"
            params = {
                'startDate': start_date_str,
                'endDate': end_date_str,
                'resampleFreq': interval.value,
                'columns': 'date,open,high,low,close,volume'
            }
            
            data = self._make_request(url, params=params)
            if not data:
                return None
            
            return self._process_price_data(data, cache_data, cache_dir, file_name)
            
        except ValueError as e:
            loge(f"Validation error in fetch_intraday_prices: {e}")
            return None
        except Exception as e:
            loge(f"Error fetching intraday prices for {symbol}: {e}")
            return None
    
    def fetch_multiple_intraday_prices(self, symbol_list: List[str], start_date_str: str,
                                     end_date_str: str, interval: TiingoIntradayInterval,
                                     cache_data: bool = False, cache_dir: str = "cache") -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday prices for multiple symbols.
        
        Args:
            symbol_list: List of stock symbols
            start_date_str: Start date
            end_date_str: End date
            interval: Data interval
            cache_data: Whether to cache data
            cache_dir: Cache directory
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not symbol_list:
            return {}
        
        results = {}
        
        for symbol in symbol_list:
            try:
                logd(f"Fetching intraday prices for {symbol}")
                df = self.fetch_intraday_prices(
                    symbol, start_date_str, end_date_str, interval, cache_data, cache_dir
                )
                
                if df is not None:
                    results[symbol] = df
                else:
                    logw(f"No data returned for {symbol}")
                    
            except Exception as e:
                loge(f"Error fetching data for {symbol}: {e}")
                continue
        
        logd(f"Fetched intraday data for {len(results)}/{len(symbol_list)} symbols")
        return results
    
    def fetch_end_of_day_prices(self, symbol: str, start_date: str, end_date: str,
                              interval: TiingoDailyInterval = TiingoDailyInterval.DAILY,
                              cache_data: bool = False, cache_dir: str = "cache") -> Optional[pd.DataFrame]:
        """
        Fetch daily stock prices with improved error handling.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            cache_data: Whether to cache data
            cache_dir: Cache directory
            
        Returns:
            DataFrame with daily prices or None
        """
        try:
            symbol = self._validate_symbol(symbol)
            self._validate_date_range(start_date, end_date)
            
            file_name = f"{symbol}_{interval.value}_{start_date}_{end_date}.csv"
            cache_path = os.path.join(cache_dir, file_name)
            
            # Check cache first
            if cache_data and os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path, parse_dates=['date'])
                    df.set_index('date', inplace=True)
                    return df
                except Exception as e:
                    logw(f"Failed to load cached data: {e}")
            
            # Fetch from API
            url = f"{self.base_url}/tiingo/daily/{symbol}/prices"
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'resampleFreq': interval.value,
                'columns': 'date,open,high,low,close,volume'
            }
            
            data = self._make_request(url, params=params)
            if not data:
                return None
            
            # Process data
            df = pd.DataFrame(data)
            if 'adjClose' in df.columns:
                df.rename(columns={'adjClose': 'adj_close'}, inplace=True)
            
            return self._process_price_data(data, cache_data, cache_dir, file_name)
            
        except ValueError as e:
            loge(f"Validation error in fetch_end_of_day_prices: {e}")
            return None
        except Exception as e:
            loge(f"Error fetching EOD prices for {symbol}: {e}")
            return None
    
    def fetch_multiple_end_of_day_prices(self, symbol_list: List[str], start_date_str: str,
                                       end_date_str: str, interval: TiingoDailyInterval = TiingoDailyInterval.DAILY,
                                       cache_data: bool = False, cache_dir: str = "cache") -> Dict[str, pd.DataFrame]:
        """
        Fetch daily prices for multiple symbols.
        
        Args:
            symbol_list: List of stock symbols
            start_date_str: Start date
            end_date_str: End date
            interval: Data interval
            cache_data: Whether to cache data
            cache_dir: Cache directory
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if not symbol_list:
            return {}
        
        results = {}
        
        for symbol in symbol_list:
            try:
                logd(f"Fetching EOD prices for {symbol}")
                df = self.fetch_end_of_day_prices(
                    symbol, start_date_str, end_date_str, interval, cache_data, cache_dir
                )
                
                if df is not None:
                    results[symbol] = df
                else:
                    logw(f"No EOD data returned for {symbol}")
                    
            except Exception as e:
                loge(f"Error fetching EOD data for {symbol}: {e}")
                continue
        
        logd(f"Fetched EOD data for {len(results)}/{len(symbol_list)} symbols")
        return results
    
    def fetch_latest_news_articles(self, news_article_limit: int = 50,
                                 cache_data: bool = False, cache_dir: str = 'cache') -> Optional[pd.DataFrame]:
        """
        Fetch latest news articles with improved processing.
        
        Args:
            news_article_limit: Maximum number of articles
            cache_data: Whether to cache data
            cache_dir: Cache directory
            
        Returns:
            DataFrame with news articles or None
        """
        start_date = datetime.today()
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date = datetime.today() + timedelta(days=1)
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        cache_file = f"news_{start_date_str}_{end_date_str}.csv"
        cache_path = os.path.join(cache_dir, cache_file)
        
        # Check cache first
        if cache_data and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                # Ensure string columns
                for col in ['title', 'description']:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                return df
            except Exception as e:
                logw(f"Failed to load cached news: {e}")
        
        try:
            url = f"{self.base_url}/tiingo/news"
            params = {
                'startDate': start_date_str,
                'endDate': end_date_str,
                'limit': min(news_article_limit, 1000)  # Cap at reasonable limit
            }
            
            data = self._make_request(url, params=params)
            if not data:
                return None
            
            # Process news data
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Clean and process text fields
            if 'title' in df.columns:
                df['title'] = df['title'].apply(clean_string)
            
            if 'description' in df.columns:
                df['description'] = df['description'].apply(clean_string)
                df.rename(columns={'description': 'text'}, inplace=True)
            
            # Process other columns
            if 'tickers' in df.columns:
                df.rename(columns={'tickers': 'symbols'}, inplace=True)
                df['symbol'] = df['symbols'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
                )
            
            # Add content column
            if 'title' in df.columns and 'text' in df.columns:
                df['content'] = df['title'] + ' ' + df['text']
            
            # Process dates
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                df = df.dropna(subset=['publishedDate'])
            
            # Remove duplicates
            if 'title' in df.columns:
                df = df.drop_duplicates(subset=['title'])
            
            # Cache results
            if cache_data and not df.empty:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                except Exception as e:
                    logw(f"Failed to cache news data: {e}")
            
            logd(f"Fetched {len(df)} news articles")
            return df
            
        except Exception as e:
            loge(f"Error fetching latest news: {e}")
            return None
    
    def fetch_news_article_by_symbol(self, symbol: str, start_date_str: str, end_date_str: str,
                                   news_article_limit: int = 50, cache_data: bool = False,
                                   cache_dir: str = 'cache') -> Optional[pd.DataFrame]:
        """
        Fetch news articles for specific symbol.
        
        Args:
            symbol: Stock symbol
            start_date_str: Start date
            end_date_str: End date
            news_article_limit: Maximum articles
            cache_data: Whether to cache
            cache_dir: Cache directory
            
        Returns:
            DataFrame with news articles or None
        """
        try:
            symbol = self._validate_symbol(symbol)
            self._validate_date_range(start_date_str, end_date_str)
            
            cache_file = f"{symbol}_{start_date_str}_{end_date_str}_news.csv"
            cache_path = os.path.join(cache_dir, cache_file)
            
            # Check cache
            if cache_data and os.path.exists(cache_path):
                try:
                    df = pd.read_csv(cache_path)
                    for col in ['title', 'description']:
                        if col in df.columns:
                            df[col] = df[col].astype(str)
                    return df
                except Exception as e:
                    logw(f"Failed to load cached news for {symbol}: {e}")
            
            # Fetch from API
            url = f"{self.base_url}/tiingo/news"
            headers = {'Authorization': f'Token {self.api_key}'}
            params = {
                'startDate': start_date_str,
                'endDate': end_date_str,
                'limit': min(news_article_limit, 1000),
                'tickers': symbol
            }
            
            data = self._make_request(url, headers=headers, params=params)
            if not data:
                return None
            
            # Process similar to latest news
            df = pd.DataFrame(data)
            if df.empty:
                return None
            
            # Clean text fields
            if 'title' in df.columns:
                df['title'] = df['title'].apply(clean_string)
            
            if 'description' in df.columns:
                df['description'] = df['description'].apply(clean_string)
            
            # Process tickers
            if 'tickers' in df.columns:
                df.rename(columns={'tickers': 'symbols'}, inplace=True)
                df['symbols'] = df['symbols'].apply(join_items)
            
            # Remove duplicates and process dates
            if 'title' in df.columns:
                df = df.drop_duplicates(subset=['title'])
            
            if 'id' in df.columns:
                df['id'] = df['id'].astype(str)
            
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
                df = df.dropna(subset=['publishedDate'])
            
            # Cache results
            if cache_data and not df.empty:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    df.to_csv(cache_path, index=False)
                except Exception as e:
                    logw(f"Failed to cache news for {symbol}: {e}")
            
            logd(f"Fetched {len(df)} news articles for {symbol}")
            return df
            
        except ValueError as e:
            loge(f"Validation error for {symbol}: {e}")
            return None
        except Exception as e:
            loge(f"Error fetching news for {symbol}: {e}")
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
TiingoDataLoader = OptimizedTiingoDataLoader