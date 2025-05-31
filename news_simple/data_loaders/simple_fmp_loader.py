"""
Optimized FMP data loader with improved error handling and type hints
"""
import requests
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import time
from utils.simple_logger import log_info, log_error, log_debug


class SimpleFMPLoader:
    """Optimized FMP data loader with comprehensive error handling."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize FMP loader with API key validation."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0'
        })
        self.last_request_time = 0.0
        self.min_request_interval = 0.1
    
    def _rate_limit(self) -> None:
        """Implement rate limiting with precise timing."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict, List]]:
        """Make API request with comprehensive error handling."""
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle empty responses
            if not data:
                log_debug(f"Empty response from {endpoint}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            log_error(f"Request error for {endpoint}: {e}")
            return None
        except ValueError as e:
            log_error(f"JSON decode error for {endpoint}: {e}")
            return None
        except Exception as e:
            log_error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener results with optimized parameters."""
        if limit <= 0:
            log_error("Limit must be positive")
            return None
        
        params = {
            "marketCapMoreThan": 100_000_000,
            "priceMoreThan": 2,
            "priceLowerThan": 500,
            "volumeMoreThan": 50_000,
            "isActivelyTrading": "true",
            "exchange": "NYSE,NASDAQ",
            "limit": min(limit, 1000)  # API limit
        }
        
        data = self._make_request("stock-screener", params)
        
        if data and isinstance(data, list):
            try:
                df = pd.DataFrame(data)
                if not df.empty and 'symbol' in df.columns:
                    # Remove any invalid symbols
                    df = df[df['symbol'].str.len() <= 5]  # Reasonable symbol length
                    df = df[df['symbol'].str.isalpha()]   # Only alphabetic symbols
                    return df
            except Exception as e:
                log_error(f"Error processing screener data: {e}")
        
        return None
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Remove duplicates and clean symbols
        unique_symbols = list(set(sym.strip().upper() for sym in symbols if sym.strip()))
        
        if not unique_symbols:
            return None
        
        # Limit to API batch size
        symbol_batch = unique_symbols[:100]
        symbol_str = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_str}")
        
        if data:
            try:
                df = pd.DataFrame(data)
                if not df.empty and 'symbol' in df.columns:
                    # Ensure numeric columns are properly typed
                    numeric_columns = ['lastSalePrice', 'bidPrice', 'askPrice', 'volume']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
            except Exception as e:
                log_error(f"Error processing price data: {e}")
        
        return None
    
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Get latest news with improved datetime handling."""
        try:
            data = self._make_request("../v4/stock-news-sentiments-rss-feed", {"page": 0})
            
            if data and isinstance(data, list):
                df = pd.DataFrame(data)
                
                if not df.empty:
                    # Improved datetime handling with timezone awareness
                    if 'publishedDate' in df.columns:
                        df['publishedDate'] = pd.to_datetime(
                            df['publishedDate'], 
                            errors='coerce',
                            utc=True  # Ensure timezone awareness
                        )
                    
                    # Create content field safely
                    if 'title' in df.columns and 'text' in df.columns:
                        df['content'] = (
                            df['title'].astype(str) + " " + 
                            df['text'].astype(str)
                        )
                    
                    # Filter out rows with missing essential data
                    essential_columns = ['symbol', 'title']
                    for col in essential_columns:
                        if col in df.columns:
                            df = df[df[col].notna() & (df[col] != '')]
                    
                    return df
                    
        except Exception as e:
            log_error(f"Error processing news data: {e}")
        
        return None