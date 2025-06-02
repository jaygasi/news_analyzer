"""
Optimized FMP data loader with improved error handling and type hints
"""
import requests
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import time
from functools import lru_cache
from utils.simple_logger import log_info, log_error, log_debug


class SimpleFMPLoader:
    """Optimized FMP data loader with comprehensive error handling."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize FMP loader with API key validation."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self._setup_session()
        self._setup_rate_limiting()
    
    def _setup_session(self) -> None:
        """Setup optimized session with connection pooling."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
    
    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting configuration."""
        self.last_request_time = 0.0
        self.min_request_interval = 0.1
    
    def _rate_limit(self) -> None:
        """Implement rate limiting with precise timing."""
        elapsed = time.time() - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict, List]]:
        """Make API request with comprehensive error handling."""
        self._rate_limit()
        
        params = params or {}
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
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
        """Get stock screener results with proper debugging and filtering."""
        if limit <= 0:
            log_error("Limit must be positive")
            return None
        
        params = self._build_screener_params(limit)
        
        log_info(f"Requesting stock screener with limit: {limit}")
        # Renaming variable for clarity before assertion
        api_response_data = self._make_request("stock-screener", params)
        
        if not self._validate_screener_response(api_response_data):
            return None
        
        # After _validate_screener_response, api_response_data is known to be a List.
        # Add an assertion to help Pylance (and for runtime safety).
        assert isinstance(api_response_data, list), "Screener data should be a list after validation"
        
        return self._process_screener_data(api_response_data)
    
    def _build_screener_params(self, limit: int) -> Dict[str, Any]:
        """Build screener request parameters."""
        return {
            "marketCapMoreThan": 100_000_000,
            "priceMoreThan": 2,
            "priceLowerThan": 500,
            "volumeMoreThan": 50_000,
            "isActivelyTrading": "true",
            "exchange": "NYSE,NASDAQ",
            "limit": min(limit, 1000)
        }
    
    def _validate_screener_response(self, data: Any) -> bool:
        """Validate screener API response."""
        if not data:
            log_error("No data received from stock screener API")
            return False
        
        if not isinstance(data, list):
            log_error(f"Expected list response, got: {type(data)}")
            return False
        
        log_info(f"Stock screener API returned {len(data)} results")
        return True
    
    def _process_screener_data(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        """Process screener data with filtering and validation."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                log_error("Empty DataFrame from stock screener")
                return None
            
            if 'symbol' not in df.columns:
                log_error(f"No 'symbol' column in response. Columns: {df.columns.tolist()}")
                return None
            
            log_info(f"DataFrame created with {len(df)} rows")
            
            # Apply symbol filtering
            filtered_df = self._filter_symbols(df)
            
            if filtered_df.empty:
                log_error("All symbols filtered out!")
                return None
            
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing screener data: {e}")
            return None
    
    def _filter_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter symbols with optimized logic."""
        initial_count = len(df)
        
        # Remove null/empty symbols
        df = df[df['symbol'].notna()]
        df = df[df['symbol'].astype(str).str.strip() != '']
        log_info(f"After removing null/empty symbols: {len(df)} from {initial_count}")
        
        # Apply length filter
        length_mask = df['symbol'].str.len() <= 6
        df_length = df[length_mask]
        log_info(f"After length filter (<=6 chars): {len(df_length)} from {len(df)}")
        
        # Apply alphabetic filter
        alpha_mask = df_length['symbol'].str.match(r'^[A-Z][A-Z0-9\-\.]*$')
        df_filtered = df_length[alpha_mask]
        log_info(f"After alpha filter: {len(df_filtered)} from {len(df_length)}")
        
        # Use lenient filtering if too few results
        if len(df_filtered) < 50:
            return self._apply_lenient_filter(df)
        
        return df_filtered
    
    def _apply_lenient_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply more lenient filtering when strict filtering yields too few results."""
        log_info("Applying lenient filtering...")
        
        lenient_mask = (
            (df['symbol'].str.len() >= 1) & 
            (df['symbol'].str.len() <= 8) &
            (~df['symbol'].str.contains(r'[^A-Z0-9\-\.]', na=False))
        )
        
        df_lenient = df[lenient_mask]
        log_info(f"Lenient filtering: {len(df_lenient)} symbols")
        
        return df_lenient
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Clean and deduplicate symbols
        unique_symbols = list(dict.fromkeys(
            sym.strip().upper() for sym in symbols if sym.strip()
        ))
        
        if not unique_symbols:
            return None
        
        # Process in batches
        symbol_batch = unique_symbols[:100]
        symbol_str = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_str}")
        
        if not data:
            return None
        
        return self._process_price_data(data)
    
    def _process_price_data(self, data: Union[Dict, List]) -> Optional[pd.DataFrame]:
        """Process price data with error handling."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return None
            
            # Convert numeric columns
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
            
            if not data or not isinstance(data, list):
                return None
            
            df = pd.DataFrame(data)
            
            if df.empty:
                return None
            
            return self._process_news_data(df)
                
        except Exception as e:
            log_error(f"Error processing news data: {e}")
            return None
    
    def _process_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process news data with optimized handling."""
        # Handle datetime
        if 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(
                df['publishedDate'], 
                errors='coerce',
                utc=True
            )
        
        # Create content field
        if 'title' in df.columns and 'text' in df.columns:
            df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
        
        # Filter essential data
        essential_columns = ['symbol', 'title']
        for col in essential_columns:
            if col in df.columns:
                df = df[df[col].notna() & (df[col] != '')]
        
        return df
    
    @lru_cache(maxsize=100)
    def get_historical_data_cached(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data with caching for repeated requests."""
        return self.get_historical_data(symbol, days)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data without caching."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        if not data:
            return None
        
        return self._process_historical_data(data, symbol)
    
    def _process_historical_data(self, data: Union[Dict, List], symbol: str) -> Optional[pd.DataFrame]:
        """Process historical data with error handling."""
        try:
            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                log_error(f"Unexpected data format for {symbol}: {type(data)}")
                return None
            
            if not historical_data:
                log_debug(f"Empty historical data for {symbol}")
                return None
            
            df = pd.DataFrame(historical_data)
            
            if 'date' not in df.columns:
                log_error(f"Missing 'date' column in historical data for {symbol}")
                return None
            
            # Process data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def __del__(self):
        """Cleanup session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()