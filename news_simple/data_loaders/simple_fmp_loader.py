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
        
        # Optimized session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        # Rate limiting
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
        """Get stock screener results with proper debugging and fixed filtering."""
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
            "limit": min(limit, 1000)
        }
        
        log_info(f"Requesting stock screener with limit: {limit}")
        data = self._make_request("stock-screener", params)
        
        if not data:
            log_error("No data received from stock screener API")
            return None
        
        if not isinstance(data, list):
            log_error(f"Expected list response, got: {type(data)}")
            return None
        
        log_info(f"Stock screener API returned {len(data)} results")
        
        try:
            df = pd.DataFrame(data)
            if df.empty:
                log_error("Empty DataFrame from stock screener")
                return None
            
            if 'symbol' not in df.columns:
                log_error(f"No 'symbol' column in response. Columns: {df.columns.tolist()}")
                return None
            
            log_info(f"DataFrame created with {len(df)} rows")
            
            # Log some sample symbols before filtering
            sample_symbols = df['symbol'].head(10).tolist()
            log_info(f"Sample symbols before filtering: {sample_symbols}")
            
            # FIXED: Proper boolean operation precedence and more lenient filtering
            initial_count = len(df)
            
            # Remove any null symbols first
            df = df[df['symbol'].notna()]
            log_info(f"After removing null symbols: {len(df)} from {initial_count}")
            
            # Convert to string and remove any empty strings
            df = df[df['symbol'].astype(str).str.strip() != '']
            log_info(f"After removing empty symbols: {len(df)}")
            
            # Apply length filter (be more generous - allow up to 6 characters)
            length_mask = df['symbol'].str.len() <= 6
            df_length = df[length_mask]
            log_info(f"After length filter (<=6 chars): {len(df_length)} from {len(df)}")
            
            # Check if symbols are mostly alphabetic (allow for dots, dashes in some symbols)
            # But still filter out obviously bad symbols
            alpha_mask = df_length['symbol'].str.match(r'^[A-Z][A-Z0-9\-\.]*$')
            df_filtered = df_length[alpha_mask]
            log_info(f"After alpha filter: {len(df_filtered)} from {len(df_length)}")
            
            # Show what symbols passed the filter
            if not df_filtered.empty:
                final_sample = df_filtered['symbol'].head(20).tolist()
                log_info(f"Sample symbols after filtering: {final_sample}")
            
            # If we get too few results, use more lenient filtering
            if len(df_filtered) < 50:
                log_info("Too few symbols after strict filtering, using lenient approach...")
                
                # More lenient: just remove obviously bad symbols
                lenient_mask = (
                    (df['symbol'].str.len() >= 1) & 
                    (df['symbol'].str.len() <= 8) &
                    (~df['symbol'].str.contains(r'[^A-Z0-9\-\.]', na=False))
                )
                df_lenient = df[lenient_mask]
                log_info(f"Lenient filtering: {len(df_lenient)} symbols")
                
                if len(df_lenient) > len(df_filtered):
                    lenient_sample = df_lenient['symbol'].head(20).tolist()
                    log_info(f"Using lenient results. Sample: {lenient_sample}")
                    return df_lenient.reset_index(drop=True)
            
            if df_filtered.empty:
                log_error("All symbols filtered out!")
                return None
            
            return df_filtered.reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing screener data: {e}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Clean and deduplicate symbols efficiently
        unique_symbols = list(dict.fromkeys(
            sym.strip().upper() for sym in symbols if sym.strip()
        ))
        
        if not unique_symbols:
            return None
        
        # Batch processing
        symbol_batch = unique_symbols[:100]
        symbol_str = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_str}")
        
        if not data:
            return None
        
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return None
            
            # Vectorized numeric conversion
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
            
            # Optimize datetime handling
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(
                    df['publishedDate'], 
                    errors='coerce',
                    utc=True
                )
            
            # Create content field efficiently
            if 'title' in df.columns and 'text' in df.columns:
                df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
            
            # Vectorized filtering for essential data
            essential_columns = ['symbol', 'title']
            for col in essential_columns:
                if col in df.columns:
                    df = df[df[col].notna() & (df[col] != '')]
            
            return df
                
        except Exception as e:
            log_error(f"Error processing news data: {e}")
            return None
    
    @lru_cache(maxsize=100)
    def get_historical_data_cached(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data with caching for repeated requests."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        # Handle both dictionary and list response formats
        if not data:
            return None
        
        try:
            # Handle different response formats from FMP API
            historical_data = None
            
            if isinstance(data, dict):
                # Response is a dictionary with 'historical' key
                if 'historical' in data and data['historical']:
                    historical_data = data['historical']
                else:
                    log_debug(f"No historical data found in response for {symbol}")
                    return None
            elif isinstance(data, list):
                # Response is directly a list of historical data
                historical_data = data
            else:
                log_error(f"Unexpected data format for {symbol}: {type(data)}")
                return None
            
            if not historical_data:
                log_debug(f"Empty historical data for {symbol}")
                return None
            
            df = pd.DataFrame(historical_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Vectorized numeric conversion
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid data
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data without caching for non-cached calls."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        # Handle both dictionary and list response formats
        if not data:
            return None
        
        try:
            # Handle different response formats from FMP API
            historical_data = None
            
            if isinstance(data, dict):
                # Response is a dictionary with 'historical' key
                if 'historical' in data and data['historical']:
                    historical_data = data['historical']
                else:
                    log_debug(f"No historical data found in response for {symbol}")
                    return None
            elif isinstance(data, list):
                # Response is directly a list of historical data
                historical_data = data
            else:
                log_error(f"Unexpected data format for {symbol}: {type(data)}")
                return None
            
            if not historical_data:
                log_debug(f"Empty historical data for {symbol}")
                return None
            
            df = pd.DataFrame(historical_data)
            
            # Ensure required columns exist
            if 'date' not in df.columns:
                log_error(f"Missing 'date' column in historical data for {symbol}")
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Vectorized numeric conversion
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid data
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def __del__(self):
        """Cleanup session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()