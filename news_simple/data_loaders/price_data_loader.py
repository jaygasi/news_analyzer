"""
Price data loader for FMP API - handles real-time prices, historical data, and screener
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug
from .base_fmp_loader import BaseFMPLoader


class PriceDataLoader(BaseFMPLoader):
    """Specialized loader for price and market data."""
    
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener with enhanced parameters for news-heavy stocks."""
        if limit <= 0:
            raise ValueError("Limit must be positive")
        
        # Enhanced screener parameters for better news coverage
        params = {
            "marketCapMoreThan": CONFIG.min_market_cap,
            "priceMoreThan": CONFIG.min_price,
            "priceLowerThan": CONFIG.max_price,
            "volumeMoreThan": CONFIG.min_volume,
            "isActivelyTrading": "true",
            "exchange": "NASDAQ,NYSE",
            "sector": "Technology,Healthcare,Financial Services,Consumer Cyclical,Industrials,Communication Services,Consumer Defensive,Energy,Real Estate",
            "limit": min(limit, 1000)
        }
        
        log_info(f"Requesting stock screener with enhanced parameters, limit: {limit}")
        data = self._make_request("stock-screener", params)
        
        if data is None:
            log_error("Failed to retrieve stock screener data")
            return None
        
        if not self._validate_response(data, "stock screener"):
            return None
        
        if not isinstance(data, list):
            log_error(f"Stock screener API returned data of type {type(data)} after validation, but a list was expected for processing.")
            return None
        
        return self._process_screener_data(data)
    
    def _process_screener_data(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        """Process screener data with enhanced filtering."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                log_error("Invalid screener data structure")
                return None
            
            log_info(f"Processing {len(df)} screener results")
            filtered_df = self._filter_symbols_for_news(df)
            
            return filtered_df.reset_index(drop=True) if not filtered_df.empty else None
            
        except Exception as e:
            log_error(f"Error processing screener data: {e}")
            return None
    
    def _filter_symbols_for_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter symbols optimized for news coverage."""
        initial_count = len(df)
        
        # Basic filters
        df = df[df['symbol'].notna() & (df['symbol'].str.strip() != '')]
        
        # News-optimized filters
        filters = (
            (df['symbol'].str.len() <= 5) &
            (df['symbol'].str.match(r'^[A-Z]+$')) &
            (~df['symbol'].str.contains(r'[.]')) &
            (df['symbol'].str.len() >= 1)
        )
        
        df_filtered = df[filters]
        
        # If too restrictive, apply lenient filtering
        if len(df_filtered) < 100:
            log_info("Applying lenient filtering for better coverage")
            lenient_filters = (
                (df['symbol'].str.len() <= 6) &
                (df['symbol'].str.match(r'^[A-Z][A-Z0-9]*$'))
            )
            df_filtered = df[lenient_filters]
        
        log_info(f"Symbol filtering: {initial_count} -> {len(df_filtered)}")
        return df_filtered
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Clean and deduplicate
        clean_symbols = list(dict.fromkeys(
            symbol.strip().upper() for symbol in symbols if symbol.strip()
        ))
        
        if not clean_symbols:
            return None
        
        # Process in optimized batches
        symbol_batch = clean_symbols[:100]
        symbol_string = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_string}")
        return self._process_price_data(data) if data else None
    
    def _process_price_data(self, data: Union[Dict, List]) -> Optional[pd.DataFrame]:
        """Process price data with type conversion."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return None
            
            # Optimize numeric conversion
            numeric_cols = ['lastSalePrice', 'bidPrice', 'askPrice', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            log_error(f"Error processing price data: {e}")
            return None
    
    @lru_cache(maxsize=100)
    def get_historical_data_cached(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Cached historical data retrieval."""
        return self.get_historical_data(symbol, days)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        return self._process_historical_data(data, symbol) if data else None
    
    def _process_historical_data(self, data: Union[Dict, List], symbol: str) -> Optional[pd.DataFrame]:
        """Process historical data with validation."""
        try:
            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                log_error(f"Unexpected historical data format for {symbol}")
                return None
            
            if not historical_data:
                return None
            
            df = pd.DataFrame(historical_data)
            
            if 'date' not in df.columns:
                log_error(f"Missing date column for {symbol}")
                return None
            
            # Process and validate data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def get_intraday_data(self, symbol: str, interval: str = '5min') -> Optional[pd.DataFrame]:
        """Get intraday price data."""
        try:
            data = self._make_request(f"historical-chart/{interval}/{symbol}")
            
            if data and isinstance(data, list):
                df = pd.DataFrame(data)
                
                # Convert datetime and numeric columns
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.dropna()
            
            return None
            
        except Exception as e:
            log_debug(f"Error getting intraday data for {symbol}: {e}")
            return None
    
    def get_market_hours(self) -> Optional[Dict[str, Any]]:
        """Get market hours information."""
        try:
            data = self._make_request("market-hours")
            if isinstance(data, dict): 
                # Return only the markets information
                return data
            if data is not None:
                log_debug(f"Market hours API returned an unexpected data type: {type(data)}. Expected dict or None.")
            return None
        
        except Exception as e:
            log_debug(f"Error getting market hours: {e}")
            return None
    
    def get_sector_performance(self) -> Optional[List[Dict[str, Any]]]:
        """Get sector performance data."""
        try:
            data = self._make_request("sectors-performance")
            
            if data and isinstance(data, list):
                return data
            
            return None
            
        except Exception as e:
            log_debug(f"Error getting sector performance: {e}")
            return None