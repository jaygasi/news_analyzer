"""
Optimized universe selector with improved caching and error handling
"""
import numpy as np
import pandas as pd
from threading import Lock
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from config import *
from utils.log_utils import logi, logw, loge, logd
from data_loaders.fmp_data_loader import FmpDataLoader
from utils.file_utils import store_csv


class OptimizedUniverseSelector:
    """Enhanced universe selector with intelligent caching and validation"""
    
    def __init__(self, fmp_api_key: Optional[str] = None) -> None:
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_running = False
            self.fmp_data_loader = FmpDataLoader(fmp_api_key) if fmp_api_key else None
            self.symbol_list: List[str] = []
            self.stock_info_df: Optional[pd.DataFrame] = None
            self.lock = Lock()
            
            # Caching and performance
            self.last_selection_time: Optional[datetime] = None
            self.cache_duration = timedelta(hours=4)  # Cache for 4 hours
            self.selection_in_progress = False
            
            # Performance tracking
            self.total_selections = 0
            self.successful_selections = 0
            self.last_symbol_count = 0

    def _should_refresh_universe(self) -> bool:
        """Determine if universe selection should be refreshed"""
        if self.last_selection_time is None:
            return True
        
        time_since_last = datetime.now() - self.last_selection_time
        return time_since_last > self.cache_duration

    def _validate_stock_data(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean stock screening data"""
        if stock_df is None or stock_df.empty:
            return pd.DataFrame()
        
        # Remove invalid symbols (containing dots for foreign exchanges)
        initial_count = len(stock_df)
        stock_df = stock_df[~stock_df['symbol'].str.contains(r'\.\w{1,4}$', regex=True, na=False)]
        
        # Remove stocks with missing critical data
        required_columns = ['symbol', 'price', 'volume', 'marketCap']
        missing_columns = [col for col in required_columns if col not in stock_df.columns]
        
        if missing_columns:
            logw(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Filter out stocks with invalid data
        stock_df = stock_df.dropna(subset=required_columns)
        stock_df = stock_df[stock_df['price'] > 0]
        stock_df = stock_df[stock_df['volume'] > 0]
        stock_df = stock_df[stock_df['marketCap'] > 0]
        
        cleaned_count = len(stock_df)
        if cleaned_count < initial_count:
            logd(f"Filtered stocks: {initial_count} -> {cleaned_count}")
        
        return stock_df

    def _apply_bid_ask_spread_filter(self, symbol_list: List[str]) -> List[str]:
        """Apply bid-ask spread filtering with error handling"""
        if not symbol_list:
            return []
        
        try:
            # Fetch real-time prices for spread calculation
            prices_df = self.fmp_data_loader.fetch_realtime_prices()
            
            if prices_df is None or prices_df.empty:
                logw("No real-time price data available for spread filtering")
                return symbol_list
            
            # Filter to symbols in our screening list
            prices_df = prices_df[prices_df['symbol'].isin(symbol_list)]
            
            if prices_df.empty:
                logw("No price data found for screened symbols")
                return symbol_list
            
            # Calculate bid-ask spread percentage safely
            def calculate_spread_safe(row: pd.Series) -> float:
                try:
                    ask_price = float(row.get('askPrice', 0))
                    bid_price = float(row.get('bidPrice', 0))
                    
                    if bid_price <= 0 or ask_price <= 0:
                        return float('inf')  # Mark as invalid
                    
                    return (ask_price - bid_price) / bid_price
                except (ValueError, ZeroDivisionError):
                    return float('inf')
            
            prices_df['spread_percentage'] = prices_df.apply(calculate_spread_safe, axis=1)
            
            # Filter stocks with acceptable spread
            filtered_df = prices_df[prices_df['spread_percentage'] <= MAX_BID_ASK_SPREAD]
            filtered_symbols = filtered_df['symbol'].unique().tolist()
            
            spread_filtered_count = len(symbol_list) - len(filtered_symbols)
            if spread_filtered_count > 0:
                logd(f"Filtered {spread_filtered_count} symbols due to wide bid-ask spreads")
            
            return filtered_symbols
            
        except Exception as e:
            loge(f"Error applying bid-ask spread filter: {e}")
            return symbol_list  # Return original list on error

    def _get_screener_parameters(self) -> Dict[str, Any]:
        """Get optimized stock screener parameters"""
        return {
            'exchange_list': EXCHANGE_LIST,
            'price_more_than': PRICE_MORE_THAN,
            'price_lower_than': PRICE_LESS_THAN,
            'volume_more_than': VOLUME_MORE_THAN,
            'market_cap_lower_than': MARKET_CAP_LOWER_THAN,
            'is_etf': False,
            'is_fund': False,
            'is_actively_trading': True,
            'limit': STOCK_SCREENER_LIMIT
        }

    def perform_selection(self) -> bool:
        """Perform optimized universe selection with caching"""
        with self.lock:
            # Check if refresh is needed
            if not self._should_refresh_universe() and self.symbol_list:
                logd(f"Using cached universe: {len(self.symbol_list)} symbols")
                return True
            
            # Prevent concurrent selections
            if self.selection_in_progress:
                logd("Universe selection already in progress")
                return False
            
            self.selection_in_progress = True
        
        try:
            self.total_selections += 1
            logi("🔍 Running optimized universe selection...")
            
            if not self.fmp_data_loader:
                logw("FMP data loader not available - using empty universe")
                return False
            
            # Get screener parameters
            params = self._get_screener_parameters()
            
            # Fetch stock screening data
            stock_list_df = self.fmp_data_loader.fetch_stock_screener_info(**params)
            
            if stock_list_df is None or stock_list_df.empty:
                logw("Stock screener returned no data")
                return False
            
            # Validate and clean data
            stock_list_df = self._validate_stock_data(stock_list_df)
            
            if stock_list_df.empty:
                logw("No valid stocks after filtering")
                return False
            
            # Extract initial symbol list
            symbol_list = stock_list_df['symbol'].unique().tolist()
            
            # Apply bid-ask spread filtering
            filtered_symbols = self._apply_bid_ask_spread_filter(symbol_list)
            
            if not filtered_symbols:
                logw("No symbols passed bid-ask spread filter")
                return False
            
            # Update stock info dataframe to match filtered symbols
            stock_list_df = stock_list_df[stock_list_df['symbol'].isin(filtered_symbols)]
            
            # Store results
            try:
                store_csv(RESULTS_DIR, 'stock_list_df.csv', stock_list_df)
            except Exception as e:
                logw(f"Could not save stock list: {e}")
            
            # Update instance variables atomically
            with self.lock:
                self.symbol_list = filtered_symbols
                self.stock_info_df = stock_list_df
                self.last_selection_time = datetime.now()
                self.last_symbol_count = len(filtered_symbols)
                self.successful_selections += 1
            
            logi(f"✅ Universe selection completed: {len(filtered_symbols)} symbols selected")
            return True
            
        except Exception as e:
            loge(f"Error in universe selection: {e}")
            return False
        
        finally:
            with self.lock:
                self.selection_in_progress = False

    def get_symbol_list(self) -> List[str]:
        """Get symbol list with automatic refresh if needed"""
        with self.lock:
            # Attempt refresh if cache is stale and we have a data loader
            if (self._should_refresh_universe() and 
                self.fmp_data_loader and 
                not self.selection_in_progress):
                
                # Trigger background refresh (non-blocking)
                try:
                    self.perform_selection()
                except Exception as e:
                    logw(f"Background universe refresh failed: {e}")
            
            # Return current list (which may be cached or newly refreshed)
            if isinstance(self.symbol_list, np.ndarray):
                return self.symbol_list.tolist()
            elif isinstance(self.symbol_list, (list, tuple)):
                return list(self.symbol_list)
            else:
                return []

    def get_stock_info_by_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get stock info by symbol with validation"""
        with self.lock:
            if (self.stock_info_df is not None and 
                not self.stock_info_df.empty):
                
                try:
                    symbol_info_df = self.stock_info_df[self.stock_info_df['symbol'] == symbol]
                    return symbol_info_df if not symbol_info_df.empty else None
                except Exception as e:
                    logw(f"Error retrieving stock info for {symbol}: {e}")
                    return None
            return None

    def get_universe_statistics(self) -> Dict[str, Any]:
        """Get comprehensive universe statistics"""
        with self.lock:
            stats = {
                'total_symbols': len(self.symbol_list),
                'total_selections_performed': self.total_selections,
                'successful_selections': self.successful_selections,
                'last_selection_time': self.last_selection_time.isoformat() if self.last_selection_time else None,
                'cache_expires_at': (self.last_selection_time + self.cache_duration).isoformat() if self.last_selection_time else None,
                'selection_in_progress': self.selection_in_progress,
                'success_rate': self.successful_selections / max(self.total_selections, 1),
                'data_loader_available': self.fmp_data_loader is not None
            }
            
            # Add sector/industry breakdown if data is available
            if self.stock_info_df is not None and not self.stock_info_df.empty:
                try:
                    if 'sector' in self.stock_info_df.columns:
                        sector_counts = self.stock_info_df['sector'].value_counts().to_dict()
                        stats['sector_breakdown'] = sector_counts
                    
                    if 'industry' in self.stock_info_df.columns:
                        industry_counts = self.stock_info_df['industry'].value_counts().head(10).to_dict()
                        stats['top_industries'] = industry_counts
                        
                except Exception as e:
                    logd(f"Error calculating sector/industry breakdown: {e}")
            
            return stats

    def force_refresh(self) -> bool:
        """Force immediate universe refresh"""
        with self.lock:
            self.last_selection_time = None  # Clear cache
        
        return self.perform_selection()

    def stop(self) -> None:
        """Stop universe selector with final statistics"""
        self.is_running = False
        stats = self.get_universe_statistics()
        logi(f"🔍 Universe selector stopped - Final stats: {stats['total_symbols']} symbols, "
             f"{stats['successful_selections']}/{stats['total_selections']} successful selections")


# Backward compatibility - maintain singleton pattern
class UniverseSelector:
    """Backward compatible singleton wrapper for OptimizedUniverseSelector"""
    _instance: Optional[OptimizedUniverseSelector] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            fmp_api_key = kwargs.get('fmp_api_key') or (args[0] if args else None)
            cls._instance = OptimizedUniverseSelector(fmp_api_key)
        return cls._instance
    
    def __getattr__(self, name):
        """Delegate all attribute access to the optimized instance"""
        return getattr(self._instance, name)