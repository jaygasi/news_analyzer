from config import *
from threading import Lock
from utils.log_utils import *
from data_loaders.fmp_data_loader import FmpDataLoader
from utils.file_utils import store_csv
import asyncio
from typing import Optional, List
import numpy as np


class UniverseSelector:
    """Singleton class for universe selection"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, fmp_api_key: Optional[str] = None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_running = False
            self.fmp_data_loader = FmpDataLoader(fmp_api_key) if fmp_api_key else None
            self.symbol_list: List[str] = []
            self.stock_info_df = None
            self.lock = Lock()

    def perform_selection(self) -> None:
        """Perform universe selection"""
        try:
            if not self.fmp_data_loader:
                logw("FMP data loader not available - using empty universe")
                return
                
            logi("Running universe selection....")

            # Load list of stocks
            stock_list_df = self.fmp_data_loader.fetch_stock_screener_info(
                exchange_list=EXCHANGE_LIST,
                price_more_than=PRICE_MORE_THAN,
                price_lower_than=PRICE_LESS_THAN,
                volume_more_than=VOLUME_MORE_THAN,
                market_cap_lower_than=MARKET_CAP_LOWER_THAN,
                is_etf=False,
                is_fund=False,
                is_actively_trading=True,
                limit=STOCK_SCREENER_LIMIT
            )
            if stock_list_df is None or len(stock_list_df) == 0:
                logw(f"Stock screener didn't return any data")
                return

            # Filter out stocks from other exchanges
            stock_list_df = stock_list_df[~stock_list_df['symbol'].str.contains(r'\.\w{1,4}$')]

            symbol_list = stock_list_df['symbol'].unique()

            # Fetch real-time prices for bid/ask spread calculation
            prices_df = self.fmp_data_loader.fetch_realtime_prices()
            if prices_df is not None and len(prices_df) > 0:
                # Filter stocks that are in the stock screener list
                prices_df = prices_df[prices_df['symbol'].isin(symbol_list)]

                # Calculate bid/ask spread
                prices_df['spread_percentage'] = prices_df.apply(
                    lambda row: ((row['askPrice'] - row['bidPrice']) / row['bidPrice'])
                    if row['bidPrice'] > 0 else 1/1000000, axis=1
                )

                # Filter out stocks with a bid-ask spread percentage larger than the threshold
                prices_df_filtered = prices_df[prices_df['spread_percentage'] <= MAX_BID_ASK_SPREAD]

                # Keep only stocks present in the filtered prices_df
                filtered_symbols = prices_df_filtered['symbol'].unique()
                stock_list_df = stock_list_df[stock_list_df['symbol'].isin(filtered_symbols)]

            # Store list of stocks
            store_csv(RESULTS_DIR, 'stock_list_df.csv', stock_list_df)

            symbol_list = stock_list_df['symbol'].unique()
            with self.lock:
                # Convert to regular list if it's a numpy array
                if isinstance(symbol_list, np.ndarray):
                    self.symbol_list = symbol_list.tolist()
                else:
                    self.symbol_list = list(symbol_list)
                self.stock_info_df = stock_list_df

        except Exception as e:
            loge(f"Error running universe selection: {str(e)}")

    def stop(self) -> None:
        """Stop universe selector"""
        self.is_running = False
        logi("Universe selector shut down")

    def get_symbol_list(self) -> List[str]:
        """Get symbol list ensuring it's always a proper list"""
        with self.lock:
            if self.symbol_list is None:
                return []
            
            # Handle numpy arrays
            if hasattr(self.symbol_list, 'tolist'):
                return self.symbol_list.tolist()
            elif isinstance(self.symbol_list, (list, tuple)):
                return list(self.symbol_list)
            else:
                return []

    def get_stock_info_by_symbol(self, symbol: str):
        """Get stock info by symbol"""
        with self.lock:
            if (self.stock_info_df is not None and 
                hasattr(self.stock_info_df, 'empty') and 
                not self.stock_info_df.empty):
                stock_symbol_info_df = self.stock_info_df[self.stock_info_df['symbol'] == symbol]
                return stock_symbol_info_df
            return None