import asyncio
import queue
import pandas as pd
from utils.log_utils import *
from config import *
from trade_management.trade_log import TradeLog
from price_trackers.price_tracker import PriceTracker
from utils.indicator_utils import *
from datetime import datetime
from typing import Dict, Any

# Disable SettingsCopy Warning
pd.options.mode.chained_assignment = None


class TradeExitWorker:
    """Worker class to detect trade exit events and update the trade log"""
    
    def __init__(self, worker_id: str, message_queue: queue.Queue, data_collection_interval: int = 5):
        self.worker_id = worker_id
        self.message_queue = message_queue
        self.data_collection_interval = data_collection_interval
        self.price_tracker = PriceTracker()
        self.trade_log = TradeLog()
        self.symbol_tracking: Dict[str, Dict] = {}
        self.message_dict: Dict[str, Dict] = {}
        self.is_running = True

    def process_message(self, message: dict) -> None:
        """Process trade exit message"""
        symbol = message.get('symbol')
        logd(f"Trade exit worker {self.worker_id} processing trade exit message for {symbol}")

        # Check if this symbol is already being tracked
        if symbol in self.symbol_tracking:
            return

        # Calculate stop from entry sell price
        bid_price = message.get('bidPrice')
        if bid_price:
            message['stop_price'] = bid_price * (1 - STOP_LOSS_PERCENT)

        # Add the symbol to the list of symbols to track
        self.symbol_tracking[symbol] = message

    def resample_to_interval(self, prices_df: pd.DataFrame, interval: str = '5min') -> pd.DataFrame:
        """Resample the data to the target interval"""
        # Ensure 'date' column is set as the index
        prices_df.set_index('date', inplace=True)

        # Resample prices, aggregating values
        resampled_df = prices_df.resample(interval).agg({
            'lastSalePrice': 'last',
            'volume': 'sum',
            'bidPrice': 'last',
            'askPrice': 'last'
        }).dropna()

        # Reset index for further processing
        resampled_df.reset_index(inplace=True)
        return resampled_df

    def calculate_exit_signals(self, symbol: str, prices_df: pd.DataFrame) -> bool:
        """Calculate exit signals"""
        # Get last price
        current_bid_price = prices_df['bidPrice'].iloc[-1]

        # Get stop price
        has_stop_signal = False
        stop_price = self.symbol_tracking.get(symbol, {}).get('stop_price')
        if stop_price:
            if current_bid_price < stop_price:
                has_stop_signal = True
                logd(f"Stop signal detected for {symbol}")
        
        # Calculate downward trend on the smoothed close price
        prices_df['close_smoothed_slope'] = calculate_trend(prices_df,
                                                            target_column='lastSalePrice',
                                                            out_column='close_smoothed_slope',
                                                            bandwidth=9,
                                                            slope_window=3)
        # Get last close slope
        last_close_slope = prices_df['close_smoothed_slope'].iloc[-1]
        has_downward_trend_signal = last_close_slope < 0.0
        if has_downward_trend_signal:
            logd(f"Downward trend signal detected for {symbol}")

        has_exit_signal = has_stop_signal or has_downward_trend_signal

        return has_exit_signal

    def check_is_trade_closed(self, symbol: str) -> bool:
        """Check if trade is already closed"""
        trade_log_df = self.trade_log.get_trade_log()
        if trade_log_df is None or len(trade_log_df) == 0:
            return False

        symbol_trade_log_df = trade_log_df[trade_log_df['symbol'] == symbol]
        if len(symbol_trade_log_df) == 0:
            return True

        # Check if the last trade log entry has an exit date
        if symbol_trade_log_df['exit_date'].iloc[-1] is not None:
            return True
        return False

    async def start_exit_tracking(self) -> None:
        """Track trade exit based on exit criteria"""
        i = 0
        while self.is_running:
            try:
                if not self.symbol_tracking:
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                if i % 20 == 0:
                    logd(f"Trade exit worker {self.worker_id} is tracking: {','.join(self.symbol_tracking.keys())}")
                    i = 0

                # Fetch prices
                all_prices_df = self.price_tracker.get_prices()
                if all_prices_df.empty:
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                # Iterate through symbols tracking and check exit criteria
                for symbol, message in list(self.symbol_tracking.items()):
                    if not self.is_running:
                        break
                        
                    # Check if the trade was closed by another worker
                    is_trade_closed = self.check_is_trade_closed(symbol)
                    if is_trade_closed:
                        # Remove symbol from tracking
                        self.symbol_tracking.pop(symbol, None)
                        logd(f"Trade exit worker: Trade is closed or record not found -> stopping exit tracking for {symbol}")
                        continue

                    # Filter prices for this symbol
                    symbol_prices_df = all_prices_df[all_prices_df['symbol'] == symbol]
                    if len(symbol_prices_df) == 0:
                        logd(f"No data for symbol {symbol}.")
                        continue

                    # Format date
                    prices_df = symbol_prices_df.copy()
                    prices_df['date'] = pd.to_datetime(prices_df['lastUpdated'], unit='ms', errors='coerce')

                    # Resample to target time interval
                    interval = '1min'
                    prices_df = self.resample_to_interval(prices_df, interval=interval)

                    # Check exit criteria
                    has_exit_signal = self.calculate_exit_signals(symbol, prices_df)
                    if has_exit_signal:
                        logd(f"Exit signal detected for {symbol}. Logging trade exit")
                        if symbol in self.message_dict:
                            self.log_trade_exit(message)

                        # Remove symbol from tracking
                        self.symbol_tracking.pop(symbol, None)
            except Exception as ex:
                loge(f"Exception in trade exit worker: {str(ex)}")
            i += 1
            await asyncio.sleep(self.data_collection_interval)

    def log_trade_exit(self, message: dict) -> None:
        """Log trade exit"""
        # Add exit information
        message['exit_date'] = datetime.today()
        message['exit_price'] = message.get('bid_price')
        message['exit_type'] = "close_slope_decline"

        # Log to the trade log
        self.trade_log.log_trade_exit(message)
        logd(f"Logged trade exit message for {message.get('symbol')}")

    async def start_message_listener(self) -> None:
        """Listen for incoming messages"""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=1)
                    if message:
                        self.process_message(message)
                        self.message_queue.task_done()
            except Exception as ex:
                loge(f"Error processing message in trade exit worker: {str(ex)}")
            await asyncio.sleep(0.5)

    def stop(self) -> None:
        """Stop the worker"""
        logi(f"Trade exit worker {self.worker_id}: Stopping...")
        self.is_running = False

    async def run_tasks(self) -> None:
        """Run both tasks concurrently"""
        await asyncio.gather(
            self.start_message_listener(),
            self.start_exit_tracking()
        )

    def run(self) -> None:
        """Run tasks asynchronously"""
        try:
            asyncio.run(self.run_tasks())
        except Exception as ex:
            loge(f"Trade exit worker {self.worker_id}: Exception in run: {str(ex)}")