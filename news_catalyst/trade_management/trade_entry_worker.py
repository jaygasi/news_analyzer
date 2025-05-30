import asyncio
import uuid
import pandas as pd
from utils.log_utils import *
from utils.trade_log_utils import calculate_bid_ask_spread
from config import *
from trade_management.trade_log import TradeLog
from trade_management.trade_exit_manager import TradeExitManager
from price_trackers.price_tracker import PriceTracker
import traceback
import queue
from typing import Optional

pd.options.mode.chained_assignment = None


class TradeEntryWorker:
    def __init__(self, worker_id: str, message_queue: queue.Queue):
        self.worker_id: str = worker_id
        self.message_queue: queue.Queue = message_queue
        self.price_tracker: PriceTracker = PriceTracker()
        self.trade_log: TradeLog = TradeLog()
        self.trade_exit_manager: TradeExitManager = TradeExitManager()
        self.is_running: bool = True

    def process_message(self, message: dict) -> None:
        """Process a trade entry message"""
        try:
            symbol = message.get('symbol')
            if not symbol:
                logw(f"Worker {self.worker_id}: Invalid message (missing symbol): {message}")
                return

            # Check if trade already exists
            trade_log_df = self.trade_log.get_trade_log()
            if trade_log_df is not None and not trade_log_df.empty:
                symbol_log_df = trade_log_df[trade_log_df['symbol'] == symbol]
                if not symbol_log_df.empty:
                    return

            ask_price = message.get('askPrice')
            bid_price = message.get('bidPrice')

            if ask_price is None or bid_price is None:
                logw(f"Worker {self.worker_id}: Missing askPrice or bidPrice in message: {message}")
                return

            bid_ask_spread = calculate_bid_ask_spread(ask_price, bid_price)
            if bid_ask_spread is None or bid_ask_spread > MAX_BID_ASK_SPREAD:
                return
            
            logd(f"Bid-ask spread for stock {symbol}: {round(bid_ask_spread, 2)}")
            message['bid_ask_spread'] = bid_ask_spread

            logged_message = self.log_trade_entry(message)
            self.trade_exit_manager.submit_message(logged_message)

        except Exception as ex:
            loge(f"Trade entry worker {self.worker_id}: Exception in process_message: {str(ex)}")

    def log_trade_entry(self, message: dict) -> dict:
        """Log trade entry"""
        message['trade_log_id'] = str(uuid.uuid4())
        message['entry_date'] = pd.Timestamp.now()
        message['entry_price'] = message.get('askPrice')
        message['exit_date'] = None
        message['exit_price'] = None

        self.trade_log.log_trade_entry(message)
        logd(f"Trade entry worker {self.worker_id}: Logged trade entry for {message.get('symbol')}")
        return message

    async def start_message_listener(self) -> None:
        """Listen for incoming messages"""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    if message:
                        self.process_message(message)
                        self.message_queue.task_done()
            except queue.Empty:
                await asyncio.sleep(0.5)
            except Exception as ex:
                loge(f"Trade entry worker {self.worker_id}: Exception in message listener: {str(ex)}")
                loge(f"Full traceback: {traceback.format_exc()}")

    def stop(self) -> None:
        """Stop the message listener"""
        logi(f"Worker {self.worker_id}: Stopping...")
        self.is_running = False

    def run(self) -> None:
        """Run the worker"""
        try:
            asyncio.run(self.start_message_listener())
        except Exception as ex:
            loge(f"Worker {self.worker_id}: Exception in run: {str(ex)}")