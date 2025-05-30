import concurrent.futures
import queue
import uuid
import threading
import time
from utils.log_utils import *
from trade_management.trade_entry_worker import TradeEntryWorker
from price_trackers.price_tracker import PriceTracker
import asyncio
from config import *
from typing import Optional


class TradeEntryManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_workers: int = 3, data_collection_interval: int = 5):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_running = False
            self.is_shutting_down = threading.Event()
            self.message_queue: queue.Queue = queue.Queue()
            self.pool_size = num_workers
            self.data_collection_interval = data_collection_interval
            self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
            self.workers = []
            self.price_tracker = PriceTracker()
            self.setup()

    def setup(self) -> None:
        """Setup the executor and workers"""
        for i in range(self.pool_size):
            worker = TradeEntryWorker(
                worker_id=str(uuid.uuid4()),
                message_queue=self.message_queue
            )
            self.workers.append(worker)

    async def check_momentum_signals(self) -> None:
        """Check for momentum signals"""
        while self.is_running and not self.is_shutting_down.is_set():
            try:
                prices_df = self.price_tracker.get_prices()

                if 'has_momentum' not in prices_df.columns:
                    continue

                momentum_signals_df = prices_df[prices_df['has_momentum'] == 1]
                if not momentum_signals_df.empty:
                    momentum_symbols = momentum_signals_df['symbol'].unique()

                    for symbol in momentum_symbols:
                        symbol_momentum_df = momentum_signals_df[momentum_signals_df['symbol'] == symbol]
                        symbol_momentum_dict = symbol_momentum_df.iloc[-1].to_dict()
                        self.submit_message(symbol_momentum_dict)
            except Exception as e:
                loge(f"Error checking momentum signals: {str(e)}")
            finally:
                await asyncio.sleep(self.data_collection_interval)

    def submit_message(self, message: dict) -> None:
        """Submit message for processing"""
        if not self.is_shutting_down.is_set():
            self.message_queue.put(message)

    def start_workers(self) -> None:
        """Start trade entry workers"""
        logi("Starting trade entry workers...")
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size)
        
        for worker in self.workers:
            self.executor.submit(worker.run)

    def stop(self) -> None:
        """Stop trade entry manager with proper cleanup"""
        logi("Stopping trade entry manager...")
        self.is_running = False
        self.is_shutting_down.set()
        
        # Stop workers first
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logw(f"Error stopping entry worker: {e}")
        
        # Give workers time to stop
        time.sleep(2)
        
        # Shutdown executor
        if self.executor:
            try:
                self.executor.shutdown(wait=False)
                time.sleep(3)
            except Exception as e:
                logw(f"Error during entry manager shutdown: {e}")
        
        # Clear remaining queue items
        try:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logw(f"Error clearing entry queue: {e}")
        
        logi("✅ Trade entry manager stopped")

    async def start(self) -> None:
        """Start the trade entry manager"""
        try:
            self.is_running = True
            self.start_workers()
            await self.check_momentum_signals()
        except asyncio.CancelledError:
            logi("Trade Entry Manager shutting down.")
        finally:
            self.is_running = False
            self.stop()