import concurrent.futures
import queue
import uuid
import threading
import time
from utils.log_utils import *
from trade_management.trade_exit_worker import TradeExitWorker
import asyncio
from config import *
from typing import Optional


class TradeExitManager:
    """Singleton to manage the pool of Trade Exit Workers"""
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
            self.setup()

    def setup(self) -> None:
        """Setup workers"""
        for i in range(self.pool_size):
            worker = TradeExitWorker(
                worker_id=str(uuid.uuid4()),
                message_queue=self.message_queue
            )
            self.workers.append(worker)

    def submit_message(self, message: dict) -> None:
        """Submit message for processing"""
        if not self.is_shutting_down.is_set():
            self.message_queue.put(message)

    def start_workers(self) -> None:
        """Start exit workers"""
        logi("Starting trade exit workers...")
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size)
        
        for worker in self.workers:
            self.executor.submit(worker.run)

    def stop(self) -> None:
        """Stop trade exit manager with proper cleanup"""
        logi("Stopping trade exit manager...")
        self.is_running = False
        self.is_shutting_down.set()
        
        # Stop workers first
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logw(f"Error stopping exit worker: {e}")
        
        # Give workers time to stop
        time.sleep(2)
        
        # Shutdown executor
        if self.executor:
            try:
                self.executor.shutdown(wait=False)
                time.sleep(3)
            except Exception as e:
                logw(f"Error during exit manager shutdown: {e}")
        
        # Clear remaining queue items
        try:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logw(f"Error clearing exit queue: {e}")
        
        logi("✅ Trade exit manager stopped")

    async def start(self) -> None:
        """Start the trade exit manager"""
        try:
            self.is_running = True
            self.start_workers()
            
            # Keep running until shutdown
            while self.is_running and not self.is_shutting_down.is_set():
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logi("Trade Exit Manager shutting down.")
        finally:
            self.is_running = False
            self.stop()