import concurrent.futures
import queue
import uuid
import threading
import time
from typing import Optional, Any

from utils.log_utils import *
from momentum_trackers.momentum_tracker_worker import MomentumTrackerWorker


class MomentumTrackerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_workers: int = 3, data_collection_interval: int = 5, 
                 sendgrid_api_key: str = "", notification_client: Optional[Any] = None, 
                 notification_type: str = "gmail"):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_shutting_down = threading.Event()
            self.message_queue: queue.Queue = queue.Queue()
            self.pool_size = num_workers
            self.data_collection_interval = data_collection_interval
            self.notification_client = notification_client
            self.notification_type = notification_type
            self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
            self.workers = []
            self.setup()

    def setup(self) -> None:
        """Setup workers"""
        for i in range(self.pool_size):
            worker = MomentumTrackerWorker(
                worker_id=str(uuid.uuid4()),
                message_queue=self.message_queue,
                data_collection_interval=self.data_collection_interval,
                notification_client=self.notification_client
            )
            self.workers.append(worker)

    def submit_message(self, message: dict) -> None:
        """Submit message for processing"""
        if not self.is_shutting_down.is_set():
            self.message_queue.put(message)

    def start(self) -> None:
        """Start momentum tracker workers"""
        logi("Starting Momentum Tracker workers...")
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size)
        
        for worker in self.workers:
            self.executor.submit(worker.run)

    def stop(self) -> None:
        """Stop momentum tracker workers with proper cleanup"""
        logi("Stopping Momentum Tracker workers...")
        self.is_shutting_down.set()
        
        # Stop all workers first
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logw(f"Error stopping momentum worker: {e}")
        
        # Give workers time to stop gracefully
        time.sleep(2)
        
        # Shutdown executor safely
        if self.executor:
            try:
                self.executor.shutdown(wait=False)
                time.sleep(3)
            except Exception as e:
                logw(f"Error during momentum tracker shutdown: {e}")
        
        # Clear remaining queue items
        try:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logw(f"Error clearing momentum queue: {e}")
        
        logi("✅ Momentum tracker manager stopped")