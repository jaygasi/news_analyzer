import concurrent.futures
import queue
import uuid
import threading
import time
from utils.log_utils import *
from event_processors.press_release_processor_worker import PressReleaseProcessorWorker
from typing import List, Optional


class PressReleaseProcessorManager:
    """Singleton to manage the pool of Press Release Processor Workers"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, num_workers: int = 2):
        # Check if the instance already exists
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_shutting_down = threading.Event()
            self.message_queue: queue.Queue = queue.Queue()
            self.pool_size = num_workers
            self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
            self.workers: List[PressReleaseProcessorWorker] = []
            self.setup()

    def setup(self) -> None:
        """Setup workers"""
        for i in range(self.pool_size):
            worker = PressReleaseProcessorWorker(
                worker_id=str(uuid.uuid4()),
                message_queue=self.message_queue
            )
            self.workers.append(worker)

    def submit_message(self, message: dict) -> None:
        """Submit message for processing"""
        if not self.is_shutting_down.is_set():
            self.message_queue.put(message)

    def start(self) -> None:
        """Start the press release processor workers"""
        logi("Starting press release processor workers...")
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size)
        
        for worker in self.workers:
            self.executor.submit(worker.run)

    def stop(self) -> None:
        """Stop the press release processor manager with proper cleanup"""
        logi("Stopping press release processor workers...")
        self.is_shutting_down.set()
        
        # Stop all workers
        for worker in self.workers:
            try:
                worker.stop()
            except Exception as e:
                logw(f"Error stopping press release worker: {e}")
        
        # Give workers time to stop gracefully
        time.sleep(2)
        
        # Shutdown executor
        if self.executor:
            try:
                self.executor.shutdown(wait=False)
                time.sleep(3)
            except Exception as e:
                logw(f"Error during press release processor shutdown: {e}")
        
        # Clear remaining queue items
        try:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logw(f"Error clearing press release queue: {e}")
        
        logi("✅ Press release processor manager shutdown complete")