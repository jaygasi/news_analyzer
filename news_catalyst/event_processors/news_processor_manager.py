import concurrent.futures
import queue
import uuid
import threading
from datetime import datetime
from typing import Any, Optional

from utils.log_utils import *
from event_processors.news_processor_worker import NewsProcessorWorker
from news_event_analyzers.news_sentiment_detector import NewsSentimentDetector
from news_event_analyzers.news_topic_detector import NewsTopicDetector
from config import MAX_NEWS_ARTICLE_AGE_MINS, NEWS_AGE_BUFFER


class NewsProcessorManager:
    """Singleton to manage the pool of News Event Processor Workers"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 num_workers: int = 1, 
                 model: Optional[Any] = None, 
                 tokenizer: Optional[Any] = None, 
                 device: Optional[str] = None, 
                 notification_client: Optional[Any] = None,
                 notification_type: str = "gmail"):
        
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.is_shutting_down = threading.Event()
            self.message_queue: queue.Queue = queue.Queue()
            self.pool_size = num_workers
            self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
            self.workers = []
            
            # Initialize components
            self.news_sentiment_detector = NewsSentimentDetector(model, tokenizer, device)
            self.news_topic_detector = NewsTopicDetector()
            
            # Notification system integration
            self.notification_client = notification_client
            self.notification_type = notification_type
            
            # Log notification setup
            if self.notification_client:
                logi(f"📧 News processor using {notification_type.upper()} notifications")
            else:
                logw("📧 News processor running without notifications")
            
            # Create workers with notification support
            self._create_workers()

    def _create_workers(self) -> None:
        """Create worker threads with notification integration"""
        self.workers.clear()
        
        for i in range(self.pool_size):
            worker = NewsProcessorWorker(
                worker_id=str(uuid.uuid4()),
                message_queue=self.message_queue,
                news_sentiment_detector=self.news_sentiment_detector,
                news_topic_detector=self.news_topic_detector,
                notification_client=self.notification_client,
                notification_type=self.notification_type
            )
            self.workers.append(worker)

    def submit_message(self, message: dict) -> None:
        """Submit message for processing"""
        if not self.is_shutting_down.is_set():
            self.message_queue.put(message)

    def start(self) -> None:
        """Start the news event processor workers"""
        logi("🚀 Starting enhanced news event processor workers...")
        
        if self.executor is None:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size)
        
        for worker in self.workers:
            self.executor.submit(worker.run)

    def stop(self) -> None:
        """Enhanced stop method with better cleanup"""
        logi("🛑 Initiating news processor shutdown...")
        self.is_shutting_down.set()

        try:
            # Stop all workers
            for worker in self.workers:
                if hasattr(worker, 'stop'):
                    worker.stop()

            # Wait a moment for workers to stop gracefully
            import time
            time.sleep(2)

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=False)
                # Give it a few seconds to shutdown gracefully
                time.sleep(3)

            # Clear remaining queue items
            try:
                while not self.message_queue.empty():
                    try:
                        self.message_queue.get_nowait()
                        self.message_queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                logw(f"Error clearing message queue: {e}")

        except Exception as e:
            loge(f"Error during news processor shutdown: {e}")
        finally:
            logi("✅ News processor manager shutdown complete")

    def is_news_recent(self, news_timestamp: datetime) -> bool:
        """Check if news is within acceptable age range"""
        age_mins = (datetime.now() - news_timestamp).total_seconds() / 60
        max_age = MAX_NEWS_ARTICLE_AGE_MINS - NEWS_AGE_BUFFER

        if age_mins > max_age:
            logd(f"News is too old (age: {age_mins:.2f} mins, max: {max_age:.2f} mins)")
            return False
        return True