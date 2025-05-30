"""
Optimized news processor worker with improved validation and performance
"""
import queue
import time
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

from utils.log_utils import *
from news_event_analyzers.news_topic_detector import NewsTopicDetector, NewsTopic
from momentum_trackers.momentum_tracker_manager import MomentumTrackerManager
from universe_selection.universe_selector import UniverseSelector
from utils.notification_utils import send_notifications
from config import *
import pandas as pd


# Topic sentiment mapping - moved to module level for efficiency
NEWS_TOPIC_SENTIMENT_MAP = {
    NewsTopic.FDA_APPROVAL: "positive",
    NewsTopic.FDA_REJECTION: "negative", 
    NewsTopic.POSITIVE_PHASE2_RESULT: "positive",
    NewsTopic.NEGATIVE_PHASE2_RESULT: "negative",
    NewsTopic.POSITIVE_PHASE3_RESULT: "positive",
    NewsTopic.NEGATIVE_PHASE3_RESULT: "negative",
    NewsTopic.POSITIVE_FINANCIAL_PERFORMANCE: "positive",
    NewsTopic.NEGATIVE_FINANCIAL_PERFORMANCE: "negative",
    NewsTopic.MARKET_GROWTH: "positive",
    NewsTopic.INNOVATION: "positive",
    NewsTopic.STOCK_BUYBACK: "positive",
    NewsTopic.STOCK_SPLIT: "neutral",
    NewsTopic.LAWSUIT_WIN: "positive",
}

BIOTECH_TOPIC_LIST = [
    NewsTopic.FDA_APPROVAL, NewsTopic.FDA_REJECTION,
    NewsTopic.POSITIVE_PHASE2_RESULT, NewsTopic.NEGATIVE_PHASE2_RESULT,
    NewsTopic.POSITIVE_PHASE3_RESULT, NewsTopic.NEGATIVE_PHASE3_RESULT
]


@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_processed: int = 0
    valid_catalysts: int = 0
    rejected_old: int = 0
    rejected_topic: int = 0
    rejected_biotech: int = 0
    rejected_duplicate: int = 0
    notification_sent: int = 0
    notification_failed: int = 0


class NewsProcessorWorker:
    """
    Optimized news processor worker with enhanced validation, 
    better error handling, and improved performance
    """

    def __init__(self, worker_id: str, message_queue: queue.Queue,
                 news_sentiment_detector: Any, news_topic_detector: NewsTopicDetector,
                 notification_client: Optional[Any] = None, notification_type: str = "gmail"):
        
        self.worker_id = worker_id
        self.message_queue = message_queue
        self.news_sentiment_detector = news_sentiment_detector
        self.news_topic_detector = news_topic_detector
        self.momentum_tracker_manager = MomentumTrackerManager()
        self.universe_selector = UniverseSelector()
        self.shutdown_flag = False
        self.notification_client = notification_client
        self.notification_type = notification_type
        
        # Enhanced caching and performance optimization
        self.processed_news_cache: Set[str] = set()
        self.stock_info_cache: Dict[str, Optional[Dict]] = {}
        self.stats = ProcessingStats()
        
        # Rate limiting for logging spam prevention
        self.rejection_counters = defaultdict(int)
        self.last_stats_log = datetime.now()
        self.stats_log_interval = 300  # 5 minutes

    def _is_news_too_old(self, published_date: Any) -> Tuple[bool, float]:
        """Optimized news age calculation with proper timezone handling"""
        try:
            now = datetime.now(timezone.utc)
            
            if isinstance(published_date, str):
                try:
                    published_date = pd.to_datetime(published_date, utc=True)
                except Exception:
                    logw(f"Could not parse date: {published_date}")
                    return True, float('inf')
            
            # Ensure timezone consistency
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            elif published_date.tzinfo != timezone.utc:
                published_date = published_date.astimezone(timezone.utc)
            
            time_diff_minutes = (now - published_date).total_seconds() / 60
            return time_diff_minutes > MAX_NEWS_ARTICLE_AGE_MINS, time_diff_minutes
            
        except Exception as ex:
            loge(f"Error calculating news age: {str(ex)}")
            return True, float('inf')

    def _get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get stock info with improved caching"""
        if symbol not in self.stock_info_cache:
            try:
                stock_info_df = self.universe_selector.get_stock_info_by_symbol(symbol)
                if stock_info_df is not None and not stock_info_df.empty:
                    self.stock_info_cache[symbol] = stock_info_df.iloc[0].to_dict()
                else:
                    self.stock_info_cache[symbol] = None
            except Exception as ex:
                logw(f"Error getting stock info for {symbol}: {str(ex)}")
                self.stock_info_cache[symbol] = None
        
        return self.stock_info_cache[symbol]

    def _validate_basic_requirements(self, news_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Enhanced validation of basic news requirements"""
        # Symbol validation
        symbol = news_dict.get('symbol', '').strip()
        if not symbol:
            return False, "missing symbol"
        
        if len(symbol) > 10 or not symbol.isalnum():
            return False, "invalid symbol format"
        
        # Duplicate check
        news_id = news_dict.get('news_id') or news_dict.get('message_id')
        if news_id and news_id in self.processed_news_cache:
            self.stats.rejected_duplicate += 1
            return False, f"already processed {news_id}"
        
        # Date validation
        published_date = news_dict.get('date') or news_dict.get('publishedDate')
        if not published_date:
            return False, "missing publishedDate"
        
        is_too_old, age_minutes = self._is_news_too_old(published_date)
        if is_too_old:
            self.stats.rejected_old += 1
            return False, f"too old ({age_minutes:.1f} mins)"
        
        # Content validation
        title = news_dict.get('title', '').strip()
        text = news_dict.get('text', '').strip()
        if not title or not text:
            return False, "missing title or text"
        
        if len(title) < 10 or len(text) < 20:
            return False, "content too short"
        
        # Cache news ID to prevent reprocessing
        if news_id:
            self.processed_news_cache.add(news_id)
            
            # Periodically clean cache to prevent memory bloat
            if len(self.processed_news_cache) > 10000:
                # Keep only recent half
                recent_ids = list(self.processed_news_cache)[-5000:]
                self.processed_news_cache = set(recent_ids)
        
        return True, "valid"

    def _analyze_topic(self, news_dict: Dict[str, Any]) -> Tuple[NewsTopic, int, str]:
        """Optimized topic analysis"""
        title = news_dict.get('title', '').strip()
        text = news_dict.get('text', '').strip()
        
        try:
            topic, keyword_count = self.news_topic_detector.detect_topics(title, text)
            
            if topic == NewsTopic.UNKNOWN:
                self.stats.rejected_topic += 1
                return topic, keyword_count, "no recognizable topic"
            
            if keyword_count < MIN_NEWS_TOPIC_CONFIDENCE:
                return topic, keyword_count, f"low keyword confidence ({keyword_count})"
            
            # Add sentiment mapping
            topic_sentiment = NEWS_TOPIC_SENTIMENT_MAP.get(topic, "neutral")
            news_dict['topic_sentiment'] = topic_sentiment
            news_dict['news_topic'] = topic
            news_dict['topic_keyword_count'] = keyword_count
            
            return topic, keyword_count, "valid"
            
        except Exception as ex:
            loge(f"Error in topic analysis: {str(ex)}")
            return NewsTopic.UNKNOWN, 0, "analysis error"

    def _validate_biotech_stock(self, symbol: str, topic: NewsTopic) -> Tuple[bool, str]:
        """Optimized biotech stock validation"""
        if topic not in BIOTECH_TOPIC_LIST:
            return True, "not biotech topic"
        
        try:
            stock_info = self._get_stock_info(symbol)
            if not stock_info:
                self.stats.rejected_biotech += 1
                return False, "no stock information found"
            
            industry = stock_info.get('industry', '')
            if industry not in BIOTECH_INDUSTRY_LIST:
                self.stats.rejected_biotech += 1
                return False, f"not biotech stock (industry={industry})"
            
            return True, "valid biotech"
            
        except Exception as ex:
            loge(f"Error validating biotech stock {symbol}: {str(ex)}")
            return False, "validation error"

    def _log_rejection_stats(self) -> None:
        """Log rejection statistics periodically"""
        now = datetime.now()
        if (now - self.last_stats_log).total_seconds() > self.stats_log_interval:
            total_rejections = (self.stats.rejected_old + self.stats.rejected_topic + 
                              self.stats.rejected_biotech + self.stats.rejected_duplicate)
            
            if total_rejections > 0:
                logi(f"📊 Worker {self.worker_id} stats: "
                     f"Processed={self.stats.total_processed}, "
                     f"Valid={self.stats.valid_catalysts}, "
                     f"Rejected={total_rejections} "
                     f"(Old={self.stats.rejected_old}, "
                     f"Topic={self.stats.rejected_topic}, "
                     f"Biotech={self.stats.rejected_biotech}, "
                     f"Duplicate={self.stats.rejected_duplicate})")
            
            self.last_stats_log = now

    def process_message(self, news_dict: Dict[str, Any]) -> None:
        """Enhanced message processing with better error handling"""
        try:
            self.stats.total_processed += 1
            symbol = news_dict.get('symbol', '')
            
            # Step 1: Basic validation
            is_valid, reason = self._validate_basic_requirements(news_dict)
            if not is_valid:
                # Reduced logging frequency for common rejections
                self.rejection_counters[reason] += 1
                if self.rejection_counters[reason] % 50 == 0:
                    logd(f"News rejection #{self.rejection_counters[reason]} for {symbol}: {reason}")
                return
            
            # Step 2: Topic analysis
            topic, keyword_count, topic_reason = self._analyze_topic(news_dict)
            if topic_reason != "valid":
                # Only log topic rejections occasionally
                if self.rejection_counters['topic'] % 20 == 0:
                    logd(f"News for {symbol} topic rejected: {topic_reason}")
                self.rejection_counters['topic'] += 1
                return
            
            # Step 3: Biotech validation (if applicable)
            is_valid_biotech, biotech_reason = self._validate_biotech_stock(symbol, topic)
            if not is_valid_biotech:
                logd(f"News for {symbol} biotech validation failed: {biotech_reason}")
                return
            
            # Step 4: Success - log and process
            self.stats.valid_catalysts += 1
            topic_sentiment = news_dict.get('topic_sentiment', 'neutral')
            title = news_dict.get('title', '')
            
            logi(f"✅ VALID catalyst for {symbol}: topic={topic.value}, "
                 f"sentiment={topic_sentiment}, keywords={keyword_count}")
            logd(f"   Title: {title[:100]}...")
            
            # Step 5: Send notifications
            if self.notification_client and ENABLE_NOTIFICATIONS:
                try:
                    success = send_notifications(self.notification_client, symbol, news_dict, self.notification_type)
                    if success:
                        self.stats.notification_sent += 1
                    else:
                        self.stats.notification_failed += 1
                except Exception as ex:
                    self.stats.notification_failed += 1
                    logw(f"Failed to send notification for {symbol}: {str(ex)}")
            
            # Step 6: Submit to momentum tracker
            try:
                self.momentum_tracker_manager.submit_message(news_dict)
            except Exception as ex:
                loge(f"Failed to submit to momentum tracker for {symbol}: {str(ex)}")
            
            # Log stats periodically
            self._log_rejection_stats()
                
        except Exception as ex:
            loge(f"Exception processing message for {news_dict.get('symbol', 'unknown')}: {str(ex)}")

    def stop(self) -> None:
        """Stop the worker with final stats"""
        self.shutdown_flag = True
        
        # Log final statistics
        logi(f"NewsProcessorWorker {self.worker_id} final stats: "
             f"Processed={self.stats.total_processed}, "
             f"Valid={self.stats.valid_catalysts}, "
             f"Notifications={self.stats.notification_sent}")

    def run(self) -> None:
        """Enhanced worker run loop with better performance"""
        logi(f"NewsProcessorWorker {self.worker_id} starting")
        
        last_queue_check = datetime.now()
        queue_check_interval = 60  # Check queue size every minute
        
        while not self.shutdown_flag:
            try:
                # Process messages
                messages_processed_this_cycle = 0
                max_messages_per_cycle = 10  # Limit to prevent blocking
                
                while (not self.shutdown_flag and 
                       not self.message_queue.empty() and 
                       messages_processed_this_cycle < max_messages_per_cycle):
                    
                    try:
                        message = self.message_queue.get_nowait()
                        if message:
                            self.process_message(message)
                            messages_processed_this_cycle += 1
                            self.message_queue.task_done()
                    except queue.Empty:
                        break
                
                # Periodic queue size monitoring
                now = datetime.now()
                if (now - last_queue_check).total_seconds() > queue_check_interval:
                    queue_size = self.message_queue.qsize()
                    if queue_size > 100:
                        logw(f"Worker {self.worker_id} queue size is large: {queue_size}")
                    last_queue_check = now
                
                # Clean up caches periodically to prevent memory leaks
                if self.stats.total_processed % 1000 == 0 and self.stats.total_processed > 0:
                    if len(self.stock_info_cache) > 500:
                        # Keep only recent entries
                        cache_items = list(self.stock_info_cache.items())
                        self.stock_info_cache = dict(cache_items[-250:])
                        logd(f"Worker {self.worker_id} cleaned stock info cache")
                
                # Short sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as ex:
                loge(f"Exception in NewsProcessorWorker {self.worker_id}: {str(ex)}")
                time.sleep(1)
        
        logi(f"NewsProcessorWorker {self.worker_id} shutting down after processing {self.stats.total_processed} messages")