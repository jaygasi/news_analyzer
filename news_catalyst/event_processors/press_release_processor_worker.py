import queue
import time
from datetime import datetime, timezone
from utils.log_utils import *
from ai_clients.openai_client import OpenAIClient
from news_event_analyzers.news_topic_detector import NewsTopicDetector, NewsTopic
from momentum_trackers.momentum_tracker_manager import MomentumTrackerManager
from config import MAX_NEWS_ARTICLE_AGE_MINS, MIN_NEWS_TOPIC_CONFIDENCE, USE_OPENAI_ANALYSIS, MIN_PHASE3_SUCCESS_SCORE_THRESHOLD
from typing import Dict, Optional


class PressReleaseProcessorWorker:
    """Worker class to process press releases with proper shutdown handling"""

    def __init__(self, worker_id: str, message_queue: queue.Queue):
        self.worker_id = worker_id
        self.message_queue = message_queue
        self.news_topic_detector = NewsTopicDetector()
        self.openai_client = OpenAIClient() if USE_OPENAI_ANALYSIS else None
        self.momentum_tracker_manager = MomentumTrackerManager()
        self.shutdown_flag = False
        self.processed_count = 0

    def process_message(self, news_dict: dict) -> None:
        """Process a single press release message"""
        try:
            # Validate message and ensure symbol exists
            symbol = news_dict.get('symbol', '').strip()
            if not symbol:
                return

            # Check age of the press release
            now = datetime.now(timezone.utc)
            published_date = news_dict.get('date', None)
            
            if published_date:
                if isinstance(published_date, str):
                    import pandas as pd
                    published_date = pd.to_datetime(published_date, utc=True)
                elif published_date.tzinfo is None:
                    published_date = published_date.replace(tzinfo=timezone.utc)
                    
                time_diff_minutes = (now - published_date).total_seconds() / 60
                if time_diff_minutes > MAX_NEWS_ARTICLE_AGE_MINS:
                    return

            # Detect news topics
            content = news_dict.get('content', '').strip()
            title = news_dict.get('title', '').strip()
            
            if not content and not title:
                return

            topic, keyword_count = self.news_topic_detector.detect_topics(title, content)
            if topic == NewsTopic.UNKNOWN:
                return

            news_dict['news_topic'] = topic
            news_dict['topic_keyword_count'] = keyword_count

            # Check topic confidence threshold
            if keyword_count < MIN_NEWS_TOPIC_CONFIDENCE:
                return

            # Process FDA approvals/rejections
            if topic in [NewsTopic.FDA_APPROVAL, NewsTopic.FDA_REJECTION]:
                logi(f"Detected FDA event: symbol={symbol}, title={title[:100]}, topic={topic.value}, keywords={keyword_count}")
                self.momentum_tracker_manager.submit_message(news_dict)

            # Handle positive Phase 3 events
            elif topic == NewsTopic.POSITIVE_PHASE3_RESULT:
                if self.openai_client and USE_OPENAI_ANALYSIS:
                    # Perform OpenAI news analysis
                    openai_result = self.openai_client.analyze_phase_results(content)
                    if openai_result is not None:
                        phase3_score = openai_result.get('phase_score', 0)
                        if phase3_score < MIN_PHASE3_SUCCESS_SCORE_THRESHOLD:
                            logi(f"Phase 3 score threshold not met: {phase3_score}, title: {title[:100]}")
                            return

                        logi(f"Positive Phase 3 catalyst: symbol={symbol}, title={title[:100]}, score={phase3_score}")
                        news_dict.update(openai_result)
                        self.momentum_tracker_manager.submit_message(news_dict)
                else:
                    # Determine positive Phase 3 success only from keyword analysis
                    logi(f"Positive Phase 3 catalyst (No AI): symbol={symbol}, title={title[:100]}, keywords={keyword_count}")
                    self.momentum_tracker_manager.submit_message(news_dict)

            # Handle negative Phase 3 results
            elif topic == NewsTopic.NEGATIVE_PHASE3_RESULT:
                logi(f"Negative Phase 3 catalyst: symbol={symbol}, title={title[:100]}, keywords={keyword_count}")
                self.momentum_tracker_manager.submit_message(news_dict)

        except Exception as ex:
            loge(f"Error processing message in worker {self.worker_id}: {str(ex)}")

    def stop(self) -> None:
        """Stop the worker"""
        self.shutdown_flag = True
        logi(f"PressReleaseProcessorWorker {self.worker_id} stop requested")

    def run(self) -> None:
        """Run the worker main loop"""
        logi(f"PressReleaseProcessorWorker {self.worker_id} starting")
        
        while not self.shutdown_flag:
            try:
                # Retrieve next message with a timeout
                try:
                    message = self.message_queue.get(timeout=1)
                    if message:
                        self.process_message(message)
                        self.processed_count += 1
                        self.message_queue.task_done()
                        
                        if self.processed_count % 50 == 0:
                            logi(f"Press release worker {self.worker_id} processed {self.processed_count} messages")
                            
                except queue.Empty:
                    continue

            except Exception as ex:
                loge(f"Exception in press release processor worker {self.worker_id}: {str(ex)}")

            time.sleep(0.5)
        
        logi(f"PressReleaseProcessorWorker {self.worker_id} shutting down after processing {self.processed_count} messages")