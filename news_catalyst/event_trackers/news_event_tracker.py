"""
Optimized news event tracker with improved error handling and performance
"""
import asyncio
import uuid
from typing import Dict, Optional, Set
from datetime import datetime

import pandas as pd

from data_loaders.fmp_data_loader import FmpDataLoader
from event_processors.news_processor_manager import NewsProcessorManager
from universe_selection.universe_selector import UniverseSelector
from utils.df_utils import generate_unique_id
from utils.log_utils import logi, logw, loge, logd
from utils.market_calendar import MarketCalendar
from data_loaders.tiingo_data_loader import TiingoDataLoader
from config import RESULTS_DIR


class OptimizedNewsEventTracker:
    """Enhanced news event tracker with improved performance and reliability"""
    
    def __init__(self, 
                 fmp_api_key: str, 
                 tiingo_api_key: str, 
                 data_collection_interval: int = 60,
                 max_processed_ids: int = 10000) -> None:
        self.fmp_data_loader = FmpDataLoader(fmp_api_key)
        self.tiingo_data_loader = TiingoDataLoader(tiingo_api_key) if tiingo_api_key else None
        self.data_collection_interval = data_collection_interval
        self.market_calendar = MarketCalendar('NYSE')
        self.universe_selector = UniverseSelector(fmp_api_key)
        self.news_processor_manager = NewsProcessorManager()
        
        # State management
        self.is_running = False
        self.processed_news_ids: Dict[str, datetime] = {}
        self.max_processed_ids = max_processed_ids
        
        # Performance tracking
        self.total_articles_processed = 0
        self.valid_articles_sent = 0
        self.last_cleanup_time = datetime.now()
        
        # Required columns for news processing
        self.required_columns = ['news_id', 'symbol', 'publishedDate', 'title', 'text', 'content', 'url']

    def _cleanup_old_processed_ids(self) -> None:
        """Clean up old processed IDs to prevent memory bloat"""
        if len(self.processed_news_ids) <= self.max_processed_ids:
            return
            
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Remove IDs older than today
        old_ids = [
            news_id for news_id, timestamp in self.processed_news_ids.items()
            if timestamp < cutoff_time
        ]
        
        for news_id in old_ids:
            del self.processed_news_ids[news_id]
        
        # If still too many, keep only the most recent half
        if len(self.processed_news_ids) > self.max_processed_ids:
            sorted_items = sorted(
                self.processed_news_ids.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            self.processed_news_ids = dict(sorted_items[:self.max_processed_ids // 2])
        
        logd(f"Cleaned up processed IDs, kept {len(self.processed_news_ids)} entries")

    def _validate_and_prepare_news_data(self, news_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate and prepare news data for processing"""
        if news_df is None or news_df.empty:
            return None
        
        try:
            # Generate unique IDs if not present
            if 'news_id' not in news_df.columns:
                news_df['news_id'] = news_df.apply(
                    lambda row: generate_unique_id(
                        row.get('symbol', ''), 
                        str(row.get('publishedDate', '')), 
                        row.get('title', '')
                    ), 
                    axis=1
                )
            
            # Filter news without symbols
            if 'symbol' in news_df.columns:
                news_df = news_df[news_df['symbol'] != "None"]
                news_df = news_df[news_df['symbol'].notna()]
                news_df = news_df[news_df['symbol'] != ""]
            
            # Ensure required columns exist
            available_columns = [col for col in self.required_columns if col in news_df.columns]
            if len(available_columns) < 4:  # Need at least symbol, title, text, date
                logw(f"Insufficient columns in news data: {available_columns}")
                return None
            
            news_df = news_df[available_columns]
            
            # Filter by universe selection
            try:
                universe_symbol_list = self.universe_selector.get_symbol_list()
                if universe_symbol_list:
                    news_df = news_df[news_df['symbol'].isin(universe_symbol_list)]
            except Exception as e:
                logw(f"Error filtering by universe: {e}")
            
            return news_df if not news_df.empty else None
            
        except Exception as e:
            loge(f"Error validating news data: {e}")
            return None

    def _save_news_for_review(self, news_df: pd.DataFrame) -> None:
        """Save news data for review with error handling"""
        try:
            import os
            from pathlib import Path
            
            # Ensure results directory exists
            results_path = Path(RESULTS_DIR)
            results_path.mkdir(parents=True, exist_ok=True)
            
            file_path = results_path / "news_df_temp.csv"
            news_df.to_csv(file_path, index=False)
            logd(f"Saved {len(news_df)} news articles for review")
            
        except Exception as e:
            logw(f"Could not save news temp file: {e}")

    def _process_news_articles(self, news_df: pd.DataFrame) -> int:
        """Process news articles and return count of articles sent for processing"""
        articles_sent = 0
        current_time = datetime.now()
        
        for _, row in news_df.iterrows():
            if not self.is_running:
                break
                
            try:
                # Check if already processed
                news_id = row.get('news_id', '')
                if news_id in self.processed_news_ids:
                    continue
                
                # Prepare message
                news_dict = row.to_dict()
                news_dict.update({
                    'message_id': str(uuid.uuid4()),
                    'date': row.get('publishedDate'),
                    'created_date': current_time
                })
                
                # Submit to processor
                self.news_processor_manager.submit_message(news_dict)
                
                # Track as processed
                self.processed_news_ids[news_id] = current_time
                articles_sent += 1
                
            except Exception as e:
                loge(f"Error processing news article: {e}")
                continue
        
        return articles_sent

    async def _fetch_and_process_news(self) -> bool:
        """Fetch and process news from all sources"""
        try:
            # Fetch news from RSS feed (primary source)
            news_df = await asyncio.to_thread(
                self.fmp_data_loader.fetch_stock_news_rss_feed
            )
            
            # Optional: Add Tiingo news (if available and configured)
            if self.tiingo_data_loader:
                try:
                    tiingo_news_df = await asyncio.to_thread(
                        self.tiingo_data_loader.fetch_latest_news_articles
                    )
                    if tiingo_news_df is not None and not tiingo_news_df.empty:
                        if news_df is not None and not news_df.empty:
                            news_df = pd.concat([news_df, tiingo_news_df], axis=0, ignore_index=True)
                        else:
                            news_df = tiingo_news_df
                except Exception as e:
                    logw(f"Error fetching Tiingo news: {e}")
            
            # Validate and prepare data
            news_df = self._validate_and_prepare_news_data(news_df)
            if news_df is None or news_df.empty:
                return False
            
            logd(f"{len(news_df)} news articles returned")
            
            # Save for review
            self._save_news_for_review(news_df)
            
            # Process articles
            articles_sent = self._process_news_articles(news_df)
            
            # Update statistics
            self.total_articles_processed += len(news_df)
            self.valid_articles_sent += articles_sent
            
            if articles_sent > 0:
                logi(f"Processed {len(news_df)} articles, sent {articles_sent} for analysis")
            
            return True
            
        except Exception as e:
            loge(f"Error in news fetch and process cycle: {e}")
            return False

    async def start(self) -> None:
        """Start the optimized news event tracker"""
        self.is_running = True
        logi("🚀 Starting optimized news event tracker...")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while self.is_running:
                try:
                    # Periodic cleanup
                    if (datetime.now() - self.last_cleanup_time).total_seconds() > 3600:  # Every hour
                        self._cleanup_old_processed_ids()
                        self.last_cleanup_time = datetime.now()
                    
                    # Fetch and process news
                    success = await self._fetch_and_process_news()
                    
                    if success:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logw(f"Too many consecutive errors ({consecutive_errors}), extended wait")
                            await asyncio.sleep(self.data_collection_interval * 2)
                            consecutive_errors = 0
                    
                    # Wait for next iteration
                    await asyncio.sleep(self.data_collection_interval)
                    
                except asyncio.CancelledError:
                    logi("📰 NewsEventTracker cancelled")
                    break
                except Exception as ex:
                    consecutive_errors += 1
                    loge(f"Exception in NewsEventTracker: {ex}")
                    
                    # Progressive backoff on errors
                    error_delay = min(self.data_collection_interval * consecutive_errors, 300)
                    await asyncio.sleep(error_delay)
                    
        except Exception as e:
            loge(f"Fatal error in NewsEventTracker: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the news event tracker with statistics"""
        self.is_running = False
        
        # Log final statistics
        logi(f"📰 NewsEventTracker stopped - "
             f"Total processed: {self.total_articles_processed}, "
             f"Valid sent: {self.valid_articles_sent}, "
             f"Tracked IDs: {len(self.processed_news_ids)}")

    def get_statistics(self) -> Dict[str, int]:
        """Get tracker statistics"""
        return {
            'total_articles_processed': self.total_articles_processed,
            'valid_articles_sent': self.valid_articles_sent,
            'tracked_news_ids': len(self.processed_news_ids),
            'is_running': int(self.is_running)
        }


# Backward compatibility alias
NewsEventTracker = OptimizedNewsEventTracker