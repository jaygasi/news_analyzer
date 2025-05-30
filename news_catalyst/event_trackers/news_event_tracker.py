import asyncio
import uuid
from data_loaders.fmp_data_loader import FmpDataLoader
from event_processors.news_processor_manager import NewsProcessorManager
from universe_selection.universe_selector import UniverseSelector
from utils.df_utils import generate_unique_id
from utils.log_utils import *
from utils.market_calendar import MarketCalendar
from data_loaders.tiingo_data_loader import TiingoDataLoader
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class NewsEventTracker:
    """Fetches news from FinancialModelingPrep RSS feed"""
    
    def __init__(self, fmp_api_key: str, tiingo_api_key: str, data_collection_interval: int = 60):
        self.fmp_data_loader = FmpDataLoader(fmp_api_key)
        self.tiingo_data_loader = TiingoDataLoader(tiingo_api_key)
        self.data_collection_interval = data_collection_interval
        self.market_calendar = MarketCalendar('NYSE')
        self.universe_selector = UniverseSelector(fmp_api_key)
        self.is_running = False
        self.processed_news_ids: Dict[str, str] = {}
        self.news_processor_manager = NewsProcessorManager()

    async def start(self) -> None:
        """Start the news event tracker"""
        self.is_running = True

        while self.is_running:
            try:
                # Optional: Check market open
                # if not self.market_calendar.is_market_open_now(extended_hours=True):
                #     await asyncio.sleep(self.data_collection_interval)
                #     logd(f"NewsEventTracker: market is closed")
                #     continue

                # Fetch news from RSS feed (non-blocking)
                news_df = await asyncio.to_thread(self.fmp_data_loader.fetch_stock_news_rss_feed)
                if news_df is None or len(news_df) == 0:
                    # No news received
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                # Generate unique ids for each record
                news_df['news_id'] = news_df.apply(
                    lambda row: generate_unique_id(row['symbol'], row['publishedDate'], row['title']), axis=1
                )

                # Optional: Fetch news from Tiingo
                # tiingo_news_df = self.tiingo_data_loader.fetch_latest_news_articles()
                # if tiingo_news_df is not None and not tiingo_news_df.empty:
                #     news_df = pd.concat([news_df, tiingo_news_df], axis=0, ignore_index=True)

                # Filter news without symbols
                news_df = news_df[news_df['symbol'] != "None"]

                # Select columns
                required_columns = ['news_id', 'symbol', 'publishedDate', 'title', 'text', 'content', 'url']
                available_columns = [col for col in required_columns if col in news_df.columns]
                news_df = news_df[available_columns]

                # Filter by universe selection symbols
                universe_symbol_list = self.universe_selector.get_symbol_list()
                news_df = news_df[news_df['symbol'].isin(universe_symbol_list)]
                if news_df is None or len(news_df) == 0:
                    # No news received
                    await asyncio.sleep(self.data_collection_interval)
                    continue
                logd(f"{len(news_df)} news articles returned")

                # Store for review
                try:
                    import os
                    path = os.path.join(RESULTS_DIR, "news_df_temp.csv")
                    news_df.to_csv(path)
                except Exception as e:
                    logw(f"Could not save news temp file: {e}")

                # Process news events
                for _, row in news_df.iterrows():
                    if not self.is_running:
                        break
                        
                    # Check if the news_event was already processed
                    news_id = row['news_id']
                    if news_id in self.processed_news_ids:
                        continue

                    # Add message info
                    news_dict = row.to_dict()
                    news_dict['message_id'] = str(uuid.uuid4())
                    news_dict['date'] = row['publishedDate']
                    news_dict['created_date'] = datetime.today()

                    # Submit to event processor for sentiment analysis
                    self.news_processor_manager.submit_message(news_dict)

                    # Track events already processed
                    self.processed_news_ids[news_id] = "ok"

                # Sleep asynchronously
                await asyncio.sleep(self.data_collection_interval)
                
            except asyncio.CancelledError:
                logi("NewsEventTracker cancelled")
                break
            except Exception as ex:
                loge(f"Exception in NewsEventTracker: {str(ex)}")
                await asyncio.sleep(self.data_collection_interval)

    def stop(self) -> None:
        """Stop the news event tracker"""
        self.is_running = False
        logi("NewsEventTracker stopped")