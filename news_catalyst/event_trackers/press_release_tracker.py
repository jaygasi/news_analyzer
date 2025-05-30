import asyncio
from data_loaders.fmp_data_loader import FmpDataLoader
from universe_selection.universe_selector import UniverseSelector
import uuid
from datetime import datetime
from utils.df_utils import generate_unique_id
from utils.market_calendar import MarketCalendar
from event_processors.news_processor_manager import NewsProcessorManager
from utils.log_utils import *
from typing import Dict


class PressReleaseTracker:
    """Fetches press releases from FinancialModelingPrep feed"""
    
    def __init__(self, fmp_api_key: str, data_collection_interval: int = 60):
        self.market_calendar = MarketCalendar('NYSE')
        self.fmp_data_loader = FmpDataLoader(fmp_api_key)
        self.universe_selector = UniverseSelector(fmp_api_key)
        self.data_collection_interval = data_collection_interval
        self.is_running = False
        self.news_processor_manager = NewsProcessorManager()
        self.processed_events: Dict[str, str] = {}

    async def start(self) -> None:
        """Start the press release tracker"""
        self.is_running = True

        while self.is_running:
            try:
                # Optional: Check market open
                # if not self.market_calendar.is_market_open_now(extended_hours=True):
                #     await asyncio.sleep(self.data_collection_interval)
                #     logd(f"PressReleaseTracker: market is closed")
                #     continue

                # Fetch press releases from FMP
                press_release_df = await asyncio.to_thread(self.fmp_data_loader.fetch_press_releases)
                if press_release_df is None or len(press_release_df) == 0:
                    # No press releases received
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                # Filter by universe selection symbols
                universe_symbol_list = self.universe_selector.get_symbol_list()
                press_release_df = press_release_df[press_release_df['symbol'].isin(universe_symbol_list)]
                if press_release_df is None or len(press_release_df) == 0:
                    # No news received
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                logd(f"{len(press_release_df)} press releases returned from FMP")

                # Generate unique ids for each record
                press_release_df['news_id'] = press_release_df.apply(
                    lambda row: generate_unique_id(row['symbol'], row['date'], row['title']), axis=1
                )

                # Process press releases
                for _, row in press_release_df.iterrows():
                    if not self.is_running:
                        break
                        
                    # Check if the press release was already published
                    news_id = row['news_id']
                    if news_id in self.processed_events:
                        continue
                    # Track press releases already processed
                    self.processed_events[news_id] = "ok"

                    # Add message info
                    news_dict = row.to_dict()
                    news_dict['message_id'] = str(uuid.uuid4())
                    news_dict['created_date'] = datetime.today()

                    # Submit to event processor for sentiment analysis
                    self.news_processor_manager.submit_message(news_dict)

                # Sleep asynchronously
                await asyncio.sleep(self.data_collection_interval)
                
            except asyncio.CancelledError:
                logi("PressReleaseTracker cancelled")
                break
            except Exception as ex:
                loge(f"Exception in PressReleaseTracker: {str(ex)}")
                await asyncio.sleep(self.data_collection_interval)

    def stop(self) -> None:
        """Stop the press release tracker"""
        self.is_running = False
        logi("PressReleaseTracker stopped")