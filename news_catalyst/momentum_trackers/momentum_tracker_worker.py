import asyncio
import queue
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

from price_trackers.price_tracker import PriceTracker
from trade_management.trade_entry_manager import TradeEntryManager
from news_event_analyzers.news_topic_keyword_lists import NewsTopicSentiment
from utils.log_utils import *
from utils.notification_utils import send_notifications
from config import *

pd.options.mode.chained_assignment = None


class MomentumTrackerWorker:
    def __init__(self, worker_id: str, message_queue: queue.Queue,
                 data_collection_interval: int = 5, notification_client: Optional[Any] = None):
        self.worker_id = worker_id
        self.data_collection_interval = data_collection_interval
        self.notification_client = notification_client
        self.message_queue = message_queue
        self.price_tracker = PriceTracker()
        self.symbols_tracking: Dict[str, datetime] = {}
        self.message_dict: Dict[str, Dict[str, Any]] = {}
        self.trade_entry_manager = TradeEntryManager()
        self.is_running = True

    def check_is_tracking_expired(self, start_date: datetime) -> bool:
        """Check if tracking has expired"""
        now = datetime.now()
        tracking_mins = (now - start_date).total_seconds() / 60
        return tracking_mins >= MOMENTUM_MAX_TRACKING_MINS

    def check_price_momentum(self, prices_df: pd.DataFrame, start_date: datetime, 
                           topic_sentiment: NewsTopicSentiment) -> bool:
        """Check for price momentum"""
        try:
            df = prices_df[prices_df['date'] >= start_date]

            if len(df) == 0:
                logw("No data available after the specified start_date")
                return False

            if 'lastSalePrice' not in df.columns or df['lastSalePrice'].isnull().all():
                logw("Missing or invalid 'lastSalePrice' data in prices_df")
                return False

            initial_price = df['lastSalePrice'].iloc[0]
            final_price = df['lastSalePrice'].iloc[-1]

            if initial_price == 0:
                logw("Initial price is zero -> cannot calculate percent change")
                return False

            price_percent_change = ((final_price - initial_price) / initial_price)

            if topic_sentiment == NewsTopicSentiment.POSITIVE:
                return price_percent_change >= PRICE_PERCENT_CHANGE_THRESHOLD
            elif topic_sentiment == NewsTopicSentiment.NEGATIVE:
                return price_percent_change <= -PRICE_PERCENT_CHANGE_THRESHOLD

        except Exception as e:
            loge(f"Error in calculating momentum: {str(e)}")
        return False

    async def start_momentum_tracking(self) -> None:
        """Start momentum tracking loop"""
        i = 0
        while self.is_running:
            try:
                if not self.symbols_tracking:
                    await asyncio.sleep(self.data_collection_interval)
                    continue

                if i % 20 == 0:
                    logd(f"Momentum worker {self.worker_id} is tracking: {','.join(self.symbols_tracking.keys())}")
                    i = 0

                for symbol, start_date in list(self.symbols_tracking.items()):
                    if not self.is_running:
                        break
                        
                    if self.check_is_tracking_expired(start_date):
                        logd(f"Tracking expired for {symbol}.")
                        self.symbols_tracking.pop(symbol, None)
                        self.message_dict.pop(symbol, None)
                        continue

                    all_prices_df = self.price_tracker.get_prices()
                    if all_prices_df.empty:
                        await asyncio.sleep(self.data_collection_interval)
                        continue

                    message = self.message_dict.get(symbol)
                    if message is None:
                        logw(f"No message found for {symbol} -> skipping momentum analysis")
                        continue

                    topic_sentiment = message.get('topic_sentiment')

                    symbol_prices_df = all_prices_df[all_prices_df['symbol'] == symbol]
                    if len(symbol_prices_df) < MOMENTUM_LOOKBACK_PERIOD:
                        continue

                    prices_df = symbol_prices_df.copy()
                    prices_df['date'] = pd.to_datetime(prices_df['lastUpdated'], unit='ms', errors='coerce')

                    has_price_momentum = self.check_price_momentum(prices_df, start_date, topic_sentiment)
                    if has_price_momentum:
                        logd(f"Momentum detected for {symbol} and news topic: {topic_sentiment}")
                        if symbol in self.message_dict:
                            news_dict = self.message_dict[symbol]
                            self.trade_entry_manager.submit_message(news_dict)

                        if self.notification_client:
                            try:
                                send_notifications(self.notification_client, symbol, message, "gmail")
                            except Exception as ex:
                                logw(f"Failed to send notification for {symbol}: {str(ex)}")

                        self.symbols_tracking.pop(symbol, None)
                        self.message_dict.pop(symbol, None)
            except Exception as ex:
                loge(f"Exception in momentum tracking: {str(ex)}")
            i += 1
            await asyncio.sleep(self.data_collection_interval)

    def process_message(self, message: Dict[str, Any]) -> None:
        """Process incoming message"""
        symbol = message.get('symbol')
        if symbol:
            self.symbols_tracking[symbol] = datetime.now()
            self.message_dict[symbol] = message

    async def start_message_listener(self) -> None:
        """Listen for incoming messages"""
        while self.is_running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=1)
                    if message:
                        self.process_message(message)
                        self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as ex:
                loge(f"Error processing message in momentum tracker worker: {str(ex)}")
            await asyncio.sleep(0.5)

    def stop(self) -> None:
        """Stop the worker"""
        logi(f"Momentum worker {self.worker_id} stopping...")
        self.is_running = False

    async def run_tasks(self) -> None:
        """Run both tasks concurrently"""
        try:
            await asyncio.gather(
                self.start_message_listener(),
                self.start_momentum_tracking()
            )
        except asyncio.CancelledError:
            logi(f"Momentum worker {self.worker_id} tasks cancelled")
            raise

    def run(self) -> None:
        """Run tasks asynchronously"""
        try:
            asyncio.run(self.run_tasks())
        except asyncio.CancelledError:
            logi(f"Momentum worker {self.worker_id} cancelled")
        except Exception as e:
            loge(f"Momentum worker {self.worker_id} error: {str(e)}")
        finally:
            logi(f"Momentum worker {self.worker_id} shutdown complete")