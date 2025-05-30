import threading
import pandas as pd
from utils.log_utils import *
from config import *
import asyncio
from utils.file_utils import *
import os
from typing import Optional


class TradeLog:
    """Singleton class to manage the trade log"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Check if the instance already exists
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.is_running = False
            self.trade_log_df = pd.DataFrame(columns=[
                "trade_log_id", "symbol", "entry_date", "entry_price",
                "exit_date", "exit_price", "exit_type"
            ])
            self.lock = threading.Lock()
            self.trade_log_path = os.path.join(TRADE_LOG_PATH, TRADE_LOG_FILE_NAME)

            # Load previous trade log csv
            self.load_trade_log()

    def load_trade_log(self) -> None:
        """Load existing trade log from file"""
        if os.path.exists(self.trade_log_path):
            with self.lock:
                loaded_df = load_csv(TRADE_LOG_PATH, TRADE_LOG_FILE_NAME)
                if loaded_df is not None:
                    self.trade_log_df = loaded_df

    def log_trade_entry(self, message: dict) -> None:
        """Logs a new trade entry by appending it to the trade log DataFrame"""
        try:
            # Convert the incoming message dictionary to a DataFrame
            message_df = pd.DataFrame([message])

            with self.lock:
                # Check for duplicate entries based on trade_log_id
                if message["trade_log_id"] in self.trade_log_df["trade_log_id"].values:
                    logw(f"Duplicate trade log entry detected for trade_log_id: {message['trade_log_id']}")
                    return

                # Append the new entry to the trade log
                self.trade_log_df = pd.concat([self.trade_log_df, message_df], axis=0, ignore_index=True)
                logd(f"Trade entry logged for trade_log_id: {message['trade_log_id']}, symbol: {message['symbol']}")

        except Exception as ex:
            loge(f"Exception in log_trade_entry for message {message}: {str(ex)}")

    def log_trade_exit(self, message: dict) -> None:
        """Log trade exit"""
        trade_log_id = message.get('trade_log_id')
        exit_price = message.get('exit_price')
        profit_loss = message.get('profit_loss')
        exit_type = message.get('exit_type')

        with self.lock:
            try:
                # Locate the row by trade_log_id
                row_index = self.trade_log_df.index[self.trade_log_df['trade_log_id'] == trade_log_id]

                if not row_index.empty:
                    # Update the row with new values
                    self.trade_log_df.loc[row_index, 'exit_price'] = exit_price
                    self.trade_log_df.loc[row_index, 'profit_loss'] = profit_loss
                    self.trade_log_df.loc[row_index, 'exit_type'] = exit_type
                    logd(f"Updated trade log for id {trade_log_id} with exit_price: {exit_price}, profit_loss: {profit_loss}")
                else:
                    logw(f"Trade log ID {trade_log_id} not found")
            except Exception as e:
                loge(f"Error updating trade log for id {trade_log_id}: {str(e)}")

    def get_lock(self) -> threading.Lock:
        """Get the trade log lock"""
        return self.lock

    def get_trade_log(self) -> Optional[pd.DataFrame]:
        """Get a copy of the trade log df"""
        with self.lock:
            if self.trade_log_df is not None and not self.trade_log_df.empty:
                return self.trade_log_df.copy()
            return None

    async def periodic_save_trade_log(self) -> None:
        """Periodically save the trade log to a file"""
        while self.is_running:
            try:
                with self.lock:
                    # Save the trade log DataFrame to a CSV file
                    store_csv(TRADE_LOG_PATH, TRADE_LOG_FILE_NAME, self.trade_log_df)
            except Exception as e:
                loge(f"Error saving trade log: {str(e)}")
            finally:
                # Wait for the specified interval before saving again
                await asyncio.sleep(60)

    async def start(self) -> None:
        """Start the trade log periodic save"""
        try:
            self.is_running = True
            # Create a task for periodic saving
            await self.periodic_save_trade_log()
        except asyncio.CancelledError:
            logi("Trade Log shut down.")

    def stop(self) -> None:
        """Stop the trade log"""
        self.is_running = False
        
        # Save one final time
        try:
            with self.lock:
                store_csv(TRADE_LOG_PATH, TRADE_LOG_FILE_NAME, self.trade_log_df)
        except Exception as e:
            loge(f"Error in final trade log save: {str(e)}")
        
        logi("Trade log shut down")