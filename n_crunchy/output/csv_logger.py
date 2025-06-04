import csv
import os
from datetime import datetime
from config import CSV_LOG_FILE
from utils.utils import get_logger

logger = get_logger(__name__)

class CSVLogger:
    """
    Logs trading decisions to a CSV file.
    """
    def __init__(self, log_file=CSV_LOG_FILE):
        self.log_file = log_file
        self.headers = [
            'timestamp',
            'ticker',
            'decision',
            'confidence',
            'news_score',
            'technical_score',
            'reasoning',
            'keywords_matched' # Added per requirements for backtesting transparency
        ]
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensures the CSV file and its directory exist, writing headers if new."""
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            logger.info(f"Created new CSV log file: {self.log_file}")
        else:
            logger.info(f"CSV log file already exists: {self.log_file}")

    def log_decision(self, decision_data: dict):
        """
        Logs a single decision to the CSV file.
        decision_data should contain keys corresponding to self.headers.
        """
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                
                # Prepare data row, ensuring all headers are present
                row_data = {header: decision_data.get(header, '') for header in self.headers}
                row_data['timestamp'] = datetime.utcnow().isoformat()
                
                # Handle 'keywords_matched' which might come from news_details
                if 'news_details' in decision_data and 'keywords:' in decision_data['news_details']:
                    row_data['keywords_matched'] = decision_data['news_details'].split('keywords:')[1].strip()
                else:
                    row_data['keywords_matched'] = ''

                writer.writerow(row_data)
            logger.info(f"Logged decision for {decision_data.get('ticker')}: {decision_data.get('decision')}")
        except IOError as e:
            logger.error(f"Error writing to CSV log file {self.log_file}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while logging decision: {e}")