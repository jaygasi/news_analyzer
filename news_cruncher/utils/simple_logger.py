"""
Simple logging utility for the financial news analysis system
Python 3.13.3 compatible
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import Config


class SimpleLogger:
    """Simple logger with file and console output"""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "financial_news_analyzer") -> logging.Logger:
        """Get or create logger instance"""
        if cls._instance is None:
            cls._instance = cls._create_logger(name)
        return cls._instance
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """Create logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(Config.LOG_FILE_PATH, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger


# Module-level logger instance
_logger = SimpleLogger.get_logger()


def log_info(message: str) -> None:
    """Log info message"""
    _logger.info(message)


def log_error(message: str) -> None:
    """Log error message"""
    _logger.error(message)


def log_debug(message: str) -> None:
    """Log debug message"""
    _logger.debug(message)


def log_warning(message: str) -> None:
    """Log warning message"""
    _logger.warning(message)