"""
Optimized logging utility with proper encoding support
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import CONFIG


class LoggerManager:
    """Singleton logger manager for consistent logging across the application."""
    
    _instance: Optional[logging.Logger] = None
    
    @classmethod
    def get_logger(cls, name: str = "trading_system") -> logging.Logger:
        """Get or create logger instance."""
        if cls._instance is None:
            cls._instance = cls._setup_logger(name)
        return cls._instance
    
    @classmethod
    def _setup_logger(cls, name: str) -> logging.Logger:
        """Setup logger with proper encoding support for cross-platform compatibility."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
        log_file = CONFIG.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Clean formatter without problematic characters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger


# Global logger instance
logger = LoggerManager.get_logger()


def _clean_message(msg: str) -> str:
    """Remove emoji and problematic unicode characters for cross-platform compatibility."""
    # Emoji replacements
    emoji_replacements = {
        '🚀': '[START]', '📋': '[INFO]', '✅': '[OK]', '💰': '[MONEY]',
        '🎯': '[TARGET]', '📊': '[STATUS]', '🏁': '[STOP]', '⚠️': '[WARNING]',
        '📈': '[CHART]'
    }
    
    cleaned = msg
    for emoji, replacement in emoji_replacements.items():
        cleaned = cleaned.replace(emoji, replacement)
    
    # Remove remaining emoji characters
    return ''.join(char for char in cleaned if ord(char) < 0x1F600 or ord(char) > 0x1F64F)


def log_info(msg: str) -> None:
    """Log info message with emoji removal for compatibility."""
    logger.info(_clean_message(msg))


def log_error(msg: str) -> None:
    """Log error message with emoji removal for compatibility."""
    logger.error(_clean_message(msg))


def log_debug(msg: str) -> None:
    """Log debug message with emoji removal for compatibility."""
    logger.debug(_clean_message(msg))


def log_warning(msg: str) -> None:
    """Log warning message with emoji removal for compatibility."""
    logger.warning(_clean_message(msg))