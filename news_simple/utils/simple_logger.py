"""
Optimized logging utility with proper encoding support
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from config import CONFIG


def setup_logger(name: str = "trading_system") -> logging.Logger:
    """Setup logger with proper encoding support for cross-platform compatibility."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
        log_file = CONFIG.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter without problematic characters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def log_info(msg: str) -> None:
    """Log info message with emoji removal for compatibility."""
    # Remove emojis and problematic unicode characters
    cleaned_msg = _clean_message(msg)
    logger.info(cleaned_msg)


def log_error(msg: str) -> None:
    """Log error message with emoji removal for compatibility."""
    cleaned_msg = _clean_message(msg)
    logger.error(cleaned_msg)


def log_debug(msg: str) -> None:
    """Log debug message with emoji removal for compatibility."""
    cleaned_msg = _clean_message(msg)
    logger.debug(cleaned_msg)


def log_warning(msg: str) -> None:
    """Log warning message with emoji removal for compatibility."""
    cleaned_msg = _clean_message(msg)
    logger.warning(cleaned_msg)


def _clean_message(msg: str) -> str:
    """Remove emoji and problematic unicode characters for cross-platform compatibility."""
    # Replace common emojis with text equivalents
    replacements = {
        '🚀': '[START]',
        '📋': '[INFO]',
        '✅': '[OK]',
        '💰': '[MONEY]',
        '🎯': '[TARGET]',
        '📊': '[STATUS]',
        '🏁': '[STOP]',
        '⚠️': '[WARNING]',
        '📈': '[CHART]'
    }
    
    cleaned = msg
    for emoji, replacement in replacements.items():
        cleaned = cleaned.replace(emoji, replacement)
    
    # Remove any remaining emoji characters (basic approach)
    cleaned = ''.join(char for char in cleaned if ord(char) < 0x1F600 or ord(char) > 0x1F64F)
    
    return cleaned