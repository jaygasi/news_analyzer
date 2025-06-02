"""
Optimized logging utility with proper encoding support and performance improvements
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Final
from config import CONFIG


class LoggerManager:
    """Singleton logger manager with optimized performance."""
    
    _instance: Optional[logging.Logger] = None
    _emoji_map: Final[Dict[str, str]] = {
        '🚀': '[START]', '📋': '[INFO]', '✅': '[OK]', '💰': '[MONEY]',
        '🎯': '[TARGET]', '📊': '[STATUS]', '🏁': '[STOP]', '⚠️': '[WARNING]',
        '📈': '[CHART]', '❌': '[ERROR]', '🔍': '[DEBUG]', '📝': '[LOG]'
    }
    
    @classmethod
    def get_logger(cls, name: str = "trading_system") -> logging.Logger:
        """Get or create logger instance with caching."""
        if cls._instance is None:
            cls._instance = cls._setup_logger(name)
        return cls._instance
    
    @classmethod
    def _setup_logger(cls, name: str) -> logging.Logger:
        """Setup logger with optimized configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        if logger.handlers:
            return logger
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_file = CONFIG.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    @classmethod
    def clean_message(cls, msg: str) -> str:
        """Remove emoji and problematic unicode characters efficiently."""
        # Fast path for ASCII-only messages
        if msg.isascii():
            return msg
        
        # Replace known emojis first
        for emoji, replacement in cls._emoji_map.items():
            if emoji in msg:
                msg = msg.replace(emoji, replacement)
        
        # Filter out remaining problematic unicode
        return ''.join(
            char for char in msg 
            if ord(char) < 0x1F600 or ord(char) > 0x1F64F
        )


# Global logger instance
logger = LoggerManager.get_logger()


def log_info(msg: str) -> None:
    """Log info message with optimized emoji removal."""
    logger.info(LoggerManager.clean_message(msg))


def log_error(msg: str) -> None:
    """Log error message with optimized emoji removal."""
    logger.error(LoggerManager.clean_message(msg))


def log_debug(msg: str) -> None:
    """Log debug message with optimized emoji removal."""
    logger.debug(LoggerManager.clean_message(msg))


def log_warning(msg: str) -> None:
    """Log warning message with optimized emoji removal."""
    logger.warning(LoggerManager.clean_message(msg))