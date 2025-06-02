"""
Optimized logging utility with performance improvements and proper encoding
"""
import logging
import sys
from pathlib import Path
from typing import Optional, ClassVar
from config import CONFIG


class LoggerManager:
    """Singleton logger manager with optimized performance."""
    
    _instance: Optional[logging.Logger] = None
    _emoji_replacements: ClassVar[dict[str, str]] = {
        '🚀': '[START]', '📋': '[INFO]', '✅': '[OK]', '💰': '[MONEY]',
        '🎯': '[TARGET]', '📊': '[STATUS]', '🏁': '[STOP]', '⚠️': '[WARNING]',
        '📈': '[CHART]', '❌': '[ERROR]', '🔍': '[DEBUG]', '📝': '[LOG]'
    }
    
    @classmethod
    def get_logger(cls, name: str = "trading_system") -> logging.Logger:
        """Get or create singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls._create_logger(name)
        return cls._instance
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """Create optimized logger configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler with proper encoding
        log_file = CONFIG.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        # Optimized formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    @classmethod
    def clean_message(cls, message: str) -> str:
        """Efficiently clean emoji and problematic characters from log messages."""
        # Fast path for ASCII-only messages
        if message.isascii():
            return message
        
        # Replace known emoji patterns
        cleaned = message
        for emoji, replacement in cls._emoji_replacements.items():
            cleaned = cleaned.replace(emoji, replacement)
        
        # Filter out remaining problematic Unicode characters
        # Focus on emoji range that commonly causes issues
        return ''.join(
            char for char in cleaned 
            if not (0x1F600 <= ord(char) <= 0x1F64F)  # Emoticons block
        )


# Module-level logger instance
_logger = LoggerManager.get_logger()


def log_info(message: str) -> None:
    """Log info message with emoji cleaning."""
    _logger.info(LoggerManager.clean_message(message))


def log_error(message: str) -> None:
    """Log error message with emoji cleaning."""
    _logger.error(LoggerManager.clean_message(message))


def log_debug(message: str) -> None:
    """Log debug message with emoji cleaning."""
    _logger.debug(LoggerManager.clean_message(message))


def log_warning(message: str) -> None:
    """Log warning message with emoji cleaning."""
    _logger.warning(LoggerManager.clean_message(message))