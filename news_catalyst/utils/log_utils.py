"""
Enhanced logging utilities with performance optimization and structured logging
"""
import os
import sys
import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

from config import LOG_DIR


class LogLevel(Enum):
    """Enhanced log levels with numeric values"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogConfig:
    """Log configuration with validation"""
    level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = False
    max_file_size: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"
    rotation: str = "daily"
    buffer_size: int = 8192
    
    def __post_init__(self):
        # Validate file size format
        if not any(self.max_file_size.endswith(unit) for unit in ['KB', 'MB', 'GB']):
            raise ValueError("max_file_size must end with KB, MB, or GB")


class StructuredLogger:
    """High-performance structured logger with async capabilities"""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self._lock = threading.RLock()
        self._setup_complete = False
        self._context: Dict[str, Any] = {}
        
        # Performance tracking
        self._log_count = 0
        self._error_count = 0
        
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with optimized configuration"""
        try:
            if LOGURU_AVAILABLE:
                self._setup_loguru()
            else:
                self._setup_standard_logging()
            
            self._setup_complete = True
            
        except Exception as e:
            print(f"Failed to setup logger: {e}", file=sys.stderr)
            self._setup_fallback_logging()
    
    def _setup_loguru(self) -> None:
        """Setup Loguru with enhanced configuration"""
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        if self.config.console_enabled:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            logger.add(
                sys.stdout,
                level=self.config.level.name,
                format=console_format,
                colorize=True,
                backtrace=True,
                diagnose=True,
                enqueue=True,  # Thread-safe logging
                catch=True
            )
        
        # File handler with rotation
        if self.config.file_enabled:
            log_file = Path(LOG_DIR) / "trading_system_{time:YYYY-MM-DD}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            
            logger.add(
                str(log_file),
                level=self.config.level.name,
                format=file_format,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                enqueue=True,
                backtrace=True,
                diagnose=True,
                catch=True
            )
        
        # JSON handler for structured logging
        if self.config.json_enabled:
            json_file = Path(LOG_DIR) / "structured_{time:YYYY-MM-DD}.json"
            
            def json_formatter(record):
                """Custom JSON formatter"""
                return json.dumps({
                    "timestamp": record["time"].astimezone(timezone.utc).isoformat(),
                    "level": record["level"].name,
                    "logger": record["name"],
                    "function": record["function"],
                    "line": record["line"],
                    "message": record["message"],
                    "context": self._context,
                    "thread": record["thread"].name,
                    "process": record["process"].id
                }) + "\n"
            
            logger.add(
                str(json_file),
                level=self.config.level.name,
                format=json_formatter,
                rotation=self.config.rotation,
                retention=self.config.retention,
                enqueue=True
            )
    
    def _setup_standard_logging(self) -> None:
        """Fallback to standard Python logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.level.name),
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.config.file_enabled:
            log_file = Path(LOG_DIR) / f"trading_system_{datetime.now().strftime('%Y-%m-%d')}.log"
            log_file.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.level.name))
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
    
    def _setup_fallback_logging(self) -> None:
        """Emergency fallback logging"""
        self.config.file_enabled = False
        self.config.json_enabled = False
        print("Using emergency fallback logging", file=sys.stderr)
    
    def add_context(self, **kwargs) -> None:
        """Add context to all subsequent logs"""
        with self._lock:
            self._context.update(kwargs)
    
    def remove_context(self, *keys) -> None:
        """Remove context keys"""
        with self._lock:
            for key in keys:
                self._context.pop(key, None)
    
    def clear_context(self) -> None:
        """Clear all context"""
        with self._lock:
            self._context.clear()
    
    def _log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log with context and performance tracking"""
        with self._lock:
            self._log_count += 1
            
            if level.upper() in ['ERROR', 'CRITICAL']:
                self._error_count += 1
        
        # Merge context with kwargs
        log_kwargs = {**self._context, **kwargs}
        
        if LOGURU_AVAILABLE and self._setup_complete:
            # Bind context to loguru
            bound_logger = logger.bind(**log_kwargs)
            getattr(bound_logger, level.lower())(message)
        else:
            # Standard logging
            extra_info = f" | Context: {log_kwargs}" if log_kwargs else ""
            getattr(logger, level.lower())(f"{message}{extra_info}")
    
    def trace(self, message: str, **kwargs) -> None:
        """Trace level logging"""
        if self.config.level.value <= LogLevel.TRACE.value:
            self._log_with_context('trace', message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Debug level logging"""
        if self.config.level.value <= LogLevel.DEBUG.value:
            self._log_with_context('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Info level logging"""
        if self.config.level.value <= LogLevel.INFO.value:
            self._log_with_context('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Warning level logging"""
        if self.config.level.value <= LogLevel.WARNING.value:
            self._log_with_context('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Error level logging"""
        if self.config.level.value <= LogLevel.ERROR.value:
            self._log_with_context('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Critical level logging"""
        if self.config.level.value <= LogLevel.CRITICAL.value:
            self._log_with_context('critical', message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback"""
        if LOGURU_AVAILABLE:
            logger.exception(message)
        else:
            logger.exception(message, extra=kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self._lock:
            return {
                'total_logs': self._log_count,
                'error_logs': self._error_count,
                'error_rate': self._error_count / max(self._log_count, 1),
                'context_keys': list(self._context.keys()),
                'setup_complete': self._setup_complete,
                'loguru_available': LOGURU_AVAILABLE
            }


# Global logger instance
_log_config = LogConfig(
    level=LogLevel.DEBUG,  # Can be configured via environment
    console_enabled=True,
    file_enabled=True,
    json_enabled=False,
    rotation="daily",
    retention="30 days"
)

_global_logger = StructuredLogger(_log_config)


# Performance optimized logging functions
def logd(message: str, **kwargs) -> None:
    """Debug logging - optimized"""
    if _log_config.level.value <= LogLevel.DEBUG.value:
        _global_logger.debug(message, **kwargs)


def logi(message: str, **kwargs) -> None:
    """Info logging - optimized"""
    if _log_config.level.value <= LogLevel.INFO.value:
        _global_logger.info(message, **kwargs)


def logw(message: str, **kwargs) -> None:
    """Warning logging - optimized"""
    if _log_config.level.value <= LogLevel.WARNING.value:
        _global_logger.warning(message, **kwargs)


def loge(message: str, **kwargs) -> None:
    """Error logging - always executed"""
    _global_logger.error(message, **kwargs)


def logc(message: str, **kwargs) -> None:
    """Critical logging - always executed"""
    _global_logger.critical(message, **kwargs)


def log_exception(message: str, **kwargs) -> None:
    """Exception logging with traceback"""
    _global_logger.exception(message, **kwargs)


# Context management functions
def add_log_context(**kwargs) -> None:
    """Add context to all logs"""
    _global_logger.add_context(**kwargs)


def remove_log_context(*keys) -> None:
    """Remove context keys"""
    _global_logger.remove_context(*keys)


def clear_log_context() -> None:
    """Clear all log context"""
    _global_logger.clear_context()


# Configuration functions
def set_log_level(level: Union[LogLevel, str]) -> None:
    """Set global log level"""
    global _log_config
    
    if isinstance(level, str):
        level = LogLevel[level.upper()]
    
    _log_config.level = level
    logi(f"Log level set to {level.name}")


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics"""
    return _global_logger.get_stats()


def setup_logger(log_file_name: Optional[str] = None, level: str = "DEBUG") -> None:
    """Setup logger - legacy compatibility function"""
    if log_file_name:
        logw(f"Custom log file names not supported in enhanced logging. Using: {log_file_name}")
    
    set_log_level(level)
    logi("Enhanced logging system initialized")


# Create log file function for backward compatibility
def create_log_file() -> None:
    """Create log file - legacy compatibility"""
    # This is handled automatically by the enhanced logger
    pass


# Performance monitoring decorator
def log_performance(func_name: Optional[str] = None):
    """Decorator to log function performance"""
    def decorator(func):
        import functools
        import time
        
        name = func_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > 1000:  # Log slow functions
                    logw(f"Slow function {name}: {execution_time:.2f}ms")
                elif execution_time > 100:
                    logd(f"Function {name}: {execution_time:.2f}ms")
                
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                loge(f"Function {name} failed after {execution_time:.2f}ms: {e}")
                raise
        
        return wrapper
    return decorator


# Initialize logging
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    setup_logger(f"trading-system-{today}.log")
except Exception as e:
    print(f"Failed to initialize logging: {e}", file=sys.stderr)