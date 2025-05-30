"""
Enhanced logging utilities with improved performance optimization and structured logging
"""
import os
import sys
import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import wraps

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

from config import LOG_DIR


class LogLevel(Enum):
    """Enhanced log levels with numeric values for efficient comparison"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogConfig:
    """Log configuration with comprehensive validation"""
    level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = False
    max_file_size: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"
    rotation: str = "daily"
    buffer_size: int = 8192
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization"""
        valid_sizes = ['KB', 'MB', 'GB']
        if not any(self.max_file_size.endswith(unit) for unit in valid_sizes):
            raise ValueError(f"max_file_size must end with one of: {valid_sizes}")
        
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")


class PerformanceOptimizedLogger:
    """High-performance logger with minimal overhead and async capabilities"""
    
    def __init__(self, config: LogConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._setup_complete = False
        self._context: Dict[str, Any] = {}
        
        # Performance tracking
        self._log_count = 0
        self._error_count = 0
        self._last_performance_log = datetime.now()
        
        # Level checking optimization (pre-compute for faster checks)
        self._current_level_value = config.level.value
        
        self._initialize_logger()
    
    def _initialize_logger(self) -> None:
        """Initialize logger with optimized configuration"""
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
        """Setup Loguru with performance-optimized configuration"""
        # Remove default handler
        logger.remove()
        
        # Console handler with optimized format
        if self.config.console_enabled:
            console_format = (
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
            
            logger.add(
                sys.stdout,
                level=self.config.level.name,
                format=console_format,
                colorize=True,
                enqueue=True,  # Thread-safe async logging
                catch=True,
                backtrace=False,  # Disable for performance
                diagnose=False   # Disable for performance
            )
        
        # File handler with rotation
        if self.config.file_enabled:
            log_file = Path(LOG_DIR) / "trading_system_{time:YYYY-MM-DD}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{function}:{line} | "
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
                catch=True,
                backtrace=False,
                diagnose=False
            )
        
        # JSON handler for structured logging (optional)
        if self.config.json_enabled:
            json_file = Path(LOG_DIR) / "structured_{time:YYYY-MM-DD}.json"
            
            def json_formatter(record: Dict[str, Any]) -> str:
                """Optimized JSON formatter"""
                return json.dumps({
                    "timestamp": record["time"].astimezone(timezone.utc).isoformat(),
                    "level": record["level"].name,
                    "function": record["function"],
                    "line": record["line"],
                    "message": record["message"],
                    "context": self._context
                }, separators=(',', ':')) + "\n"  # Compact JSON
            
            logger.add(
                str(json_file),
                level=self.config.level.name,
                format=json_formatter,
                rotation=self.config.rotation,
                retention=self.config.retention,
                enqueue=True
            )
    
    def _setup_standard_logging(self) -> None:
        """Fallback to optimized standard Python logging"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level.name))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.file_enabled:
            log_file = Path(LOG_DIR) / f"trading_system_{datetime.now().strftime('%Y-%m-%d')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.level.name))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Set root logger level
        logger.setLevel(getattr(logging, self.config.level.name))
    
    def _setup_fallback_logging(self) -> None:
        """Emergency fallback logging to console only"""
        self.config = LogConfig(file_enabled=False, json_enabled=False)
        print("Using emergency fallback logging", file=sys.stderr)
    
    def update_context(self, **kwargs: Any) -> None:
        """Update logging context efficiently"""
        with self._lock:
            self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context efficiently"""
        with self._lock:
            self._context.clear()
    
    def _should_log(self, level: LogLevel) -> bool:
        """Fast level checking without function call overhead"""
        return level.value >= self._current_level_value
    
    def _log_message(self, level: str, message: str, **kwargs: Any) -> None:
        """Optimized core logging method"""
        with self._lock:
            self._log_count += 1
            
            if level.upper() in ('ERROR', 'CRITICAL'):
                self._error_count += 1
        
        # Merge context efficiently
        if self._context or kwargs:
            log_kwargs = {**self._context, **kwargs} if kwargs else self._context
        else:
            log_kwargs = {}
        
        if LOGURU_AVAILABLE and self._setup_complete:
            # Use loguru with bound context
            if log_kwargs:
                bound_logger = logger.bind(**log_kwargs)
                getattr(bound_logger, level.lower())(message)
            else:
                getattr(logger, level.lower())(message)
        else:
            # Standard logging fallback
            extra_info = f" | {log_kwargs}" if log_kwargs else ""
            getattr(logger, level.lower())(f"{message}{extra_info}")
    
    # Optimized logging methods with early return
    def trace(self, message: str, **kwargs: Any) -> None:
        """Trace level logging with performance optimization"""
        if self._should_log(LogLevel.TRACE):
            self._log_message('trace', message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Debug level logging with performance optimization"""
        if self._should_log(LogLevel.DEBUG):
            self._log_message('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Info level logging with performance optimization"""
        if self._should_log(LogLevel.INFO):
            self._log_message('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Warning level logging with performance optimization"""
        if self._should_log(LogLevel.WARNING):
            self._log_message('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Error level logging - always processed"""
        self._log_message('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Critical level logging - always processed"""
        self._log_message('critical', message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback"""
        if LOGURU_AVAILABLE:
            logger.exception(message)
        else:
            logger.exception(message, extra=kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics"""
        with self._lock:
            uptime = (datetime.now() - self._last_performance_log).total_seconds()
            return {
                'total_logs': self._log_count,
                'error_logs': self._error_count,
                'error_rate': self._error_count / max(self._log_count, 1),
                'logs_per_second': self._log_count / max(uptime, 1),
                'context_keys': list(self._context.keys()),
                'setup_complete': self._setup_complete,
                'loguru_available': LOGURU_AVAILABLE
            }


# Global logger instance with optimized configuration
_log_config = LogConfig(
    level=LogLevel.DEBUG,
    console_enabled=True,
    file_enabled=True,
    json_enabled=False,
    rotation="daily",
    retention="30 days"
)

_global_logger = PerformanceOptimizedLogger(_log_config)


# High-performance logging functions with minimal overhead
def logd(message: str, **kwargs: Any) -> None:
    """Debug logging - highly optimized"""
    if _log_config.level.value <= LogLevel.DEBUG.value:
        _global_logger.debug(message, **kwargs)


def logi(message: str, **kwargs: Any) -> None:
    """Info logging - highly optimized"""
    if _log_config.level.value <= LogLevel.INFO.value:
        _global_logger.info(message, **kwargs)


def logw(message: str, **kwargs: Any) -> None:
    """Warning logging - highly optimized"""
    if _log_config.level.value <= LogLevel.WARNING.value:
        _global_logger.warning(message, **kwargs)


def loge(message: str, **kwargs: Any) -> None:
    """Error logging - always executed with minimal overhead"""
    _global_logger.error(message, **kwargs)


def logc(message: str, **kwargs: Any) -> None:
    """Critical logging - always executed"""
    _global_logger.critical(message, **kwargs)


def log_exception(message: str, **kwargs: Any) -> None:
    """Exception logging with traceback"""
    _global_logger.exception(message, **kwargs)


# Context management functions
def add_log_context(**kwargs: Any) -> None:
    """Add context to all logs efficiently"""
    _global_logger.update_context(**kwargs)


def clear_log_context() -> None:
    """Clear all log context efficiently"""
    _global_logger.clear_context()


# Configuration functions
def set_log_level(level: Union[LogLevel, str]) -> None:
    """Set global log level with validation"""
    global _log_config
    
    if isinstance(level, str):
        try:
            level = LogLevel[level.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level: {level}")
    
    _log_config = LogConfig(level=level)
    _global_logger._current_level_value = level.value
    logi(f"Log level set to {level.name}")


def get_log_stats() -> Dict[str, Any]:
    """Get comprehensive logging statistics"""
    return _global_logger.get_performance_stats()


# Performance monitoring decorator with caching
def log_performance(func_name: Optional[str] = None, 
                   threshold_ms: float = 100.0) -> Callable:
    """Optimized decorator to log function performance"""
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Only log if above threshold to reduce noise
                if execution_time_ms > threshold_ms:
                    if execution_time_ms > 1000:  # 1 second
                        logw(f"Slow function {name}: {execution_time_ms:.2f}ms")
                    else:
                        logd(f"Function {name}: {execution_time_ms:.2f}ms")
                
                return result
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                loge(f"Function {name} failed after {execution_time_ms:.2f}ms: {e}")
                raise
        
        return wrapper
    return decorator


# Backward compatibility functions
def setup_logger(log_file_name: Optional[str] = None, level: str = "DEBUG") -> None:
    """Setup logger - legacy compatibility function"""
    if log_file_name:
        logw(f"Custom log file names not supported in enhanced logging")
    
    set_log_level(level)
    logi("Enhanced logging system initialized")


def create_log_file() -> None:
    """Create log file - legacy compatibility (handled automatically)"""
    pass


# Initialize logging system
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    setup_logger()
except Exception as e:
    print(f"Failed to initialize logging: {e}", file=sys.stderr)