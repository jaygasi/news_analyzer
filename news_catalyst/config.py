"""
Enhanced configuration module with improved type safety, validation, and missing constants
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()


class Environment(Enum):
    """Environment enumeration for better config management"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class DataProvider(Enum):
    """Supported data providers with failover capability"""
    FMP = "fmp"
    POLYGON = "polygon" 
    IEX = "iex"
    ALPHA_VANTAGE = "alpha_vantage"


@dataclass(frozen=True)
class DirectoryConfig:
    """Immutable directory configuration with Path objects"""
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    trade_log_path: Path = field(default_factory=lambda: Path("trade_logs"))
    trade_log_file_name: str = "trade_log.csv"
    
    def __post_init__(self) -> None:
        """Ensure all directories exist"""
        for dir_path in [self.log_dir, self.cache_dir, self.results_dir, 
                        self.plots_dir, self.trade_log_path]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Data collection and processing configuration with validation"""
    # Real-time data intervals (seconds)
    price_tick_interval: float = 1.0
    news_check_interval: float = 30.0
    momentum_calc_interval: float = 5.0
    
    # API rate limits
    api_rate_limit_calls: int = 100
    api_rate_limit_period: int = 60
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # Data providers with priorities
    primary_provider: DataProvider = DataProvider.FMP
    backup_providers: List[DataProvider] = field(default_factory=lambda: [DataProvider.IEX, DataProvider.FMP])
    
    def __post_init__(self) -> None:
        """Validate configuration values"""
        if self.price_tick_interval <= 0:
            raise ValueError("price_tick_interval must be positive")
        if self.news_check_interval <= 0:
            raise ValueError("news_check_interval must be positive")
        if self.api_rate_limit_calls <= 0:
            raise ValueError("api_rate_limit_calls must be positive")


@dataclass(frozen=True)
class AIConfig:
    """AI and analysis configuration with model weights"""
    # Model ensemble weights (must sum to 1.0)
    finbert_weight: float = 0.4
    openai_weight: float = 0.3
    claude_weight: float = 0.2
    local_llm_weight: float = 0.1
    
    # Analysis thresholds
    min_confidence_threshold: float = 0.7
    consensus_threshold: float = 0.8
    
    # Feature flags
    use_openai: bool = False
    use_gemini: bool = True
    use_ollama: bool = False
    use_ensemble: bool = True
    
    def __post_init__(self) -> None:
        """Validate model weights sum to 1.0"""
        total_weight = (self.finbert_weight + self.openai_weight + 
                       self.claude_weight + self.local_llm_weight)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")


@dataclass(frozen=True)
class TradingConfig:
    """Trading and risk management configuration with validation"""
    # Position sizing (as percentages 0.0-1.0)
    max_position_size: float = 0.05  # 5% of portfolio
    max_portfolio_risk: float = 0.02  # 2% daily risk
    max_sector_exposure: float = 0.30  # 30% in one sector
    max_correlation: float = 0.7
    
    # Execution parameters
    max_bid_ask_spread: float = 0.05
    stop_loss_atr_multiplier: float = 2.0
    min_volume_threshold: int = 10000
    
    # Dynamic thresholds
    base_momentum_threshold: float = 0.01
    volatility_adjustment: bool = True
    regime_adjustment: bool = True
    
    def __post_init__(self) -> None:
        """Validate trading parameters"""
        if not 0 < self.max_position_size <= 1.0:
            raise ValueError("max_position_size must be between 0 and 1.0")
        if not 0 < self.max_portfolio_risk <= 1.0:
            raise ValueError("max_portfolio_risk must be between 0 and 1.0")


@dataclass(frozen=True)
class NotificationConfig:
    """Notification system configuration with comprehensive settings"""
    enable_notifications: bool = False
    use_gmail: bool = False
    use_sendgrid: bool = False
    
    # Rate limiting
    max_emails_per_minute: int = 25
    batch_size: int = 5
    batch_timeout: float = 3.0
    cooldown_minutes: int = 1
    duplicate_window_minutes: int = 5
    
    # Retry logic
    max_retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Email settings
    send_email_notifications: bool = True
    send_daily_summary: bool = True
    
    def __post_init__(self) -> None:
        """Validate notification settings"""
        if self.max_emails_per_minute <= 0:
            raise ValueError("max_emails_per_minute must be positive")
        if self.batch_timeout <= 0:
            raise ValueError("batch_timeout must be positive")


class OptimizedConfig:
    """Centralized configuration with validation and environment support"""
    
    def __init__(self, env: Environment = Environment.PRODUCTION) -> None:
        self.env = env
        self.directories = DirectoryConfig()
        self.data = DataConfig()
        self.ai = AIConfig()
        self.trading = TradingConfig()
        self.notifications = NotificationConfig()
        
        # Environment-specific overrides
        self._apply_environment_overrides()
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific settings"""
        if self.env == Environment.DEVELOPMENT:
            self._apply_dev_overrides()
        elif self.env == Environment.TESTING:
            self._apply_test_overrides()
    
    def _apply_dev_overrides(self) -> None:
        """Apply development environment settings"""
        object.__setattr__(self.data, 'api_rate_limit_calls', 10)
        object.__setattr__(self.notifications, 'enable_notifications', False)
    
    def _apply_test_overrides(self) -> None:
        """Apply testing environment settings"""
        object.__setattr__(self.trading, 'max_position_size', 0.01)
        object.__setattr__(self.notifications, 'enable_notifications', False)
    
    @property
    def exchange_list(self) -> str:
        """Supported exchanges as comma-separated string"""
        return "nyse,nasdaq,amex"
    
    @property
    def biotech_industries(self) -> List[str]:
        """Biotech industry classifications for filtering"""
        return ['Biotechnology', 'Medical - Diagnostics & Research', 'Pharmaceuticals']
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Securely retrieve API keys from environment with fallbacks"""
        # Try multiple environment variable patterns
        env_vars = [
            f"{provider.upper()}_API_KEY",
            f"{provider.lower()}_api_key",
            provider.upper(),
            provider.lower()
        ]
        
        for env_var in env_vars:
            key = os.getenv(env_var)
            if key:
                return key.strip()
        return None
    
    def validate_config(self) -> List[str]:
        """Comprehensive configuration validation"""
        errors = []
        
        # Trading validation
        if self.trading.max_position_size > 0.2:
            errors.append("Position size too large (>20%)")
        
        # Data validation
        if self.data.price_tick_interval < 0.05:
            errors.append("Price tick interval too aggressive (<0.05s)")
        
        # API key validation
        if not self.get_api_key('FMP') and self.data.primary_provider == DataProvider.FMP:
            errors.append("Missing FMP API key for primary provider")
        
        # Notification validation
        if self.notifications.enable_notifications and self.notifications.use_gmail:
            if not (self.get_api_key('gmail_email') and self.get_api_key('gmail_app_password')):
                errors.append("Gmail notifications enabled but credentials missing")
        
        return errors
    
    def get_directory_paths(self) -> Dict[str, str]:
        """Get all directory paths as strings for backward compatibility"""
        return {
            'log_dir': str(self.directories.log_dir),
            'cache_dir': str(self.directories.cache_dir),
            'results_dir': str(self.directories.results_dir),
            'plots_dir': str(self.directories.plots_dir),
            'trade_log_path': str(self.directories.trade_log_path)
        }


# Global configuration instance
CONFIG = OptimizedConfig()

# Create directories on import
CONFIG.directories.__post_init__()

# Backward compatibility - expose commonly used values as module constants
_paths = CONFIG.get_directory_paths()
LOG_DIR = _paths['log_dir']
CACHE_DIR = _paths['cache_dir']
RESULTS_DIR = _paths['results_dir']  # This was missing and causing errors
PLOTS_DIR = _paths['plots_dir']
TRADE_LOG_PATH = _paths['trade_log_path']
TRADE_LOG_FILE_NAME = CONFIG.directories.trade_log_file_name

# Trading parameters
MAX_BID_ASK_SPREAD = CONFIG.trading.max_bid_ask_spread
MAX_POSITION_SIZE = CONFIG.trading.max_position_size
STOP_LOSS_PERCENT = 0.02  # Will be replaced by dynamic ATR-based stops

# AI parameters
USE_OPENAI_ANALYSIS = CONFIG.ai.use_openai
MIN_NEWS_TOPIC_CONFIDENCE = 1
MAX_NEWS_ARTICLE_AGE_MINS = 60.0
NEWS_AGE_BUFFER = 5.0

# Data collection intervals
NEWS_DATA_COLLECTION_INTERVAL = int(CONFIG.data.news_check_interval)
PRICE_TRACKER_DATA_COLLECTION_INTERVAL = int(CONFIG.data.price_tick_interval * 10)
MOMENTUM_TRACKING_TIME_INTERVAL = int(CONFIG.data.momentum_calc_interval * 10)

# Notification settings with proper defaults
ENABLE_NOTIFICATIONS = CONFIG.notifications.enable_notifications
USE_GMAIL_NOTIFICATIONS = CONFIG.notifications.use_gmail
GMAIL_EMAIL = CONFIG.get_api_key('gmail_email') or ''
GMAIL_APP_PASSWORD = CONFIG.get_api_key('gmail_app_password') or ''
ALERT_TO_EMAIL = CONFIG.get_api_key('alert_to_email') or CONFIG.get_api_key('gmail_email') or ''
ALERT_FROM_EMAIL = GMAIL_EMAIL

# Notification rate limiting
NOTIFICATION_BATCH_TIMEOUT = CONFIG.notifications.batch_timeout
NOTIFICATION_RATE_LIMIT = CONFIG.notifications.max_emails_per_minute
NOTIFICATION_COOLDOWN_MINUTES = CONFIG.notifications.cooldown_minutes
NOTIFICATION_DUPLICATE_WINDOW = CONFIG.notifications.duplicate_window_minutes * 60
NOTIFICATION_BATCH_SIZE = CONFIG.notifications.batch_size
SEND_EMAIL_NOTIFICATIONS = CONFIG.notifications.send_email_notifications
SEND_DAILY_SUMMARY = CONFIG.notifications.send_daily_summary
GMAIL_MAX_RATE_PER_MINUTE = CONFIG.notifications.max_emails_per_minute

# Performance settings
SHUTDOWN_TIMEOUT = 15
MAX_MEMORY_USAGE_MB = 2000
MAX_DATAFRAME_ROWS = 100000
ENABLE_PERFORMANCE_METRICS = True

# Biotech settings
BIOTECH_INDUSTRY_LIST = CONFIG.biotech_industries
MIN_PHASE3_SUCCESS_SCORE_THRESHOLD = 0.4

# Worker configuration with dynamic sizing
NUM_NEWS_EVENT_PROCESSOR_WORKERS = max(1, CONFIG.data.api_rate_limit_calls // 20)
MOMENTUM_TRACKER_NUM_WORKERS = 3

# Universe selection parameters
STOCK_SCREENER_LIMIT = 2000
EXCHANGE_LIST = CONFIG.exchange_list
PRICE_MORE_THAN = 2.0
PRICE_LESS_THAN = 500.0
MARKET_CAP_LOWER_THAN = 10_000_000_000  # 10B market cap limit (improved readability)
VOLUME_MORE_THAN = 50_000

# Momentum tracking parameters
MOMENTUM_MAX_TRACKING_MINS = 60
MOMENTUM_LOOKBACK_PERIOD = 5
PRICE_PERCENT_CHANGE_THRESHOLD = 0.02

# Model names
GEMINI_MODEL = "gemini-pro"

# Timeouts and retry settings
NEWS_PROCESSOR_TIMEOUT = 30
EVENT_TRACKER_RETRY_DELAY = 5
MAX_RETRY_ATTEMPTS = CONFIG.notifications.max_retry_attempts
DEFAULT_BATCH_SIZE = 100
CONNECTION_TIMEOUT = CONFIG.data.request_timeout

# Export validation function for external use
def validate_configuration() -> List[str]:
    """Public interface for configuration validation"""
    return CONFIG.validate_config()


# Module-level configuration validation on import
_config_errors = validate_configuration()
if _config_errors:
    import warnings
    warnings.warn(f"Configuration issues detected: {_config_errors}")