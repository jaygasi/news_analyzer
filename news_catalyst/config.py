"""
Optimized configuration module with better type safety and validation
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
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
    """Immutable directory configuration"""
    log_dir: str = "logs"
    cache_dir: str = "cache"
    results_dir: str = "results"
    plots_dir: str = "plots"
    trade_log_path: str = "trade_logs"
    trade_log_file_name: str = "trade_log.csv"


@dataclass(frozen=True)
class DataConfig:
    """Data collection and processing configuration"""
    # Real-time data intervals (seconds)
    price_tick_interval: float = 3.0  # 3 seconds
    news_check_interval: float = 30.0  # 30 seconds
    momentum_calc_interval: float = 0.5  # 500ms
    
    # API rate limits
    api_rate_limit_calls: int = 100
    api_rate_limit_period: int = 60
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # Data providers with priorities
    primary_provider: DataProvider = DataProvider.POLYGON
    backup_providers: List[DataProvider] = None
    
    def __post_init__(self):
        if self.backup_providers is None:
            object.__setattr__(self, 'backup_providers', [DataProvider.IEX, DataProvider.FMP])


@dataclass(frozen=True)
class AIConfig:
    """AI and analysis configuration"""
    # Model ensemble weights
    finbert_weight: float = 0.4
    openai_weight: float = 0.3
    claude_weight: float = 0.2
    local_llm_weight: float = 0.1
    
    # Analysis thresholds
    min_confidence_threshold: float = 0.7
    consensus_threshold: float = 0.8
    
    # Feature flags
    use_openai: bool = False
    use_gemini: bool = False
    use_ollama: bool = False
    use_ensemble: bool = True


@dataclass(frozen=True)
class TradingConfig:
    """Trading and risk management configuration"""
    # Position sizing
    max_position_size: float = 0.05  # 5% of portfolio
    max_portfolio_risk: float = 0.02  # 2% daily risk
    max_sector_exposure: float = 0.30  # 30% in one sector
    max_correlation: float = 0.7
    
    # Execution
    max_bid_ask_spread: float = 0.05
    stop_loss_atr_multiplier: float = 2.0
    min_volume_threshold: int = 10000
    
    # Dynamic thresholds
    base_momentum_threshold: float = 0.01
    volatility_adjustment: bool = True
    regime_adjustment: bool = True


@dataclass(frozen=True)
class NotificationConfig:
    """Notification system configuration"""
    enable_notifications: bool = True
    use_gmail: bool = True
    use_sendgrid: bool = False
    
    # Rate limiting
    max_emails_per_minute: int = 25
    batch_size: int = 5
    batch_timeout: float = 3.0
    
    # Retry logic
    max_retry_attempts: int = 3
    retry_delay: float = 2.0
    cooldown_minutes: int = 1


class OptimizedConfig:
    """Centralized configuration with validation and environment support"""
    
    def __init__(self, env: Environment = Environment.PRODUCTION):
        self.env = env
        self.directories = DirectoryConfig()
        self.data = DataConfig()
        self.ai = AIConfig()
        self.trading = TradingConfig()
        self.notifications = NotificationConfig()
        
        # Environment-specific overrides
        if env == Environment.DEVELOPMENT:
            self._apply_dev_overrides()
        elif env == Environment.TESTING:
            self._apply_test_overrides()
    
    def _apply_dev_overrides(self):
        """Apply development environment settings"""
        object.__setattr__(self.data, 'api_rate_limit_calls', 10)
        object.__setattr__(self.notifications, 'enable_notifications', False)
    
    def _apply_test_overrides(self):
        """Apply testing environment settings"""
        object.__setattr__(self.trading, 'max_position_size', 0.01)
        object.__setattr__(self.notifications, 'enable_notifications', False)
    
    @property
    def exchange_list(self) -> str:
        """Supported exchanges"""
        return "nyse,nasdaq,amex"
    
    @property
    def biotech_industries(self) -> List[str]:
        """Biotech industry classifications"""
        return ['Biotechnology', 'Medical - Diagnostics & Research', 'Pharmaceuticals']
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Securely retrieve API keys from environment"""
        env_var = f"{provider.upper()}_API_KEY"
        return os.getenv(env_var)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        if self.trading.max_position_size > 0.2:
            errors.append("Position size too large (>20%)")
        
        if self.data.price_tick_interval < 0.05:
            errors.append("Price tick interval too aggressive")
        
        if not self.get_api_key('FMP') and self.data.primary_provider == DataProvider.FMP:
            errors.append("Missing FMP API key")
        
        return errors


# Global configuration instance
CONFIG = OptimizedConfig()

# Backward compatibility - expose commonly used values
LOG_DIR = CONFIG.directories.log_dir
CACHE_DIR = CONFIG.directories.cache_dir
RESULTS_DIR = CONFIG.directories.results_dir
TRADE_LOG_PATH = CONFIG.directories.trade_log_path
TRADE_LOG_FILE_NAME = CONFIG.directories.trade_log_file_name

# Trading parameters
MAX_BID_ASK_SPREAD = CONFIG.trading.max_bid_ask_spread
MAX_POSITION_SIZE = CONFIG.trading.max_position_size
STOP_LOSS_PERCENT = 0.02  # Will be replaced by dynamic ATR-based stops

# AI parameters
USE_OPENAI_ANALYSIS = CONFIG.ai.use_openai
MIN_NEWS_TOPIC_CONFIDENCE = 1
MAX_NEWS_ARTICLE_AGE_MINS = 240.0  # Increased to 4 hours

# Data collection intervals
NEWS_DATA_COLLECTION_INTERVAL = int(CONFIG.data.news_check_interval)
PRICE_TRACKER_DATA_COLLECTION_INTERVAL = int(CONFIG.data.price_tick_interval * 10)
MOMENTUM_TRACKING_TIME_INTERVAL = int(CONFIG.data.momentum_calc_interval * 10)

# Notification settings
ENABLE_NOTIFICATIONS = CONFIG.notifications.enable_notifications
USE_GMAIL_NOTIFICATIONS = CONFIG.notifications.use_gmail
GMAIL_EMAIL = os.getenv('GMAIL_EMAIL')
GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
ALERT_TO_EMAIL = os.getenv('ALERT_TO_EMAIL')
ALERT_FROM_EMAIL = os.getenv('GMAIL_EMAIL')

# Performance settings
SHUTDOWN_TIMEOUT = 15
MAX_MEMORY_USAGE_MB = 2000  # Increased from 1000MB
MAX_DATAFRAME_ROWS = 100000  # Increased capacity
ENABLE_PERFORMANCE_METRICS = True

# Biotech settings
BIOTECH_INDUSTRY_LIST = CONFIG.biotech_industries
MIN_PHASE3_SUCCESS_SCORE_THRESHOLD = 0.4

# Worker configuration
NUM_NEWS_EVENT_PROCESSOR_WORKERS = 3  # Increased from 2
MOMENTUM_TRACKER_NUM_WORKERS = 3  # Increased from 2

# Universe selection
STOCK_SCREENER_LIMIT = 2000  # Increased from 1000
EXCHANGE_LIST = CONFIG.exchange_list
PRICE_MORE_THAN = 2.0  # Increased minimum price
PRICE_LESS_THAN = 500.0  # Decreased maximum price for better liquidity
MARKET_CAP_LOWER_THAN = 10000000000  # 10B market cap limit
VOLUME_MORE_THAN = 50000  # Increased minimum volume