"""
Enhanced configuration with technical analysis parameters
"""
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Enhanced configuration with technical analysis settings"""
    
    # Directories
    log_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results")
    trade_log_path: Path = Path("trade_logs")
    trade_log_file: str = "enhanced_trade_log.csv"
    
    # Trading parameters
    position_size: float = 10000.0  # Base position size
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.10   # 10% take profit
    min_confidence_score: float = 0.7  # Minimum combined confidence
    
    # Technical analysis thresholds
    min_liquidity_score: float = 0.3      # Minimum liquidity requirement
    max_bid_ask_spread: float = 0.05      # 5% maximum spread
    min_volume_score: float = 0.2         # Minimum volume activity
    min_technical_confidence: float = 0.3 # Minimum technical confidence
    
    # Data collection intervals (seconds)
    news_check_interval: int = 30
    price_check_interval: int = 5
    
    # API settings
    api_rate_limit: int = 100
    request_timeout: int = 30
    
    # Universe selection - enhanced criteria
    max_symbols: int = 500
    min_price: float = 2.0
    max_price: float = 500.0
    min_volume: int = 100000  # Increased for better liquidity
    min_market_cap: int = 100_000_000  # $100M minimum
    
    # AI ensemble weights
    finbert_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # Position sizing factors
    min_position_multiplier: float = 0.5   # 50% minimum position
    max_position_multiplier: float = 1.5   # 150% maximum position
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for directory in [self.log_dir, self.cache_dir, self.results_dir, self.trade_log_path]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables"""
        key = os.getenv(f"{provider.upper()}_API_KEY")
        return key.strip() if key else None

# Global config instance
CONFIG = Config()

# Ensure directories exist
CONFIG.__post_init__()