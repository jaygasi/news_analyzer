"""
Enhanced configuration with better Gemini model and rate limiting
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Enhanced configuration with comprehensive type hints and multi-API support."""
    
    # Directories
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    trade_log_path: Path = field(default_factory=lambda: Path("trade_logs"))
    trade_log_file: str = "enhanced_trade_log.csv"
    
    # Trading parameters
    position_size: float = 10000.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    
    # CONFIDENCE THRESHOLD - LOWER FOR TESTING
    min_confidence_score: float = 0.5
    
    # Technical analysis thresholds
    min_liquidity_score: float = 0.3
    max_bid_ask_spread: float = 0.05
    min_volume_score: float = 0.2
    min_technical_confidence: float = 0.3
    
    # Data collection intervals (seconds)
    news_check_interval: int = 30
    price_check_interval: int = 5
    
    # API settings
    api_rate_limit: int = 100
    request_timeout: int = 30
    
    # TESTING MODE
    testing_mode: bool = False
    
    # Universe selection
    max_symbols: int = 500
    min_price: float = 2.0
    max_price: float = 500.0
    min_volume: int = 100000
    min_market_cap: int = 100_000_000
    
    # AI ensemble weights
    finbert_weight: float = 0.5
    keyword_weight: float = 0.3
    gemini_weight: float = 0.2
    
    # Position sizing factors
    min_position_multiplier: float = 0.5
    max_position_multiplier: float = 1.5
    
    # Enhanced sentiment analysis settings
    enable_enhanced_sentiment: bool = True
    min_quality_confidence: float = 0.7  # Higher bar for quality
    enable_multi_source_confirmation: bool = True

    # Enhanced filtering
    min_magnitude_score: float = 0.4
    min_credibility_score: float = 0.6
    
    def __post_init__(self) -> None:
        """Create directories and validate configuration."""
        self._create_directories()
        self._validate_parameters()
    
    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for directory in [self.log_dir, self.cache_dir, self.results_dir, self.trade_log_path]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        validations = [
            (self.position_size > 0, "Position size must be positive"),
            (0 < self.stop_loss_pct < 1, "Stop loss percentage must be between 0 and 1"),
            (0 < self.take_profit_pct < 1, "Take profit percentage must be between 0 and 1"),
            (0 <= self.min_confidence_score <= 1, "Minimum confidence score must be between 0 and 1")
        ]
        
        for condition, error_msg in validations:
            if not condition:
                raise ValueError(error_msg)
        
        # Validate ensemble weights
        total_weight = self.finbert_weight + self.keyword_weight + self.gemini_weight
        if abs(total_weight - 1.0) > 0.1:
            raise ValueError(f"AI ensemble weights should sum to 1.0, got {total_weight}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables with validation."""
        if not provider:
            return None
        
        provider_map = {
            'fmp': 'FMP_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'polygon': 'POLYGON_API_KEY',
            'tiingo': 'TIINGO_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_API_KEY'
        }
        
        env_var = provider_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
        key = os.getenv(env_var)
        return key.strip() if key else None
    
    def get_gemini_model(self) -> str:
        """Get Gemini model from environment variable with better fallback."""
        model = os.getenv('GEMINI_MODEL')
        if model and model.strip():
            return model.strip()
        
        return 'gemini-1.5-flash'  # Better rate limits than Pro


# Global config instance
CONFIG = Config()