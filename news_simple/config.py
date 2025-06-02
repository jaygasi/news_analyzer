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
    
    # CONFIDENCE THRESHOLD - OPTIMIZED FOR BETTER SIGNAL QUALITY
    min_confidence_score: float = 0.45
    
    # Technical analysis thresholds
    min_liquidity_score: float = 0.25
    max_bid_ask_spread: float = 0.06
    min_volume_score: float = 0.15
    min_technical_confidence: float = 0.25
    
    # Data collection intervals (seconds) - optimized
    news_check_interval: int = 25
    price_check_interval: int = 4
    
    # API settings
    api_rate_limit: int = 120
    request_timeout: int = 25
    
    # TESTING MODE
    testing_mode: bool = False
    
    # Universe selection - optimized
    max_symbols: int = 400
    min_price: float = 1.5
    max_price: float = 600.0
    min_volume: int = 75000
    min_market_cap: int = 75_000_000
    
    # AI ensemble weights - rebalanced for better performance
    finbert_weight: float = 0.45
    keyword_weight: float = 0.35
    gemini_weight: float = 0.20
    
    # Position sizing factors
    min_position_multiplier: float = 0.4
    max_position_multiplier: float = 1.8
    
    # Enhanced sentiment analysis settings
    enable_enhanced_sentiment: bool = True
    min_quality_confidence: float = 0.65
    enable_multi_source_confirmation: bool = True

    # Enhanced filtering - more permissive for better coverage
    min_magnitude_score: float = 0.35
    min_credibility_score: float = 0.55
    
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
            self.finbert_weight = 0.45
            self.keyword_weight = 0.35
            self.gemini_weight = 0.20
    
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
        """Get Gemini model from environment with fallback to optimized default."""
        model = os.getenv('GEMINI_MODEL')
        if model and model.strip():
            model_name = model.strip()
            # Validate model name format
            valid_models = [
                'gemini-1.5-flash', 'gemini-1.5-pro', 
                'gemini-pro', 'gemini-flash'
            ]
            if any(valid in model_name for valid in valid_models):
                return model_name
        
        return 'gemini-1.5-flash'  # Safe default with good rate limits


# Global config instance
CONFIG = Config()