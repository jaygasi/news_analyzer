"""
Enhanced configuration with improved validation and type safety
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()


@dataclass
class Config:
    """Enhanced configuration with comprehensive type hints and validation."""
    
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
    min_confidence_score: float = 0.45
    
    # Technical analysis thresholds
    min_liquidity_score: float = 0.25
    max_bid_ask_spread: float = 0.06
    min_volume_score: float = 0.15
    min_technical_confidence: float = 0.25
    
    # System intervals (seconds)
    news_check_interval: int = 25
    price_check_interval: int = 4
    
    # News fetching configuration
    news_page_limit: int = 5
    news_per_page_limit: int = 100
    max_total_news_articles: int = 500
    
    # API settings
    api_rate_limit: int = 120
    request_timeout: int = 25
    testing_mode: bool = False
    
    # Universe selection
    max_symbols: int = 400
    min_price: float = 1.5
    max_price: float = 600.0
    min_volume: int = 75000
    min_market_cap: int = 75_000_000
    
    # AI ensemble weights
    finbert_weight: float = 0.45
    keyword_weight: float = 0.35
    gemini_weight: float = 0.20
    
    # Position sizing
    min_position_multiplier: float = 0.4
    max_position_multiplier: float = 1.8
    
    # Enhanced sentiment analysis
    enable_enhanced_sentiment: bool = True
    min_quality_confidence: float = 0.65
    enable_multi_source_confirmation: bool = True
    min_magnitude_score: float = 0.35
    min_credibility_score: float = 0.55

    def __post_init__(self) -> None:
        """Validate configuration and create directories."""
        self._create_directories()
        self._validate_parameters()
        self._normalize_weights()
    
    def _create_directories(self) -> None:
        """Create required directories."""
        directories = [self.log_dir, self.cache_dir, self.results_dir, self.trade_log_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        validations = [
            (self.position_size > 0, "Position size must be positive"),
            (0 < self.stop_loss_pct < 1, "Stop loss must be between 0 and 1"),
            (0 < self.take_profit_pct < 1, "Take profit must be between 0 and 1"),
            (0 <= self.min_confidence_score <= 1, "Confidence score must be between 0 and 1"),
            (self.news_page_limit > 0, "News page limit must be positive"),
            (self.min_price > 0, "Minimum price must be positive"),
            (self.max_price > self.min_price, "Maximum price must exceed minimum price"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(message)
    
    def _normalize_weights(self) -> None:
        """Normalize ensemble weights to sum to 1.0."""
        total_weight = self.finbert_weight + self.keyword_weight + self.gemini_weight
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.finbert_weight /= total_weight
            self.keyword_weight /= total_weight
            self.gemini_weight /= total_weight
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specified provider."""
        if not provider:
            return None
        
        provider_mapping = {
            'fmp': 'FMP_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'polygon': 'POLYGON_API_KEY',
            'tiingo': 'TIINGO_API_KEY',
            'alpha_vantage': 'ALPHA_VANTAGE_API_KEY'
        }
        
        env_var = provider_mapping.get(provider.lower(), f"{provider.upper()}_API_KEY")
        key = os.getenv(env_var)
        return key.strip() if key else None
    
    def get_gemini_model(self) -> str:
        """Get Gemini model with validation and fallback."""
        model = os.getenv('GEMINI_MODEL', '').strip()
        
        if model:
            valid_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-flash']
            if any(valid_model in model for valid_model in valid_models):
                return model
        
        return 'gemini-1.5-flash'  # Safe default


# Global config instance
CONFIG = Config()