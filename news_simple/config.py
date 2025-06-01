"""
Enhanced configuration with better Gemini model and rate limiting
"""
import os
from dataclasses import dataclass, field
from typing import Optional
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
    # Set to 0.5 to see more trading activity for debugging
    # Change back to 0.7 once you confirm the system works
    min_confidence_score: float = 0.5  # Was 0.7 - temporarily lowered for testing
    
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
    
    # TESTING MODE - Set to True to bypass market hours for testing
    testing_mode: bool = False  # Set to True for testing after market hours
    
    # Universe selection - enhanced criteria
    max_symbols: int = 500
    min_price: float = 2.0
    max_price: float = 500.0
    min_volume: int = 100000
    min_market_cap: int = 100_000_000
    
    # AI ensemble weights - Adjusted for better Gemini fallback
    finbert_weight: float = 0.5          # Increased when Gemini fails
    keyword_weight: float = 0.3          # Increased when Gemini fails  
    gemini_weight: float = 0.2           # Reduced weight due to rate limits
    
    # Position sizing factors
    min_position_multiplier: float = 0.5
    max_position_multiplier: float = 1.5
    
    def __post_init__(self) -> None:
        """Create directories and validate configuration."""
        self._create_directories()
        self._validate_parameters()
    
    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        directories = [self.log_dir, self.cache_dir, self.results_dir, self.trade_log_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
        if not (0 < self.stop_loss_pct < 1):
            raise ValueError("Stop loss percentage must be between 0 and 1")
        if not (0 < self.take_profit_pct < 1):
            raise ValueError("Take profit percentage must be between 0 and 1")
        if not (0 <= self.min_confidence_score <= 1):
            raise ValueError("Minimum confidence score must be between 0 and 1")
        
        # Validate ensemble weights sum close to 1.0
        total_weight = self.finbert_weight + self.keyword_weight + self.gemini_weight
        if abs(total_weight - 1.0) > 0.1:
            raise ValueError(f"AI ensemble weights should sum to 1.0, got {total_weight}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment variables with validation."""
        if not provider:
            return None
        
        # Map provider names to environment variable names
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
        
        # BETTER DEFAULT: Use Flash for much better rate limits
        # Flash: 15 RPM, 1M TPM, 1500 RPD (FREE)
        # vs Pro: 2 RPM, 32K TPM, 50 RPD (FREE)
        return 'gemini-1.5-flash'  # Much better rate limits!


# Global config instance
CONFIG = Config()