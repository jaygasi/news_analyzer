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
    min_confidence_score: float = 0.40  # Reduced from 0.45 to allow more trades
    
    # Technical analysis thresholds
    min_liquidity_score: float = 0.20  # Reduced from 0.25
    max_bid_ask_spread: float = 0.08  # Increased from 0.06
    min_volume_score: float = 0.10  # Reduced from 0.15
    min_technical_confidence: float = 0.20  # Reduced from 0.25
    
    # System intervals (seconds) - Optimized for better news flow
    news_check_interval: int = 25  # Reduced from 30 for faster news processing
    price_check_interval: int = 5
    
    # News fetching configuration - Enhanced for comprehensive coverage
    news_page_limit: int = 10  # Increased from 8 for more comprehensive coverage
    news_per_page_limit: int = 40  # Reduced from 50 to balance API limits
    max_total_news_articles: int = 1000  # Increased from 800
    
    # API settings
    api_rate_limit: int = 120
    request_timeout: int = 25
    testing_mode: bool = False
    
    # Universe selection - Optimized for news coverage
    max_symbols: int = 1000  # Increased from 800 for better coverage
    min_price: float = 0.50  # Reduced from 1.0 for more coverage
    max_price: float = 1500.0  # Increased from 1000
    min_volume: int = 30000  # Reduced from 50000 for more symbols
    min_market_cap: int = 25_000_000  # Reduced from 50M for broader coverage
    
    # AI ensemble weights
    finbert_weight: float = 0.45
    keyword_weight: float = 0.35
    gemini_weight: float = 0.20
    
    # Position sizing
    min_position_multiplier: float = 0.3  # Reduced from 0.4
    max_position_multiplier: float = 2.0  # Increased from 1.8
    
    # Enhanced sentiment analysis - More lenient for testing
    enable_enhanced_sentiment: bool = True
    min_quality_confidence: float = 0.55  # Reduced from 0.65
    enable_multi_source_confirmation: bool = True
    min_magnitude_score: float = 0.25  # Reduced from 0.35
    min_credibility_score: float = 0.45  # Reduced from 0.55

    def __post_init__(self) -> None:
        """Validate configuration and create directories."""
        self._create_directories()
        self._validate_parameters()
        self._normalize_weights()
        self._apply_testing_mode_adjustments()
    
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
    
    def _apply_testing_mode_adjustments(self) -> None:
        """Apply testing mode adjustments if enabled."""
        # Check environment variable for testing mode
        if os.getenv('TESTING_MODE', '').lower() in ('true', '1', 'yes'):
            self.testing_mode = True
            
            # More lenient parameters for testing
            self.min_confidence_score = max(0.30, self.min_confidence_score - 0.10)
            self.min_liquidity_score = max(0.10, self.min_liquidity_score - 0.10)
            self.min_technical_confidence = max(0.10, self.min_technical_confidence - 0.10)
            self.news_check_interval = max(15, self.news_check_interval - 10)
            
            print(f"[CONFIG] Testing mode enabled - adjusted thresholds for development")
    
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