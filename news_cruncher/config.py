"""
Configuration for simplified financial news analysis system with service toggles
Python 3.13.3 compatible with type hints and modern features
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the financial news analysis system"""
    
    # API Keys
    FMP_API_KEY: str = os.getenv('FMP_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Emergency fallback APIs
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    TIINGO_API_KEY: str = os.getenv('TIINGO_API_KEY', '')
    
    # Service Enable/Disable Toggles (based on your latest run results)
    ENABLE_FINBERT: bool = os.getenv('ENABLE_FINBERT', 'true').lower() == 'true'
    ENABLE_GEMINI: bool = os.getenv('ENABLE_GEMINI', 'true').lower() == 'true'
    ENABLE_OPENAI: bool = os.getenv('ENABLE_OPENAI', 'false').lower() == 'true'  # Disabled due to quota
    ENABLE_CLAUDE: bool = os.getenv('ENABLE_CLAUDE', 'false').lower() == 'true'  # Disabled due to invalid key
    ENABLE_ALPHA_VANTAGE: bool = os.getenv('ENABLE_ALPHA_VANTAGE', 'true').lower() == 'true'
    ENABLE_POLYGON: bool = os.getenv('ENABLE_POLYGON', 'true').lower() == 'true'
    ENABLE_TIINGO: bool = os.getenv('ENABLE_TIINGO', 'false').lower() == 'true'  # Disabled due to 403 error
    ENABLE_KEYWORD_ANALYSIS: bool = os.getenv('ENABLE_KEYWORD_ANALYSIS', 'true').lower() == 'true'
    
    # LLM Configuration
    LLM_PRIORITY: List[str] = ['finbert', 'gemini', 'openai', 'claude']
    EMERGENCY_FALLBACKS: List[str] = ['alpha_vantage', 'polygon', 'tiingo']
    
    # Analysis Thresholds - LOWERED for better CSV logging
    MIN_CONFIDENCE_THRESHOLD: float = 0.6  # Lowered from 0.7
    MIN_NEWS_LENGTH: int = 50
    
    # Dynamic News Age Configuration
    DEFAULT_NEWS_AGE_HOURS: int = 24  # Used for first run only
    MAX_NEWS_AGE_HOURS: int = 72     # Maximum lookback even if last run was longer ago
    MIN_NEWS_AGE_MINUTES: int = 5    # Minimum gap to prevent too-frequent fetching
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data'
    OUTPUT_DIR: Path = BASE_DIR / 'output'
    
    CSV_OUTPUT_PATH: Path = OUTPUT_DIR / 'trading_decisions.csv'
    SQLITE_DB_PATH: Path = DATA_DIR / 'processed_articles.db'
    LOG_FILE_PATH: Path = BASE_DIR / 'application.log'
    
    # Database Settings
    DB_TIMEOUT: int = 30
    
    # API Rate Limiting
    FMP_REQUESTS_PER_MINUTE: int = 300
    LLM_REQUEST_DELAY: float = 1.0
    
    # News Fetching
    MAX_NEWS_ARTICLES: int = 1000
    NEWS_SOURCES: List[str] = [
        'stock_news',
        'press-releases', 
        'earnings-call-transcript',
        'analyst-estimates'
    ]
    
    @classmethod
    def get_enabled_llm_services(cls) -> List[str]:
        """Get list of enabled LLM services"""
        enabled = []
        if cls.ENABLE_FINBERT and cls.has_finbert_dependencies():
            enabled.append('finbert')
        if cls.ENABLE_GEMINI and cls.GEMINI_API_KEY:
            enabled.append('gemini')
        if cls.ENABLE_OPENAI and cls.OPENAI_API_KEY:
            enabled.append('openai')
        if cls.ENABLE_CLAUDE and cls.ANTHROPIC_API_KEY:
            enabled.append('claude')
        return enabled
    
    @classmethod
    def get_enabled_emergency_services(cls) -> List[str]:
        """Get list of enabled emergency services"""
        enabled = []
        if cls.ENABLE_ALPHA_VANTAGE and cls.ALPHA_VANTAGE_API_KEY:
            enabled.append('alpha_vantage')
        if cls.ENABLE_POLYGON and cls.POLYGON_API_KEY:
            enabled.append('polygon')
        if cls.ENABLE_TIINGO and cls.TIINGO_API_KEY:
            enabled.append('tiingo')
        return enabled
    
    @classmethod
    def has_finbert_dependencies(cls) -> bool:
        """Check if FinBERT dependencies are available"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present and enabled"""
        return {
            'fmp': bool(cls.FMP_API_KEY),
            'gemini': bool(cls.GEMINI_API_KEY) and cls.ENABLE_GEMINI,
            'openai': bool(cls.OPENAI_API_KEY) and cls.ENABLE_OPENAI,
            'anthropic': bool(cls.ANTHROPIC_API_KEY) and cls.ENABLE_CLAUDE,
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_API_KEY) and cls.ENABLE_ALPHA_VANTAGE,
            'polygon': bool(cls.POLYGON_API_KEY) and cls.ENABLE_POLYGON,
            'tiingo': bool(cls.TIINGO_API_KEY) and cls.ENABLE_TIINGO
        }


# Create directories on import
Config.create_directories()