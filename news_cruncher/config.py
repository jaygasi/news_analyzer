"""
Configuration for simplified financial news analysis system
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
    
    # Emergency fallback APIs (only when LLM quotas exhausted)
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    TIINGO_API_KEY: str = os.getenv('TIINGO_API_KEY', '')
    
    # LLM Configuration
    LLM_PRIORITY: List[str] = ['finbert', 'gemini', 'openai', 'claude']
    EMERGENCY_FALLBACKS: List[str] = ['alpha_vantage', 'polygon', 'tiingo']
    
    # Analysis Thresholds
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    MIN_NEWS_LENGTH: int = 50
    MAX_NEWS_AGE_HOURS: int = 24
    
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
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        return {
            'fmp': bool(cls.FMP_API_KEY),
            'gemini': bool(cls.GEMINI_API_KEY),
            'openai': bool(cls.OPENAI_API_KEY),
            'anthropic': bool(cls.ANTHROPIC_API_KEY),
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_API_KEY),
            'polygon': bool(cls.POLYGON_API_KEY),
            'tiingo': bool(cls.TIINGO_API_KEY)
        }


# Create directories on import
Config.create_directories()