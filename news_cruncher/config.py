"""
Configuration for simplified financial news analysis system with service toggles
Python 3.13.3 compatible with type hints and modern features

🎯 CONFIGURATION GUIDE:
This file controls all major aspects of the financial news analysis system.
Each setting below includes detailed explanations of its impact on system behavior.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Configuration class for the financial news analysis system
    
    💡 TIP: Most settings can be overridden via environment variables
    Example: Set ENABLE_FINBERT=false in .env to disable FinBERT
    """
    
    # ================================================================
    # 🔑 API KEYS - Required for various services
    # ================================================================
    
    # 🚨 REQUIRED: FMP (Financial Modeling Prep) API Key
    # EFFECT: Without this, the entire system cannot fetch news data
    # GET KEY: https://financialmodelingprep.com/developer/docs
    FMP_API_KEY: str = os.getenv('FMP_API_KEY', '')
    
    # 🤖 AI/LLM Service API Keys (Optional but recommended for best results)
    # EFFECT: More services enabled = more robust predictions via consensus
    
    # Google Gemini API Key
    # EFFECT: Adds Google's LLM analysis (33% weight in multi-source)
    # COST: Free tier: 15 requests/minute, 1500 requests/day
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # OpenAI API Key  
    # EFFECT: Adds GPT analysis (20% weight in multi-source)
    # COST: Pay-per-use, can be expensive with high volume
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    # Anthropic Claude API Key
    # EFFECT: Adds Claude analysis (15% weight in multi-source)  
    # COST: Pay-per-use, good quality but limited free tier
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    # 🚨 Emergency fallback APIs (Optional but recommended for robustness)
    # EFFECT: Provide backup analysis when primary LLMs fail or hit quotas
    
    # Alpha Vantage API Key
    # EFFECT: Adds news sentiment analysis (13% weight)
    # COST: Free tier: 5 calls/minute, 500 calls/day
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    # Polygon API Key
    # EFFECT: Adds news analysis via Polygon's news API (10% weight)
    # COST: Free tier: 5 calls/minute, quite restrictive
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    
    # Tiingo API Key
    # EFFECT: Adds news analysis via Tiingo's API (4% weight)
    # COST: Free tier available, good for backup
    TIINGO_API_KEY: str = os.getenv('TIINGO_API_KEY', '')
    
    # ================================================================
    # 🔧 SERVICE ENABLE/DISABLE TOGGLES
    # ================================================================
    
    # 🧠 FinBERT (Financial BERT) - Local ML Model
    # EFFECT: Provides specialized financial sentiment analysis (41% weight)
    # REQUIREMENT: Requires PyTorch and Transformers libraries
    # PERFORMANCE: CPU-based, adds ~2-3 seconds per analysis
    # RECOMMENDATION: Keep enabled - it's free and finance-specific
    ENABLE_FINBERT: bool = os.getenv('ENABLE_FINBERT', 'true').lower() == 'true'
    
    # 🌟 Google Gemini - High-quality LLM
    # EFFECT: Provides sophisticated language understanding (33% weight)
    # RATE LIMITS: 15 requests/minute, 1500/day on free tier
    # RECOMMENDATION: Keep enabled - excellent free tier
    ENABLE_GEMINI: bool = os.getenv('ENABLE_GEMINI', 'true').lower() == 'true'
    
    # 💰 OpenAI GPT Models
    # EFFECT: High-quality analysis but can be expensive (20% weight)
    # RATE LIMITS: Depends on your billing tier
    # RECOMMENDATION: Disable if you're cost-conscious, enable for best quality
    ENABLE_OPENAI: bool = os.getenv('ENABLE_OPENAI', 'false').lower() == 'true'  # Disabled due to quota
    
    # 🎭 Anthropic Claude
    # EFFECT: Good analysis quality, different perspective (15% weight)
    # RATE LIMITS: Limited free tier, pay-per-use
    # RECOMMENDATION: Enable if you have credits, otherwise keep disabled
    ENABLE_CLAUDE: bool = os.getenv('ENABLE_CLAUDE', 'false').lower() == 'true'  # Disabled due to invalid key
    
    # 📊 Alpha Vantage News Sentiment
    # EFFECT: Provides real market sentiment data (13% weight)
    # RATE LIMITS: 5 calls/minute, 500/day free
    # RECOMMENDATION: Enable for additional market context
    ENABLE_ALPHA_VANTAGE: bool = os.getenv('ENABLE_ALPHA_VANTAGE', 'true').lower() == 'true'
    
    # 📈 Polygon News Analysis
    # EFFECT: Market data company's news analysis (10% weight)
    # RATE LIMITS: 5 calls/minute on free tier (very restrictive)
    # RECOMMENDATION: Enable if you have paid plan, otherwise expect frequent limits
    ENABLE_POLYGON: bool = os.getenv('ENABLE_POLYGON', 'true').lower() == 'true'
    
    # 📰 Tiingo News Analysis
    # EFFECT: Financial data provider's news analysis (4% weight)
    # RATE LIMITS: More generous free tier than Polygon
    # RECOMMENDATION: Good backup option, enable if other services fail
    ENABLE_TIINGO: bool = os.getenv('ENABLE_TIINGO', 'false').lower() == 'true'  # Disabled due to 403 error
    
    # 🔤 Enhanced Keyword Analysis
    # EFFECT: Fallback analysis using 80+ financial keywords (3% weight)
    # PERFORMANCE: Instant, no API calls required
    # RECOMMENDATION: Always keep enabled - it's free and provides baseline analysis
    ENABLE_KEYWORD_ANALYSIS: bool = os.getenv('ENABLE_KEYWORD_ANALYSIS', 'true').lower() == 'true'
    
    # ================================================================
    # 🎛️ ANALYSIS CONFIGURATION
    # ================================================================
    
    # 🎯 LLM Service Priority Order
    # EFFECT: Order in which services are attempted (first = highest priority)
    # MODIFY: Reorder based on your preferred services or API reliability
    LLM_PRIORITY: List[str] = ['finbert', 'gemini', 'openai', 'claude']
    
    # 🚨 Emergency Service Priority Order  
    # EFFECT: Fallback order when primary LLMs fail
    # MODIFY: Reorder based on your API quotas and reliability
    EMERGENCY_FALLBACKS: List[str] = ['alpha_vantage', 'polygon', 'tiingo']
    
    # ================================================================
    # ⚖️ DECISION THRESHOLDS - Critical for CSV output volume
    # ================================================================
    
    # 🎯 Minimum Confidence for CSV Logging
    # CURRENT: 0.6 (60%)
    # EFFECT: Higher = fewer but higher-quality decisions logged
    #         Lower = more decisions logged but potentially lower quality
    # RANGE: 0.1-0.9 recommended
    # EXAMPLE: 0.8 = very conservative (only highest confidence)
    #          0.4 = more liberal (more trading signals)
    MIN_CONFIDENCE_THRESHOLD: float = 0.6  # Lowered from 0.7
    
    # 📰 Minimum News Article Length
    # CURRENT: 50 characters
    # EFFECT: Filters out very short/low-quality articles
    # RANGE: 20-200 characters
    # EXAMPLE: 100 = stricter quality filtering
    #          20 = accept more brief articles
    MIN_NEWS_LENGTH: int = 50
    
    # ================================================================
    # ⏰ TIME CONFIGURATION - Controls how far back to look for news
    # ================================================================
    
    # 🏁 Default News Age for First Run
    # CURRENT: 24 hours
    # EFFECT: On first run, how far back to look for news
    # RANGE: 1-48 hours recommended
    # EXAMPLE: 6 = only very recent news on startup
    #          48 = capture more historical context
    DEFAULT_NEWS_AGE_HOURS: int = 24  # Used for first run only
    
    # ⏰ Maximum News Age (Safety Cap)
    # CURRENT: 72 hours (3 days)
    # EFFECT: Even if system was down for weeks, never look back more than this
    # RANGE: 24-168 hours (1-7 days)
    # EXAMPLE: 24 = always fresh news only
    #          168 = willing to process week-old news after downtime
    MAX_NEWS_AGE_HOURS: int = 72     # Maximum lookback even if last run was longer ago
    
    # ⚡ Minimum Gap Between Runs
    # CURRENT: 5 minutes
    # EFFECT: Prevents too-frequent processing (API rate limiting protection)
    # RANGE: 1-60 minutes
    # EXAMPLE: 1 = very frequent updates (higher API usage)
    #          30 = conservative, less frequent updates
    MIN_NEWS_AGE_MINUTES: int = 5    # Minimum gap to prevent too-frequent fetching
    
    # ================================================================
    # 📁 FILE PATHS - Where data is stored
    # ================================================================
    
    # 🏠 Base Directory (Auto-detected)
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / 'data'
    OUTPUT_DIR: Path = BASE_DIR / 'output'
    
    # 📋 CSV Output File
    # EFFECT: Where trading decisions are logged
    # MODIFY: Change filename if you want multiple output files
    CSV_OUTPUT_PATH: Path = OUTPUT_DIR / 'trading_decisions.csv'
    
    # 💾 SQLite Database File
    # EFFECT: Tracks processed articles to prevent reprocessing
    # WARNING: Deleting this file will cause all articles to be reprocessed
    SQLITE_DB_PATH: Path = DATA_DIR / 'processed_articles.db'
    
    # 📝 Application Log File
    # EFFECT: Where system logs are written
    # MODIFY: Change for different log file names
    LOG_FILE_PATH: Path = BASE_DIR / 'application.log'
    
    # ================================================================
    # 🔧 PERFORMANCE SETTINGS
    # ================================================================
    
    # ⏱️ Database Timeout
    # CURRENT: 30 seconds
    # EFFECT: How long to wait for database operations
    # RANGE: 10-60 seconds
    # EXAMPLE: 60 = more patient with slow disks
    #          10 = fail faster on database issues
    DB_TIMEOUT: int = 30
    
    # 🚀 FMP API Rate Limiting
    # CURRENT: 300 requests per minute
    # EFFECT: Controls how fast we call FMP API
    # WARNING: Exceeding your plan's limits will cause API failures
    # TYPICAL LIMITS: Free=250/month, Starter=1000/month, Professional=10000/month
    FMP_REQUESTS_PER_MINUTE: int = 300
    
    # ⏳ LLM Request Delay
    # CURRENT: 1.0 seconds between LLM calls
    # EFFECT: Prevents hitting rate limits on AI services
    # RANGE: 0.1-5.0 seconds
    # EXAMPLE: 0.5 = faster processing, higher risk of rate limits
    #          2.0 = safer for free tiers, slower processing
    LLM_REQUEST_DELAY: float = 1.0
    
    # ================================================================
    # 📊 DATA VOLUME CONTROLS - Critical for performance
    # ================================================================
    
    # 📈 Maximum Articles Per Cycle
    # CURRENT: 1000 articles
    # EFFECT: Caps processing to prevent overwhelming the system
    # PERFORMANCE: 1000 articles ≈ 20-30 minutes processing time
    # RANGE: 100-2000 recommended
    # EXAMPLE: 500 = faster cycles, might miss some news
    #          2000 = comprehensive but very slow cycles
    MAX_NEWS_ARTICLES: int = 1000
    
    # 📰 News Sources to Fetch
    # CURRENT: Stock news, press releases, earnings, analyst estimates
    # EFFECT: Which FMP endpoints to query for news
    # MODIFY: Comment out sources you don't want
    # WARNING: Removing all sources will break the system
    NEWS_SOURCES: List[str] = [
        'stock_news',                    # General stock news (usually 20-50 articles)
        'press-releases',                # Company press releases (usually 100+ articles)  
        'earnings-call-transcript',      # Earnings transcripts (usually 10-30 articles)
        'analyst-estimates'              # Analyst reports (usually 20-50 articles)
    ]
    
    # ================================================================
    # 🛠️ UTILITY METHODS
    # ================================================================
    
    @classmethod
    def get_enabled_llm_services(cls) -> List[str]:
        """
        Get list of enabled LLM services
        
        EFFECT: Returns services that will be used for analysis
        EXAMPLE: ['finbert', 'gemini'] if only those are enabled
        """
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
        """
        Get list of enabled emergency services
        
        EFFECT: Returns fallback services for when primary LLMs fail
        EXAMPLE: ['alpha_vantage', 'polygon'] if those are configured
        """
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
        """
        Check if FinBERT dependencies are available
        
        EFFECT: Determines if FinBERT can be enabled
        REQUIREMENT: Requires 'pip install torch transformers'
        """
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    @classmethod
    def create_directories(cls) -> None:
        """
        Create necessary directories if they don't exist
        
        EFFECT: Ensures data and output folders exist
        SAFETY: Prevents file path errors on first run
        """
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """
        Validate that required API keys are present and enabled
        
        EFFECT: Shows which services are actually available
        RETURN: Dictionary of service_name: available_boolean
        """
        return {
            'fmp': bool(cls.FMP_API_KEY),
            'gemini': bool(cls.GEMINI_API_KEY) and cls.ENABLE_GEMINI,
            'openai': bool(cls.OPENAI_API_KEY) and cls.ENABLE_OPENAI,
            'anthropic': bool(cls.ANTHROPIC_API_KEY) and cls.ENABLE_CLAUDE,
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_API_KEY) and cls.ENABLE_ALPHA_VANTAGE,
            'polygon': bool(cls.POLYGON_API_KEY) and cls.ENABLE_POLYGON,
            'tiingo': bool(cls.TIINGO_API_KEY) and cls.ENABLE_TIINGO
        }

    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration
        
        EFFECT: Useful for debugging and understanding current settings
        RETURN: Dictionary with key configuration values
        """
        enabled_services = cls.get_enabled_llm_services() + cls.get_enabled_emergency_services()
        
        return {
            'enabled_services': enabled_services,
            'service_count': len(enabled_services),
            'confidence_threshold': cls.MIN_CONFIDENCE_THRESHOLD,
            'max_articles': cls.MAX_NEWS_ARTICLES,
            'news_age_range': f"{cls.MIN_NEWS_AGE_MINUTES}min - {cls.MAX_NEWS_AGE_HOURS}h",
            'output_file': str(cls.CSV_OUTPUT_PATH),
            'has_finbert_deps': cls.has_finbert_dependencies()
        }


# ================================================================
# 🚀 QUICK START RECOMMENDATIONS
# ================================================================

"""
💡 RECOMMENDED CONFIGURATIONS:

🆓 FREE TIER SETUP (Minimal costs):
- Enable: FinBERT, Gemini, Alpha Vantage, Keywords
- Set MAX_NEWS_ARTICLES = 500
- Set MIN_CONFIDENCE_THRESHOLD = 0.7

💰 PREMIUM SETUP (Best quality):  
- Enable: All services with valid API keys
- Set MAX_NEWS_ARTICLES = 1000
- Set MIN_CONFIDENCE_THRESHOLD = 0.6
- Set LLM_REQUEST_DELAY = 0.5

⚡ FAST TESTING SETUP:
- Enable: FinBERT, Keywords only
- Set MAX_NEWS_ARTICLES = 100
- Set MIN_CONFIDENCE_THRESHOLD = 0.5

🛡️ CONSERVATIVE SETUP (High precision):
- Enable: FinBERT, Gemini, Alpha Vantage
- Set MIN_CONFIDENCE_THRESHOLD = 0.8
- Set MIN_NEWS_LENGTH = 100

🎯 TROUBLESHOOTING:

❌ No decisions logged?
- Lower MIN_CONFIDENCE_THRESHOLD to 0.4
- Check if services are actually enabled in logs
- Ensure API keys are valid

⏱️ Processing too slow?
- Reduce MAX_NEWS_ARTICLES to 500
- Disable expensive services (OpenAI, Claude)
- Increase LLM_REQUEST_DELAY to avoid rate limits

💸 API costs too high?
- Disable OpenAI and Claude
- Use only free tier services
- Reduce MAX_NEWS_ARTICLES

🔄 Missing news?
- Increase MAX_NEWS_AGE_HOURS to 48
- Add more NEWS_SOURCES
- Check FMP API quotas
"""

# Create directories on import
Config.create_directories()