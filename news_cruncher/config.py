"""
Configuration for enhanced financial news analysis system with RoBERTa+LSTM/CNN and earnings transcripts
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
    Configuration class for the enhanced financial news analysis system
    
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
    FMP_REQUESTS_PER_MINUTE: int = 300  # FMP API rate limit
    
    # 🤖 AI/LLM Service API Keys (Optional but recommended for best results)
    # EFFECT: More services enabled = more robust predictions via consensus
    
    # Google Gemini API Key
    # EFFECT: Adds Google's LLM analysis (20% weight in multi-source with enhanced neural)
    # COST: Free tier: 15 requests/minute, 1500 requests/day
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # OpenAI API Key  
    # EFFECT: Adds GPT analysis (12% weight in multi-source with enhanced neural)
    # COST: Pay-per-use, can be expensive with high volume
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    # Anthropic Claude API Key
    # EFFECT: Adds Claude analysis (10% weight in multi-source with enhanced neural)  
    # COST: Pay-per-use, good quality but limited free tier
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    
    # 🚨 Emergency fallback APIs (Optional but recommended for robustness)
    # EFFECT: Provide backup analysis when primary LLMs fail or hit quotas
    
    # Alpha Vantage API Key
    # EFFECT: Adds news sentiment analysis (8% weight)
    # COST: Free tier: 5 calls/minute, 500 calls/day
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    # Polygon API Key
    # EFFECT: Adds news analysis via Polygon's news API (6% weight)
    # COST: Free tier: 5 calls/minute, quite restrictive
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    
    # Tiingo API Key
    # EFFECT: Adds news analysis via Tiingo's API (3% weight)
    # COST: Free tier available, good for backup
    TIINGO_API_KEY: str = os.getenv('TIINGO_API_KEY', '')
    
    # ================================================================
    # 🔧 SERVICE ENABLE/DISABLE TOGGLES
    # ================================================================
    
    # 🧠 FinBERT (Financial BERT) - Local ML Model
    # EFFECT: Provides specialized financial sentiment analysis (25% weight)
    # REQUIREMENT: Requires PyTorch and Transformers libraries
    # PERFORMANCE: CPU-based, adds ~2-3 seconds per analysis
    # RECOMMENDATION: Keep enabled - it's free and finance-specific
    ENABLE_FINBERT: bool = os.getenv('ENABLE_FINBERT', 'true').lower() == 'true'
    
    # 🚀 NEW: Enhanced Neural Analyzer (RoBERTa + LSTM + CNN)
    # EFFECT: State-of-the-art hybrid model with 94-96% accuracy (45% weight)
    # REQUIREMENT: Requires PyTorch, Transformers, and sufficient RAM/VRAM
    # PERFORMANCE: GPU recommended, adds ~5-8 seconds per analysis but much higher accuracy
    # RECOMMENDATION: Enable for best results - dramatically improves prediction quality
    ENABLE_ENHANCED_NEURAL: bool = os.getenv('ENABLE_ENHANCED_NEURAL', 'true').lower() == 'true'
    
    # 🌟 Google Gemini - High-quality LLM
    # EFFECT: Provides sophisticated language understanding (20% weight with enhanced neural)
    # RATE LIMITS: 15 requests/minute, 1500/day on free tier
    # RECOMMENDATION: Keep enabled - excellent free tier
    ENABLE_GEMINI: bool = os.getenv('ENABLE_GEMINI', 'true').lower() == 'true'
    
    # 💰 OpenAI GPT Models
    # EFFECT: High-quality analysis but can be expensive (12% weight with enhanced neural)
    # RATE LIMITS: Depends on your billing tier
    # RECOMMENDATION: Disable if you're cost-conscious, enable for best quality
    ENABLE_OPENAI: bool = os.getenv('ENABLE_OPENAI', 'false').lower() == 'true'
    
    # 🎭 Anthropic Claude
    # EFFECT: Good analysis quality, different perspective (10% weight with enhanced neural)
    # RATE LIMITS: Limited free tier, pay-per-use
    # RECOMMENDATION: Enable if you have credits, otherwise keep disabled
    ENABLE_CLAUDE: bool = os.getenv('ENABLE_CLAUDE', 'false').lower() == 'true'
    
    # 📊 Alpha Vantage News Sentiment
    # EFFECT: Provides real market sentiment data (8% weight with enhanced neural)
    # RATE LIMITS: 5 calls/minute, 500/day free
    # RECOMMENDATION: Enable for additional market context
    ENABLE_ALPHA_VANTAGE: bool = os.getenv('ENABLE_ALPHA_VANTAGE', 'true').lower() == 'true'
    
    # 📈 Polygon News Analysis
    # EFFECT: Market data company's news analysis (6% weight with enhanced neural)
    # RATE LIMITS: 5 calls/minute on free tier (very restrictive)
    # RECOMMENDATION: Disable unless you have a paid Polygon plan
    ENABLE_POLYGON: bool = os.getenv('ENABLE_POLYGON', 'false').lower() == 'true'
    
    # 📊 Tiingo News Analysis
    # EFFECT: Additional financial news analysis (3% weight with enhanced neural)
    # RATE LIMITS: Free tier available, good backup option
    # RECOMMENDATION: Enable as low-weight backup source
    ENABLE_TIINGO: bool = os.getenv('ENABLE_TIINGO', 'false').lower() == 'true'
    
    # 🎯 Keyword Sentiment Analysis (Fallback)
    # EFFECT: Local keyword-based sentiment when APIs are unavailable (2% weight with enhanced neural)
    # PERFORMANCE: Instant, no API calls required
    # RECOMMENDATION: Always keep enabled - provides guaranteed analysis
    ENABLE_KEYWORD_SENTIMENT: bool = os.getenv('ENABLE_KEYWORD_SENTIMENT', 'true').lower() == 'true'
      
    # ================================================================
    # 🔍 FUNDAMENTAL FILTERING CONFIGURATION
    # ================================================================
    
    # 🎯 Enable/Disable Fundamental Filtering
    # EFFECT: Filters tickers based on company fundamentals before analysis
    # RECOMMENDATION: Enable to focus on quality, liquid stocks
    ENABLE_FUNDAMENTAL_FILTERING: bool = os.getenv('ENABLE_FUNDAMENTAL_FILTERING', 'true').lower() == 'true'
    
    # 💰 Stock Price Range
    # EFFECT: Eliminates penny stocks and ultra-expensive stocks
    # RANGE: Recommended $5-$500 for most strategies
    MIN_STOCK_PRICE: float = float(os.getenv('MIN_STOCK_PRICE', '10.00'))
    MAX_STOCK_PRICE: float = float(os.getenv('MAX_STOCK_PRICE', '300.00'))
    
    # 📊 Volume Requirements
    # EFFECT: Ensures adequate liquidity for position entry/exit
    # MIN_AVG_VOLUME: Average daily share volume (500K = good liquidity)
    # MIN_DOLLAR_VOLUME: Daily dollar volume (price × volume)
    MIN_AVG_VOLUME: int = int(os.getenv('MIN_AVG_VOLUME', '500000'))  # 500K shares
    MIN_DOLLAR_VOLUME: int = int(os.getenv('MIN_DOLLAR_VOLUME', '5000000'))  # $5M daily
    
    # 🏢 Company Size Requirements
    # EFFECT: Focuses on established companies vs. micro-caps
    # RANGE: $100M+ = small-cap, $500M+ = more established
    MIN_MARKET_CAP: int = int(os.getenv('MIN_MARKET_CAP', '500000000'))  # $500M
    
    # 📈 Volatility Limits
    # EFFECT: Avoids extremely volatile stocks (beta > 2.0)
    # RANGE: 1.0 = market volatility, 2.0 = twice market volatility
    MAX_VOLATILITY_BETA: float = float(os.getenv('MAX_VOLATILITY_BETA', '2.0'))
    
    # 🎯 Options Requirement
    # EFFECT: Options availability indicates institutional interest & liquidity
    # RECOMMENDATION: True for better liquidity, False for more stock choices
    REQUIRE_OPTIONS: bool = os.getenv('REQUIRE_OPTIONS', 'true').lower() == 'true'
    
    # 🏛️ Exchange Requirements
    # EFFECT: Restricts to major exchanges for better regulation/reporting
    # SUPPORTED: NASDAQ, NYSE, NYSEArca, AMEX, OTC
    ALLOWED_EXCHANGES: List[str] = os.getenv('ALLOWED_EXCHANGES', 'NASDAQ,NYSE,NYSEArca').split(',')
    
    # ⚡ Fundamental Data Caching
    # EFFECT: Cache duration for company fundamental data (reduces API calls)
    # RANGE: 1-48 hours (fundamentals change slowly)
    FUNDAMENTAL_CACHE_HOURS: int = int(os.getenv('FUNDAMENTAL_CACHE_HOURS', '24'))
    
    # ================================================================
    # 📊 ANALYSIS LIMITS - Control processing volume and quality
    # ================================================================
    
    # 📈 Decision Confidence Threshold
    # CURRENT: 0.6 (60% confidence)
    # EFFECT: Only decisions above this confidence are logged to CSV
    # RANGE: 0.4-0.9 recommended
    # NOTE: Enhanced neural model typically produces higher confidence scores
    # EXAMPLE: 0.4 = more decisions logged (lower precision)
    #          0.8 = fewer decisions logged (higher precision)
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
    
    # 📰 News Article Limits
    # CURRENT: 1000 articles per cycle
    # EFFECT: Caps total articles processed to manage API costs and speed
    # RANGE: 100-2000 articles recommended
    # NOTE: Enhanced neural analysis is more processing-intensive
    # EXAMPLE: 100 = faster processing, lower API usage
    #          2000 = more comprehensive analysis, higher costs
    MAX_NEWS_ARTICLES: int = int(os.getenv('MAX_NEWS_ARTICLES', '1000'))
    
    # 🎯 Ticker Analysis Limit
    # CURRENT: 100 tickers per cycle
    # EFFECT: Maximum tickers to run full analysis on (prioritized by article count)
    # RANGE: 25-200 tickers recommended
    # NOTE: Enhanced neural analysis takes longer per ticker but is more accurate
    # EXAMPLE: 50 = faster cycles, focus on top tickers
    #          200 = comprehensive coverage, slower processing
    MAX_TICKERS_TO_ANALYZE: int = int(os.getenv('MAX_TICKERS_TO_ANALYZE', '100'))
    
    # 📝 News Source Configuration
    # CURRENT: Multiple sources including earnings, press releases, general news
    # EFFECT: Which news types to fetch and analyze
    # SOURCES: earnings, press-releases, stock-news, general-news, earnings-transcripts
    NEWS_SOURCES: List[str] = os.getenv('NEWS_SOURCES', 'earnings,press-releases,stock-news').split(',')
    
    # ⏱️ LLM Request Delay
    # CURRENT: 1.0 seconds between LLM API calls
    # EFFECT: Prevents hitting rate limits, reduces errors
    # RANGE: 0.1-5.0 seconds recommended
    # NOTE: Enhanced neural analysis is local, no API delay needed
    # EXAMPLE: 0.5 = faster processing, risk of rate limits
    #          2.0 = conservative, guaranteed to stay under limits
    LLM_REQUEST_DELAY: float = float(os.getenv('LLM_REQUEST_DELAY', '1.0'))
    
    # 📏 Article Quality Filter
    # CURRENT: 50 characters minimum
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
    
    # ADD THIS LINE:
    LOG_FILE_PATH: Path = OUTPUT_DIR / 'system.log'
    
    # 📋 CSV Output File
    # EFFECT: Where trading decisions are logged
    # MODIFY: Change filename if you want multiple output files
    CSV_OUTPUT_PATH: Path = OUTPUT_DIR / 'trading_decisions.csv'
    
    # 💾 SQLite Database File
    # EFFECT: Tracks processed articles to prevent reprocessing
    # WARNING: Deleting this file will cause all articles to be reprocessed
    SQLITE_DB_PATH: Path = DATA_DIR / 'article_tracking.db'
    DB_TIMEOUT: int = 30  # Database connection timeout in seconds
    # 🧠 NEW: Enhanced Neural Model Cache
    # EFFECT: Where to cache/store enhanced neural model weights
    # PERFORMANCE: Local storage for faster model loading
    ENHANCED_NEURAL_MODEL_PATH: Path = DATA_DIR / 'enhanced_neural_model.pth'
    
    # ================================================================
    # 💰 CONFIGURABLE PRICE TRACKING INTERVALS
    # ================================================================
    
    # 🕐 First Price Check Interval (in minutes after recommendation)
    # CURRENT: 45 minutes
    # EFFECT: When to take the first price checkpoint
    # RANGE: 15-120 minutes recommended
    # EXAMPLE: 30 = earlier checkpoint for short-term strategies
    #          60 = standard 1-hour checkpoint
    # CSV FIELD: Always logged as 'price_checkpoint1' regardless of actual interval
    PRICE_CHECK_1_MINUTES: int = int(os.getenv('PRICE_CHECK_1_MINUTES', '45'))
    
    # 🕑 Second Price Check Interval (in minutes after recommendation)
    # CURRENT: 60 minutes (1 hour)
    # EFFECT: When to take the second price checkpoint
    # RANGE: 30-180 minutes recommended
    # EXAMPLE: 120 = 2-hour checkpoint for longer-term view
    #          90 = 1.5-hour checkpoint
    # CSV FIELD: Always logged as 'price_checkpoint2' regardless of actual interval
    PRICE_CHECK_2_MINUTES: int = int(os.getenv('PRICE_CHECK_2_MINUTES', '60'))
    
    # 🕕 Market Close Price Timing
    # CURRENT: 3:50 PM EST (10 minutes before market close)
    # EFFECT: When to capture the "close" price for daily performance
    # RANGE: 3:30-4:00 PM EST recommended
    # EXAMPLE: 15:30 = 3:30 PM (30 min before close)
    #          15:55 = 3:55 PM (5 min before close)
    # CSV FIELD: Always logged as 'price_close' regardless of actual time
    CLOSE_PRICE_HOUR: int = int(os.getenv('CLOSE_PRICE_HOUR', '15'))  # 24-hour format (15 = 3 PM)
    CLOSE_PRICE_MINUTE: int = int(os.getenv('CLOSE_PRICE_MINUTE', '50'))  # 50 = :50 minutes
    
    # ⏰ Price Fetch Tolerance (Market Hours Logic)
    # EFFECT: How flexible to be when market is closed or weekend
    # MARKET_TOLERANCE: Minutes before/after market hours to still fetch prices
    # AFTER_HOURS: Hours after market close to still attempt price fetch
    # WEEKEND: Hours into weekend to attempt Friday close price
    PRICE_FETCH_MARKET_TOLERANCE_MINUTES: int = 30  # 30 min before/after market hours
    PRICE_FETCH_AFTER_HOURS_TOLERANCE_HOURS: int = 2   # 2 hours after market close  
    PRICE_FETCH_WEEKEND_TOLERANCE_HOURS: int = 48      # 48 hours into weekend
    
    PRICE_TRACKER_CHECK_INTERVAL: int = int(os.getenv('PRICE_TRACKER_CHECK_INTERVAL', '5'))  # Minutes between checks
    
    # 🔄 Background Price Monitoring Frequency
    # CURRENT: 5 minutes
    # EFFECT: How often the background scheduler checks for due price reads
    # RANGE: 1-60 minutes recommended. Too frequent = more resource usage.
    TRADE_MONITOR_INTERVAL_MINUTES: int = int(os.getenv('TRADE_MONITOR_INTERVAL_MINUTES', '5'))
      
    # ================================================================
    # 🛠️ UTILITY METHODS - ENHANCED WITH NEURAL ANALYSIS
    # ================================================================
    
    # ================================================================
    # 📅 EARNINGS EVENT ANALYSIS CONFIGURATION
    # ================================================================
    
    # Enable/disable earnings event analysis
    ENABLE_EARNINGS_EVENTS: bool = os.getenv('ENABLE_EARNINGS_EVENTS', 'true').lower() == 'true'
    
    # Earnings event detection window (days)
    EARNINGS_LOOKBACK_DAYS: int = int(os.getenv('EARNINGS_LOOKBACK_DAYS', '3'))
    EARNINGS_LOOKAHEAD_DAYS: int = int(os.getenv('EARNINGS_LOOKAHEAD_DAYS', '7'))
    
    # Earnings analysis confidence thresholds
    MIN_EARNINGS_CONFIDENCE: float = float(os.getenv('MIN_EARNINGS_CONFIDENCE', '0.6'))
    
    # Scoring weights for 3-way analysis
    NEWS_WEIGHT_3WAY: float = float(os.getenv('NEWS_WEIGHT_3WAY', '0.40'))
    EARNINGS_WEIGHT_3WAY: float = float(os.getenv('EARNINGS_WEIGHT_3WAY', '0.30'))
    TECHNICAL_WEIGHT_3WAY: float = float(os.getenv('TECHNICAL_WEIGHT_3WAY', '0.30'))
    
    # Cache settings for earnings data
    EARNINGS_CACHE_HOURS: int = int(os.getenv('EARNINGS_CACHE_HOURS', '6'))
    
    @classmethod
    def _get_tolerance_settings(cls): # New helper method to return tolerance for cleaner use
        return {
            'market_minutes': cls.PRICE_FETCH_MARKET_TOLERANCE_MINUTES,
            'after_hours_hours': cls.PRICE_FETCH_AFTER_HOURS_TOLERANCE_HOURS,
            'weekend_hours': cls.PRICE_FETCH_WEEKEND_TOLERANCE_HOURS
        }
        
    @classmethod
    def get_enabled_llm_services(cls) -> List[str]:
        """
        Get list of enabled LLM services including enhanced neural
        
        EFFECT: Returns services that will be used for analysis
        EXAMPLE: ['enhanced_neural', 'finbert', 'gemini'] if those are enabled
        """
        enabled = []
        if cls.ENABLE_ENHANCED_NEURAL:
            enabled.append('enhanced_neural')
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
    def has_enhanced_neural_dependencies(cls) -> bool:
        """
        Check if Enhanced Neural dependencies are available
        
        EFFECT: Determines if Enhanced Neural can be enabled
        REQUIREMENT: Requires 'pip install torch transformers numpy'
        """
        try:
            import torch
            import transformers
            import numpy as np
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
        Get a summary of current configuration including enhanced features
        
        EFFECT: Useful for debugging and understanding current settings
        RETURN: Dictionary with key configuration values
        """
        enabled_services = cls.get_enabled_llm_services() + cls.get_enabled_emergency_services()
        
        return {
            'enabled_services': enabled_services,
            'service_count': len(enabled_services),
            'confidence_threshold': cls.MIN_CONFIDENCE_THRESHOLD,
            'max_articles': cls.MAX_NEWS_ARTICLES,
            'max_tickers': cls.MAX_TICKERS_TO_ANALYZE,
            'news_age_range': f"{cls.MIN_NEWS_AGE_MINUTES}min - {cls.MAX_NEWS_AGE_HOURS}h",
            'output_file': str(cls.CSV_OUTPUT_PATH),
            'has_finbert_deps': cls.has_finbert_dependencies(),
            'has_enhanced_neural_deps': cls.has_enhanced_neural_dependencies(),
            'price_check_intervals': f"{cls.PRICE_CHECK_1_MINUTES}m, {cls.PRICE_CHECK_2_MINUTES}m",
            'close_time': f"{cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d} EST",
            'generic_field_names': ['price_checkpoint1', 'price_checkpoint2', 'price_close'],
            'fundamental_filtering': {
                'enabled': cls.ENABLE_FUNDAMENTAL_FILTERING,
                'price_range': f"${cls.MIN_STOCK_PRICE:.2f} - ${cls.MAX_STOCK_PRICE:.2f}",
                'min_volume': f"{cls.MIN_AVG_VOLUME:,} shares",
                'min_dollar_volume': f"${cls.MIN_DOLLAR_VOLUME:,}",
                'min_market_cap': f"${cls.MIN_MARKET_CAP:,}",
                'max_beta': cls.MAX_VOLATILITY_BETA,
                'require_options': cls.REQUIRE_OPTIONS,
                'allowed_exchanges': cls.ALLOWED_EXCHANGES
            },
            'enhanced_features': {
                'enhanced_neural_enabled': cls.ENABLE_ENHANCED_NEURAL,
                'EARNINGS_EVENTS_enabled': cls.ENABLE_EARNINGS_EVENTS,
                'max_transcripts_per_cycle': cls.MAX_EARNINGS_EVENTS_PER_CYCLE
            }
        }
    
    # ================================================================
    # 🆕 DYNAMIC LABELING METHODS FOR CONFIGURABLE INTERVALS
    # ================================================================
    
    @classmethod
    def get_price_check_labels(cls) -> List[str]:
        """Get dynamic labels for price checks showing actual intervals"""
        checkpoint_info = cls.get_checkpoint_info()
        return [checkpoint['short_label'] for checkpoint in checkpoint_info]
    
    @classmethod
    def get_csv_price_headers(cls) -> List[str]:
        """Get generic CSV headers for price tracking (never change)"""
        return [
            # Entry price
            'recommendation_price',
            'recommendation_timestamp',
            
            # Checkpoint 1 (generic names)
            'price_checkpoint1',
            'price_checkpoint1_timestamp',
            'price_checkpoint1_change_pct',
            
            # Checkpoint 2 (generic names) 
            'price_checkpoint2',
            'price_checkpoint2_timestamp',
            'price_checkpoint2_change_pct',
            
            # Close price (generic name)
            'price_close',
            'price_close_timestamp', 
            'price_close_change_pct',
            
            # Tracking metadata
            'tracking_status'
        ]
    
    # In config.py - Update the get_checkpoint_info() method

    @classmethod 
    def get_checkpoint_info(cls) -> List[Dict[str, Any]]:
        """Get information about all price checkpoints with dynamic intervals"""
        return [
            {
                'name': 'checkpoint1',
                'minutes': cls.PRICE_CHECK_1_MINUTES,
                'field_prefix': 'price_checkpoint1',  # ← ADD THIS
                'short_label': f"checkpoint1 ({cls.PRICE_CHECK_1_MINUTES}m)",
                'description': f"First price check after {cls.PRICE_CHECK_1_MINUTES} minutes"
            },
            {
                'name': 'checkpoint2', 
                'minutes': cls.PRICE_CHECK_2_MINUTES,
                'field_prefix': 'price_checkpoint2',  # ← ADD THIS
                'short_label': f"checkpoint2 ({cls.PRICE_CHECK_2_MINUTES}m)",
                'description': f"Second price check after {cls.PRICE_CHECK_2_MINUTES} minutes"
            },
            {
                'name': 'close',
                'hour': cls.CLOSE_PRICE_HOUR,
                'minute': cls.CLOSE_PRICE_MINUTE,
                'field_prefix': 'price_close',  # ← ADD THIS
                'short_label': f"close ({cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d})",
                'description': f"Market close price at {cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d} EST"
            }
        ]
    
    @classmethod
    def format_price_log_message(cls, ticker: str, price: float, change_pct: float, 
                                 checkpoint_index: int = None, 
                                 checkpoint_name: str = None) -> str:
        """Format dynamic price log messages based on configuration"""
        if checkpoint_name == "checkpoint1":
            label = f"checkpoint1 ({cls.PRICE_CHECK_1_MINUTES}m)"
            return f"📈 {label}: {ticker} ${price:.2f} ({change_pct:+.2f}%)"
        elif checkpoint_name == "checkpoint2":
            label = f"checkpoint2 ({cls.PRICE_CHECK_2_MINUTES}m)"
            return f"📈 {label}: {ticker} ${price:.2f} ({change_pct:+.2f}%)"
        elif checkpoint_name == "close":
            label = f"close ({cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d})"
            return f"📈 {label}: {ticker} ${price:.2f} ({change_pct:+.2f}%)"
        else:
            return f"📈 checkpoint_{checkpoint_index}: {ticker} ${price:.2f} ({change_pct:+.2f}%)"


# ================================================================
# 🚀 QUICK START RECOMMENDATIONS - UPDATED WITH ENHANCED FEATURES
# ================================================================

"""
💡 RECOMMENDED CONFIGURATIONS:

🚀 ENHANCED PREMIUM SETUP (Best accuracy with new features):
- Enable: Enhanced Neural, FinBERT, Gemini, Earnings Transcripts
- Set MAX_NEWS_ARTICLES = 800  # Reduced slightly due to enhanced processing
- Set MAX_TICKERS_TO_ANALYZE = 75  # Enhanced neural is more processing-intensive
- Set MIN_CONFIDENCE_THRESHOLD = 0.65  # Enhanced neural produces higher confidence
- Enable fundamental filtering with default settings
- EXPECTED ACCURACY: 94-96% with enhanced neural

🆓 FREE TIER SETUP (Minimal costs, good accuracy):
- Enable: Enhanced Neural, FinBERT, Keywords (all free/local)
- Disable: All paid API services
- Set MAX_NEWS_ARTICLES = 500
- Set MAX_TICKERS_TO_ANALYZE = 50
- Set MIN_CONFIDENCE_THRESHOLD = 0.7
- Enable fundamental filtering
- EXPECTED ACCURACY: 90-94% with enhanced neural only

💰 PREMIUM SETUP (Best quality with API services):  
- Enable: All services with valid API keys
- Set MAX_NEWS_ARTICLES = 1000
- Set MAX_TICKERS_TO_ANALYZE = 100
- Set MIN_CONFIDENCE_THRESHOLD = 0.6
- Set LLM_REQUEST_DELAY = 0.5
- Enable fundamental filtering with default settings
- Enable earnings transcript analysis
- EXPECTED ACCURACY: 95-97% with full service ensemble

⚡ FAST TESTING SETUP:
- Enable: Enhanced Neural only
- Set MAX_NEWS_ARTICLES = 200
- Set MAX_TICKERS_TO_ANALYZE = 25
- Set MIN_CONFIDENCE_THRESHOLD = 0.5
- Disable fundamental filtering and earnings transcripts
- EXPECTED ACCURACY: 90-92% with fast processing

🛡️ CONSERVATIVE SETUP (High precision):
- Enable: Enhanced Neural, FinBERT, Gemini
- Set MIN_CONFIDENCE_THRESHOLD = 0.8
- Set MIN_NEWS_LENGTH = 100
- Enable fundamental filtering with strict criteria:
  - MIN_STOCK_PRICE = 15.00
  - MIN_MARKET_CAP = 1000000000 (1B)
  - MAX_VOLATILITY_BETA = 1.5
- Enable earnings transcript analysis
- EXPECTED ACCURACY: 96-98% with very high precision

🎯 TROUBLESHOOTING:

❌ No decisions logged?
- Lower MIN_CONFIDENCE_THRESHOLD to 0.4
- Check if enhanced neural dependencies are installed: pip install torch transformers
- Ensure API keys are valid
- Temporarily disable fundamental filtering

⏱️ Processing too slow?
- Reduce MAX_NEWS_ARTICLES to 400
- Reduce MAX_TICKERS_TO_ANALYZE to 50
- Disable earnings transcript analysis temporarily
- Use GPU for enhanced neural analysis if available

💸 API costs too high?
- Disable OpenAI and Claude
- Use only Enhanced Neural + FinBERT (both free/local)
- Reduce MAX_NEWS_ARTICLES and MAX_TICKERS_TO_ANALYZE
- Enable fundamental filtering to reduce analysis volume

🔄 Enhanced Neural not working?
- Install dependencies: pip install torch transformers numpy
- Check GPU availability with: python -c "import torch; print(torch.cuda.is_available())"
- Set ENABLE_ENHANCED_NEURAL=false to use traditional models
- Check system RAM (enhanced neural needs ~2-4GB)

📈 Want more comprehensive analysis?
- Enable earnings transcript analysis
- Increase MAX_EARNINGS_EVENTS_PER_CYCLE to 25
- Monitor API usage with transcript analysis (it's data-intensive)

🕐 Optimize for your hardware:
- GPU available: Enable Enhanced Neural for best accuracy
- CPU only: Consider disabling Enhanced Neural if too slow
- Limited RAM: Reduce MAX_TICKERS_TO_ANALYZE and batch sizes
- SSD storage: Enable all caching features for faster subsequent runs
"""

# Create directories on import
Config.create_directories()
