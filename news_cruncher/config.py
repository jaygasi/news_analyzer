"""
Enhanced Configuration System for Financial News Analysis
Manages all settings for neural networks, earnings analysis, and fundamental filtering
Python 3.13.3 compatible - FIXED VERSION
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Enhanced configuration class with neural networks and earnings analysis settings"""
    
    # ================================================================
    # 🗂️ BASE DIRECTORY CONFIGURATION
    # ================================================================
    
    # Base Directory (Auto-detected)
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
    # 🛠️ Logging Settings
    # ================================================================
        # NEW: Rejected stock logging configuration
    ENABLE_REJECTED_STOCK_LOGGING = True  # Set to False to disable rejected stock logging
    MAX_REJECTED_STOCKS_TO_LOG = 30  # Number of rejected stocks to log (set to 0 to disable)
    
    # NEW: CSV logging configuration  
    ONLY_LOG_TRADING_DECISIONS = True  # Only log LONG/SHORT decisions, skip NONE
    
    # NEW: Price tracking scheduler configuration
    START_SCHEDULER_ON_FIRST_DECISION = True  # Wait for first trading decision before starting scheduler
    
    # ================================================================
    # 📅 EARNINGS EVENT ANALYSIS CONFIGURATION
    # ================================================================
    
    # Enable/disable earnings event analysis
    ENABLE_EARNINGS_EVENTS: bool = os.getenv('ENABLE_EARNINGS_EVENTS', 'true').lower() == 'true'
    
    # FIXED: Add the missing MAX_EARNINGS_EVENTS_PER_CYCLE configuration
    # Maximum number of earnings transcripts to fetch per cycle
    # CURRENT: 15 transcripts per cycle
    # EFFECT: Limits API usage and processing time for earnings transcripts
    # RANGE: 5-50 recommended depending on API quota
    # EXAMPLE: 5 = conservative, faster processing
    #          25 = comprehensive analysis, higher API usage
    MAX_EARNINGS_EVENTS_PER_CYCLE: int = int(os.getenv('MAX_EARNINGS_EVENTS_PER_CYCLE', '15'))
    
    # Earnings event detection window (days)
    EARNINGS_LOOKBACK_DAYS: int = int(os.getenv('EARNINGS_LOOKBACK_DAYS', '3'))
    EARNINGS_LOOKAHEAD_DAYS: int = int(os.getenv('EARNINGS_LOOKAHEAD_DAYS', '7'))
    
    # Earnings analysis confidence thresholds
    MIN_EARNINGS_CONFIDENCE: float = float(os.getenv('MIN_EARNINGS_CONFIDENCE', '0.6'))
    
    # 2-way scoring weights (traditional: news + technical)
    NEWS_WEIGHT_2WAY: float = float(os.getenv('NEWS_WEIGHT_2WAY', '0.70'))
    TECHNICAL_WEIGHT_2WAY: float = float(os.getenv('TECHNICAL_WEIGHT_2WAY', '0.30'))

    # Component-specific confidence thresholds
    MIN_NEWS_CONFIDENCE: float = float(os.getenv('MIN_NEWS_CONFIDENCE', '0.5'))
    MIN_TECHNICAL_CONFIDENCE: float = float(os.getenv('MIN_TECHNICAL_CONFIDENCE', '0.4'))

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
        if cls.ENABLE_FINBERT:
            enabled.append('finbert')
        if cls.ENABLE_GEMINI:
            enabled.append('gemini')
        if cls.ENABLE_OPENAI:
            enabled.append('openai')
        if cls.ENABLE_CLAUDE:
            enabled.append('claude')
        if cls.ENABLE_ALPHA_VANTAGE:
            enabled.append('alpha_vantage')
        if cls.ENABLE_POLYGON:
            enabled.append('polygon')
        if cls.ENABLE_TIINGO:
            enabled.append('tiingo')
        return enabled
    
    # ================================================================
    # 🔑 API KEYS & SERVICE CONFIGURATION
    # ================================================================
    
    # 🏢 Financial Modeling Prep (FMP) - REQUIRED
    # EFFECT: Primary data source for news, earnings, and price data
    # COST: Free tier available, paid tiers for higher limits
    # GET KEY: https://financialmodelingprep.com/developer/docs
    FMP_API_KEY: str = os.getenv('FMP_API_KEY', '')
    FMP_REQUESTS_PER_MINUTE: int = int(os.getenv('FMP_REQUESTS_PER_MINUTE', '10'))
    # Enhanced API rate limiting (to prevent 429 errors)
    FMP_MIN_REQUEST_INTERVAL = 0.2  # Minimum seconds between FMP API requests (5 per second max)
    FMP_RETRY_DELAY = 60  # Seconds to wait after 429 error before retry
    
    # Failed ticker caching (prevents repeated API calls to bad tickers)
    ENABLE_FAILED_TICKER_CACHE = True          # Enable/disable the cache
    FAILED_TICKER_CACHE_DAYS = 30              # Days to cache failed tickers
    FAILED_TICKER_MAX_RETRIES = 3              # How many failures before caching
    FAILED_TICKER_CACHE_DB = "data/failed_tickers_cache.db"  # Cache database file
    
    # 🧠 Enhanced Neural Analysis - HIGHEST ACCURACY
    # CURRENT: ENABLED (94-96% accuracy)
    # EFFECT: Uses RoBERTa+LSTM+CNN for superior sentiment analysis
    # COST: Free (runs locally on CPU/GPU)
    # PERFORMANCE: ~2-4GB RAM, 15-20 sec/ticker, GPU accelerated
    ENABLE_ENHANCED_NEURAL: bool = os.getenv('ENABLE_ENHANCED_NEURAL', 'true').lower() == 'true'
    
    # Enhanced Neural Analyzer Settings
    ENHANCED_NEURAL_MIXED_PRECISION: bool = True  # Use mixed precision if CUDA available
    ENHANCED_NEURAL_GRADIENT_ACCUMULATION: int = 4  # Gradient accumulation steps
    ENHANCED_NEURAL_WARMUP: bool = True  # Enable model warming
    
    # 🤖 FinBERT Financial Sentiment Analysis - HIGH ACCURACY
    # CURRENT: ENABLED (~85% accuracy)
    # EFFECT: Financial domain-specific BERT model for sentiment
    # COST: Free (runs locally)
    # PERFORMANCE: ~1-2GB RAM, 5-10 sec/ticker
    ENABLE_FINBERT: bool = os.getenv('ENABLE_FINBERT', 'true').lower() == 'true'
    
    # 🟢 Google Gemini - OPTIONAL HIGH QUALITY
    # EFFECT: Google's advanced LLM for analysis
    # COST: Free tier available with generous limits
    # GET KEY: https://makersuite.google.com/app/apikey
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    ENABLE_GEMINI: bool = os.getenv('ENABLE_GEMINI', 'false').lower() == 'true'
    
    # 🔵 OpenAI GPT - OPTIONAL HIGH QUALITY
    # EFFECT: ChatGPT/GPT-4 for sentiment analysis
    # COST: Pay-per-use, typically $0.01-0.10 per ticker
    # GET KEY: https://platform.openai.com/api-keys
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    ENABLE_OPENAI: bool = os.getenv('ENABLE_OPENAI', 'false').lower() == 'true'
    
    # 🟣 Anthropic Claude - OPTIONAL HIGH QUALITY  
    # EFFECT: Claude for sophisticated analysis
    # COST: Pay-per-use, competitive pricing
    # GET KEY: https://console.anthropic.com/
    CLAUDE_API_KEY: str = os.getenv('CLAUDE_API_KEY', '')
    ENABLE_CLAUDE: bool = os.getenv('ENABLE_CLAUDE', 'false').lower() == 'true'
    
    # 📊 Alpha Vantage - FALLBACK SERVICE
    # EFFECT: Financial news and sentiment analysis
    # COST: Free tier available
    # GET KEY: https://www.alphavantage.co/support/#api-key
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    ENABLE_ALPHA_VANTAGE: bool = os.getenv('ENABLE_ALPHA_VANTAGE', 'false').lower() == 'true'
    
    # 🔺 Polygon - FALLBACK SERVICE
    # EFFECT: Market news and sentiment
    # COST: Free tier available
    # GET KEY: https://polygon.io/
    POLYGON_API_KEY: str = os.getenv('POLYGON_API_KEY', '')
    ENABLE_POLYGON: bool = os.getenv('ENABLE_POLYGON', 'false').lower() == 'true'
    
    # 📈 Tiingo - FALLBACK SERVICE  
    # EFFECT: Financial news analysis
    # COST: Free tier available
    # GET KEY: https://api.tiingo.com/
    TIINGO_API_KEY: str = os.getenv('TIINGO_API_KEY', '')
    ENABLE_TIINGO: bool = os.getenv('ENABLE_TIINGO', 'false').lower() == 'true'
    
    # 🔤 Enhanced Keyword Sentiment - ULTRA FALLBACK
    # EFFECT: Rule-based sentiment analysis as last resort
    # COST: Free (built-in)
    # ACCURACY: ~60-70% (basic but reliable)
    ENABLE_KEYWORD_SENTIMENT: bool = os.getenv('ENABLE_KEYWORD_SENTIMENT', 'true').lower() == 'true'
    
    # ================================================================
    # 🔍 ENHANCED FUNDAMENTAL FILTERING CONFIGURATION
    # ================================================================
    
    # Enable/disable fundamental filtering
    ENABLE_FUNDAMENTAL_FILTERING: bool = os.getenv('ENABLE_FUNDAMENTAL_FILTERING', 'true').lower() == 'true'
    
    # Price range filtering
    MIN_STOCK_PRICE: float = float(os.getenv('MIN_STOCK_PRICE', '3.00'))
    MAX_STOCK_PRICE: float = float(os.getenv('MAX_STOCK_PRICE', '1000.00'))
    
    # Volume filtering
    MIN_AVG_VOLUME: int = int(os.getenv('MIN_AVG_VOLUME', '250000'))  # Minimum average daily volume
    MIN_DOLLAR_VOLUME: int = int(os.getenv('MIN_DOLLAR_VOLUME', '5000000'))  # Minimum daily dollar volume
    
    # Market cap filtering
    MIN_MARKET_CAP: int = int(os.getenv('MIN_MARKET_CAP', '250000000'))  # $250M minimum
    
    # Risk filtering
    MAX_VOLATILITY_BETA: float = float(os.getenv('MAX_VOLATILITY_BETA', '2.0'))  # Maximum beta (volatility)
    
    # Options availability
    REQUIRE_OPTIONS: bool = os.getenv('REQUIRE_OPTIONS', 'false').lower() == 'true'
    
    # Exchange filtering
    ALLOWED_EXCHANGES: List[str] = os.getenv('ALLOWED_EXCHANGES', 'NASDAQ,NYSE,NYSEArca,CBOE,BATS,AMEX').split(',')
    
    # ================================================================
    # 📊 ANALYSIS THRESHOLDS & LIMITS
    # ================================================================
    
    # 🎯 Decision Confidence Threshold
    # CURRENT: 0.6 (60% confidence minimum)
    # EFFECT: Only decisions above this confidence are logged/tracked
    # RANGE: 0.5-0.9 recommended
    # EXAMPLE: 0.5 = more decisions, lower quality
    #          0.8 = fewer decisions, higher quality
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
    
    # ================================================================
    # ⏰ TIMING & SCHEDULING CONFIGURATION
    # ================================================================
    
    # 🔄 Main Cycle Interval
    # CURRENT: 5 minutes between analysis cycles
    # EFFECT: How often to fetch news and make new decisions
    # RANGE: 1-60 minutes recommended
    # EXAMPLE: 1 = very frequent updates, higher API usage
    #          15 = less frequent, more conservative resource usage
    CYCLE_INTERVAL_MINUTES: int = int(os.getenv('CYCLE_INTERVAL_MINUTES', '5'))
    
    # 📰 News Lookback Configuration
    # CURRENT: 24 hours default, 48 hours maximum
    # EFFECT: How far back to fetch news articles
    # RANGE: 1-72 hours recommended
    # NOTE: Longer lookback = more articles but older news
    DEFAULT_NEWS_LOOKBACK_HOURS: int = int(os.getenv('DEFAULT_NEWS_LOOKBACK_HOURS', '24'))
    MAX_NEWS_AGE_HOURS: int = int(os.getenv('MAX_NEWS_AGE_HOURS', '48'))
    
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
        """
        Get comprehensive checkpoint configuration information
        
        Returns list of checkpoint dictionaries with all metadata:
        - short_label: Human-readable label (e.g., "checkpoint1 (45m)")
        - field_prefix: CSV field prefix (e.g., "price_checkpoint1")
        - minutes: Minutes after recommendation (None for close)
        - description: Full description
        """
        return [
            {
                'short_label': f'checkpoint1 ({cls.PRICE_CHECK_1_MINUTES}m)',
                'field_prefix': 'price_checkpoint1',
                'minutes': cls.PRICE_CHECK_1_MINUTES,
                'description': f'First price check {cls.PRICE_CHECK_1_MINUTES} minutes after recommendation'
            },
            {
                'short_label': f'checkpoint2 ({cls.PRICE_CHECK_2_MINUTES}m)',
                'field_prefix': 'price_checkpoint2', 
                'minutes': cls.PRICE_CHECK_2_MINUTES,
                'description': f'Second price check {cls.PRICE_CHECK_2_MINUTES} minutes after recommendation'
            },
            {
                'short_label': f'close ({cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d})',
                'field_prefix': 'price_close',
                'minutes': None,  # Special case for close time
                'description': f'Market close price at {cls.CLOSE_PRICE_HOUR:02d}:{cls.CLOSE_PRICE_MINUTE:02d} EST'
            }
        ]
    
    # ================================================================
    # 📊 ENHANCED SYSTEM STATUS & REPORTING
    # ================================================================
    
    @classmethod
    def get_system_config_summary(cls) -> Dict[str, Any]:
        """Get comprehensive system configuration summary for logging/debugging"""
        return {
            'neural_analysis': {
                'enhanced_neural_enabled': cls.ENABLE_ENHANCED_NEURAL,
                'finbert_enabled': cls.ENABLE_FINBERT,
                'expected_accuracy': '94-96% with enhanced neural, 85-90% traditional'
            },
            'api_services': {
                'required': {
                    'fmp': bool(cls.FMP_API_KEY),
                },
                'optional': {
                    'gemini': cls.ENABLE_GEMINI and bool(cls.GEMINI_API_KEY),
                    'openai': cls.ENABLE_OPENAI and bool(cls.OPENAI_API_KEY),
                    'claude': cls.ENABLE_CLAUDE and bool(cls.CLAUDE_API_KEY),
                    'alpha_vantage': cls.ENABLE_ALPHA_VANTAGE and bool(cls.ALPHA_VANTAGE_API_KEY),
                    'polygon': cls.ENABLE_POLYGON and bool(cls.POLYGON_API_KEY),
                    'tiingo': cls.ENABLE_TIINGO and bool(cls.TIINGO_API_KEY)
                }
            },
            'processing_limits': {
                'max_articles': cls.MAX_NEWS_ARTICLES,
                'max_tickers': cls.MAX_TICKERS_TO_ANALYZE,
                'min_confidence': cls.MIN_CONFIDENCE_THRESHOLD,
                'cycle_interval_minutes': cls.CYCLE_INTERVAL_MINUTES
            },
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
                'earnings_events_enabled': cls.ENABLE_EARNINGS_EVENTS,
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
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_configuration(cls) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required API keys
        if not cls.FMP_API_KEY:
            issues.append("❌ FMP_API_KEY is required but not set")
        
        # Check if at least one analysis method is enabled
        analysis_methods = [
            cls.ENABLE_ENHANCED_NEURAL,
            cls.ENABLE_FINBERT,
            cls.ENABLE_GEMINI and bool(cls.GEMINI_API_KEY),
            cls.ENABLE_OPENAI and bool(cls.OPENAI_API_KEY),
            cls.ENABLE_CLAUDE and bool(cls.CLAUDE_API_KEY),
            cls.ENABLE_ALPHA_VANTAGE and bool(cls.ALPHA_VANTAGE_API_KEY),
            cls.ENABLE_POLYGON and bool(cls.POLYGON_API_KEY),
            cls.ENABLE_TIINGO and bool(cls.TIINGO_API_KEY),
            cls.ENABLE_KEYWORD_SENTIMENT
        ]
        
        if not any(analysis_methods):
            issues.append("⚠️ No analysis methods are enabled and configured")
        
        # Check price tracking intervals
        if cls.PRICE_CHECK_1_MINUTES >= cls.PRICE_CHECK_2_MINUTES:
            issues.append(f"⚠️ First checkpoint ({cls.PRICE_CHECK_1_MINUTES}m) should be before second checkpoint ({cls.PRICE_CHECK_2_MINUTES}m)")
        
        # Check fundamental filtering ranges
        if cls.MIN_STOCK_PRICE >= cls.MAX_STOCK_PRICE:
            issues.append(f"⚠️ MIN_STOCK_PRICE ({cls.MIN_STOCK_PRICE}) should be less than MAX_STOCK_PRICE ({cls.MAX_STOCK_PRICE})")
        
        # Check confidence threshold
        if not 0.1 <= cls.MIN_CONFIDENCE_THRESHOLD <= 0.95:
            issues.append(f"⚠️ MIN_CONFIDENCE_THRESHOLD ({cls.MIN_CONFIDENCE_THRESHOLD}) should be between 0.1 and 0.95")
        
        return issues

# ================================================================
# 🧠 ADAPTIVE LEARNING CONFIGURATION
# ================================================================
    ENABLE_ADAPTIVE_LEARNING: bool = os.getenv('ENABLE_ADAPTIVE_LEARNING', 'False').lower() == 'true'
    ADAPTIVE_LEARNING_MIN_TRADES: int = int(os.getenv('ADAPTIVE_LEARNING_MIN_TRADES', '10'))
    ADAPTIVE_LEARNING_CHECK_HOURS: int = int(os.getenv('ADAPTIVE_LEARNING_CHECK_HOURS', '6'))
    ADAPTIVE_LEARNING_BATCH_SIZE: int = int(os.getenv('ADAPTIVE_LEARNING_BATCH_SIZE', '8'))
    ADAPTIVE_LEARNING_LEARNING_RATE: float = float(os.getenv('ADAPTIVE_LEARNING_LEARNING_RATE', '2e-5'))
    ADAPTIVE_LEARNING_EPOCHS: int = int(os.getenv('ADAPTIVE_LEARNING_EPOCHS', '3'))

    # Thresholds for performance label creation in multi_modal_learning_system.py
    # These define what constitutes a 'poor', 'good', or 'excellent' trade based on 'price_close_change_pct'
    ADAPTIVE_PROFIT_THRESHOLD: float = float(os.getenv('ADAPTIVE_PROFIT_THRESHOLD', '0.0')) # Trades >= this are 'good' (label 1)
    ADAPTIVE_GOOD_TRADE_THRESHOLD: float = float(os.getenv('ADAPTIVE_GOOD_TRADE_THRESHOLD', '2.0')) # Trades >= this are 'excellent' (label 2)
                                                                                                   # Trades < ADAPTIVE_PROFIT_THRESHOLD are 'poor' (label 0)

    # Mapping from model output indices (0, 1, 2 from create_performance_labels)
    # to 'BUY', 'SELL', 'NEUTRAL' for the multi-modal model's inference.
    # This MUST align with the labeling logic above.
    ADAPTIVE_LABEL_MAP_POOR: str = os.getenv('ADAPTIVE_LABEL_MAP_POOR', 'SELL')         # Corresponds to label 0
    ADAPTIVE_LABEL_MAP_GOOD: str = os.getenv('ADAPTIVE_LABEL_MAP_GOOD', 'NEUTRAL')    # Corresponds to label 1
    ADAPTIVE_LABEL_MAP_EXCELLENT: str = os.getenv('ADAPTIVE_LABEL_MAP_EXCELLENT', 'BUY') # Corresponds to label 2


# ================================================================
# 📋 CONFIGURATION USAGE EXAMPLES & DOCUMENTATION
# ================================================================

CONFIG_USAGE_EXAMPLES = """
🔧 Enhanced Configuration Usage Examples:

⚡ PERFORMANCE OPTIMIZATION:
- Too slow? Reduce MAX_TICKERS_TO_ANALYZE to 50
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
