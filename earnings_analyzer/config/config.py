import os
from dotenv import load_dotenv
from typing import List, Dict, Set

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for Earnings News Trader"""

    # ===== GENERAL APPLICATION SETTINGS =====
    USER_AGENT = os.getenv('USER_AGENT', 'EarningsNewsTrader/1.0')
    SCHEDULER_SLEEP_INTERVAL_SECONDS = int(os.getenv('SCHEDULER_SLEEP_INTERVAL_SECONDS', '60'))

    # ===== DATABASE CONFIGURATION =====
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///earnings_trader.db')

    # ===== LOGGING CONFIGURATION =====
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # ===== LLM PROVIDER CONFIGURATION =====
    USE_CLAUDE = os.getenv('USE_CLAUDE', 'true').lower() == 'true'
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-3-sonnet-20240229')

    USE_OPENAI = os.getenv('USE_OPENAI', 'false').lower() == 'true'
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')

    USE_GEMINI = os.getenv('USE_GEMINI', 'true').lower() == 'true'
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.0-pro')

    USE_GROK = os.getenv('USE_GROK', 'false').lower() == 'true'
    GROK_API_KEY = os.getenv('GROK_API_KEY', '')
    GROK_MODEL = os.getenv('GROK_MODEL', 'grok-1')

    LLM_PRIORITY_ORDER: List[str] = ['gemini', 'claude', 'openai', 'grok']

    # ===== SENTIMENT ANALYSIS & SIGNAL PROCESSING CONFIGURATION =====
    USE_TEXTBLOB_ANALYSIS = os.getenv('USE_TEXTBLOB_ANALYSIS', 'true').lower() == 'true'
    USE_VADER_ANALYSIS = os.getenv('USE_VADER_ANALYSIS', 'true').lower() == 'true'

    POSITIVE_SENTIMENT_INDICATORS: List[str] = [
        'beat', 'beats', 'exceeded', 'strong', 'growth', 'up', 'higher',
        'positive', 'good', 'better', 'improvement', 'gains', 'surge', 'outperform'
    ]
    NEGATIVE_SENTIMENT_INDICATORS: List[str] = [
        'miss', 'missed', 'below', 'weak', 'decline', 'down', 'lower',
        'negative', 'bad', 'worse', 'drop', 'fall', 'loss', 'underperform'
    ]
    MAX_KEY_PHRASES_QUICK_SENTIMENT: int = int(os.getenv('MAX_KEY_PHRASES_QUICK_SENTIMENT', '3'))
    MAX_ARTICLES_PER_RUN: int = int(os.getenv('MAX_ARTICLES_PER_RUN', '50'))

    # ===== TRADING SIGNAL CONFIGURATION =====
    MIN_CONFIDENCE_SCORE = float(os.getenv('MIN_CONFIDENCE_SCORE', '0.7'))
    URGENT_SIGNAL_CONFIDENCE_THRESHOLD = float(os.getenv('URGENT_SIGNAL_CONFIDENCE_THRESHOLD', '0.8'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.05'))

    # ===== NEWS COLLECTION CONFIGURATION =====
    NEWS_COLLECTION_DAYS_BACK = int(os.getenv('NEWS_COLLECTION_DAYS_BACK', '3'))
    OPERATIONAL_EARNINGS_DAYS_AHEAD: int = int(os.getenv('OPERATIONAL_EARNINGS_DAYS_AHEAD', '7'))
    COLLECT_GENERAL_NEWS_IF_NO_TARGETS: bool = os.getenv('COLLECT_GENERAL_NEWS_IF_NO_TARGETS', 'false').lower() == 'true'
    DEFAULT_FALLBACK_TICKERS_IF_NO_EARNINGS: List[str] = ['SPY', 'QQQ', 'DIA']

    MAX_ARTICLE_AGE_DAYS: int = int(os.getenv('MAX_ARTICLE_AGE_DAYS', '7'))
    MAX_ARTICLES_PER_RSS_FEED: int = int(os.getenv('MAX_ARTICLES_PER_RSS_FEED', '10'))
    RSS_FEED_TIMEOUT_SECONDS: int = int(os.getenv('RSS_FEED_TIMEOUT_SECONDS', '10'))
    ARTICLE_FETCH_TIMEOUT_SECONDS: int = int(os.getenv('ARTICLE_FETCH_TIMEOUT_SECONDS', '15'))
    RSS_INTER_FEED_DELAY_SECONDS: float = float(os.getenv('RSS_INTER_FEED_DELAY_SECONDS', '0.5'))

    MIN_FULL_ARTICLE_LENGTH_THRESHOLD: int = int(os.getenv('MIN_FULL_ARTICLE_LENGTH_THRESHOLD', '300'))
    MAX_ARTICLE_CONTENT_LENGTH: int = int(os.getenv('MAX_ARTICLE_CONTENT_LENGTH', '10000'))

    HTML_NOISE_SELECTORS: List[str] = [
        'script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe',
        '.ad', '.ads', '.advertisement', '.sidebar', '.popup', '.cookie-banner'
    ]
    HTML_CONTENT_SELECTORS: List[str] = [
        'article .article-content', 'article .article-body', 'article .story-content',
        'article', '.article-content', '.article-body', '.story-content',
        '.entry-content', '.post-content', '.content', 'main .content-area',
        'main .main-content', '.article-text', 'div[role="main"]', 'main'
    ]

    EARNINGS_KEYWORDS: Dict[str, int] = {
        'earnings': 3, 'quarterly results': 3, 'eps': 2, 'earnings report': 3,
        'earnings call': 2, 'revenue': 2, 'profit': 2, 'quarterly': 1,
        'beat estimates': 2, 'miss estimates': 2, 'guidance': 2, 'forecast': 1, 'outlook': 1,
        'financial results': 2, 'beats expectations': 2, 'misses expectations': 2,
        'consensus': 1, 'analyst estimates': 1, 'forward guidance': 2,
        'same-store sales': 1, 'comparable sales': 1, 'gross margin': 1,
        'operating margin': 1, 'ebitda': 1, 'free cash flow': 1, 'dividend': 1,
        'share buyback': 1, 'conference call': 2
    }
    EARNINGS_PRIORITY_KEYWORD_THRESHOLD: int = int(os.getenv('EARNINGS_PRIORITY_KEYWORD_THRESHOLD', '3'))
    EARNINGS_TICKER_MATCH_BONUS: int = int(os.getenv('EARNINGS_TICKER_MATCH_BONUS', '2'))
    MAX_EARNINGS_RELEVANCE_SCORE: float = float(os.getenv('MAX_EARNINGS_RELEVANCE_SCORE', '15.0'))
    MIN_EARNINGS_RELEVANCE_SCORE_THRESHOLD: float = float(os.getenv('MIN_EARNINGS_RELEVANCE_SCORE_THRESHOLD', '3.0'))
    MIN_EARNINGS_CONFIDENCE_TO_COLLECT: float = float(os.getenv('MIN_EARNINGS_CONFIDENCE_TO_COLLECT', '0.3'))

    RSS_FEEDS: Dict[str, Dict[str, str]] = {
        'Yahoo Finance Top Stories': {
            'url': 'https://finance.yahoo.com/rss/topstories', 'priority': 'high'
        },
        'Yahoo Finance Company News': {
            'url': 'https://finance.yahoo.com/rss/company-news', 'priority': 'high'
        },
        'MarketWatch Top Stories': {
            'url': 'http://feeds.marketwatch.com/marketwatch/topstories/', 'priority': 'high'
        },
        'Investing.com News': {
            'url': 'https://www.investing.com/rss/news.rss', 'priority': 'high'
        },
        'CNBC Top News': {
             'url': 'https://www.cnbc.com/id/100003114/device/rss/rss.html', 'priority': 'high'
        },
        'Reuters Business': {
            'url': 'http://feeds.reuters.com/reuters/businessNews', 'priority': 'medium'
        },
        'Google News - Business (US)': {
            'url': 'https://news.google.com/rss/search?q=when:24h+allinurl:business&hl=en-US&gl=US&ceid=US:en', 'priority': 'low'
        },
    }
    RSS_PRIORITY_ORDER: Dict[str, int] = {'high': 1, 'medium': 2, 'low': 3}

    COMPANY_NAME_TO_TICKER_MAP: Dict[str, str] = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta platforms': 'META', 'facebook': 'META',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'salesforce': 'CRM', 'intel': 'INTC',
        'advanced micro devices': 'AMD', 'urban outfitters': 'URBN',
    }
    TICKER_EXCLUSION_LIST: Set[str] = {
        'CEO', 'CFO', 'COO', 'CTO', 'EPS', 'IPO', 'ETF', 'SEC', 'FED', 'FOMC',
        'USA', 'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'API', 'JSON', 'XML', 'HTML',
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'NEW',
        'NOW', 'GET', 'HAS', 'ITS', 'TWO', 'WHO', 'OIL', 'GAS', 'BIG', 'TOP',
        'BUY', 'SELL', 'HOLD', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'MON', 'TUE', 'WED', 'THU', 'FRI',
        'AM', 'PM', 'EST', 'PST', 'GMT', 'UTC', 'NEWS', 'TECH', 'BIZ'
    }

    # ===== EXTERNAL API KEYS & CONFIG =====
    FMP_API_KEY = os.getenv('FMP_API_KEY', '')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY', '')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')

    EARNINGS_API_PRIORITY: List[str] = ['finnhub', 'fmp', 'polygon', 'tiingo', 'alpha_vantage']
    API_TIMEOUT_SECONDS: int = int(os.getenv('API_TIMEOUT_SECONDS', '15'))
    API_CALL_DELAY_SECONDS: float = float(os.getenv('API_CALL_DELAY_SECONDS', '1.0'))
    MAX_EARNINGS_EVENTS_FROM_APIS: int = int(os.getenv('MAX_EARNINGS_EVENTS_FROM_APIS', '100'))
    MIN_EARNINGS_FOR_YAHOO_FALLBACK: int = int(os.getenv('MIN_EARNINGS_FOR_YAHOO_FALLBACK', '10'))
    YAHOO_FALLBACK_TICKERS_STR = os.getenv('YAHOO_FALLBACK_TICKERS', 'AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,JNJ,WMT,PG')
    YAHOO_FALLBACK_TICKERS: List[str] = [ticker.strip() for ticker in YAHOO_FALLBACK_TICKERS_STR.split(',') if ticker.strip()] if YAHOO_FALLBACK_TICKERS_STR else []
    MAX_YAHOO_TICKERS_TO_CHECK: int = int(os.getenv('MAX_YAHOO_TICKERS_TO_CHECK', '50'))
    YAHOO_API_CALL_DELAY_SECONDS: float = float(os.getenv('YAHOO_API_CALL_DELAY_SECONDS', '0.5'))

    # ===== EMAIL NOTIFICATION CONFIGURATION =====
    GMAIL_EMAIL = os.getenv('GMAIL_EMAIL', '')
    GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD', '')
    ALERT_EMAIL = os.getenv('ALERT_EMAIL', GMAIL_EMAIL)

    # ===== TESTING & DEBUGGING CONFIGURATION =====
    MAJOR_TICKERS: List[str] = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'V', 'JNJ',
        'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'DIS', 'BA', 'CAT', 'WMT', 'COST', 'HD',
    ]
    DEFAULT_TEST_TICKERS: List[str] = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
    MAX_TEST_EARNINGS_TO_DISPLAY: int = int(os.getenv('MAX_TEST_EARNINGS_TO_DISPLAY', '10'))
    MAX_TEST_ARTICLES_TO_DISPLAY_PER_TICKER: int = int(os.getenv('MAX_TEST_ARTICLES_TO_DISPLAY_PER_TICKER', '2'))
    NUM_TOP_TICKERS_FOR_STATS: int = int(os.getenv('NUM_TOP_TICKERS_FOR_STATS', '5'))

    # ===== TRADING EXECUTION CONFIGURATION =====
    MAX_DAILY_SIGNALS = int(os.getenv('MAX_DAILY_SIGNALS', '10'))
    MIN_MARKET_CAP = float(os.getenv('MIN_MARKET_CAP', '1000000000'))
    HIGH_CONFIDENCE_MULTIPLIER = float(os.getenv('HIGH_CONFIDENCE_MULTIPLIER', '1.5'))
    LOW_CONFIDENCE_MULTIPLIER = float(os.getenv('LOW_CONFIDENCE_MULTIPLIER', '0.5'))
    DEFAULT_STOP_LOSS_PCT = float(os.getenv('DEFAULT_STOP_LOSS_PCT', '0.08'))
    DEFAULT_TAKE_PROFIT_PCT = float(os.getenv('DEFAULT_TAKE_PROFIT_PCT', '0.15'))

    @classmethod
    def validate(cls):
        """Validate configuration and return status"""
        issues = []
        llm_providers_configured = []
        
        # Check LLM providers
        if cls.USE_CLAUDE:
            if cls.ANTHROPIC_API_KEY: 
                llm_providers_configured.append('Claude')
            else: 
                issues.append("Claude enabled (USE_CLAUDE=true) but ANTHROPIC_API_KEY missing.")
        
        if cls.USE_OPENAI:
            if cls.OPENAI_API_KEY: 
                llm_providers_configured.append('OpenAI')
            else: 
                issues.append("OpenAI enabled (USE_OPENAI=true) but OPENAI_API_KEY missing.")
        
        if cls.USE_GEMINI:
            if cls.GOOGLE_API_KEY: 
                llm_providers_configured.append('Gemini')
            else: 
                issues.append("Gemini enabled (USE_GEMINI=true) but GOOGLE_API_KEY missing.")
        
        if cls.USE_GROK:
            if cls.GROK_API_KEY: 
                llm_providers_configured.append('Grok')
            else: 
                issues.append("Grok enabled (USE_GROK=true) but GROK_API_KEY missing.")

        # Check if any LLM provider is available
        if not llm_providers_configured and (cls.USE_CLAUDE or cls.USE_OPENAI or cls.USE_GEMINI or cls.USE_GROK):
            issues.append("LLM providers are enabled, but no corresponding API keys are found.")
        elif not (cls.USE_CLAUDE or cls.USE_OPENAI or cls.USE_GEMINI or cls.USE_GROK):
             issues.append("No LLM providers are enabled. Sentiment analysis will be limited.")

        # Check email configuration
        if not cls.GMAIL_EMAIL or not cls.GMAIL_APP_PASSWORD:
            issues.append("GMAIL_EMAIL or GMAIL_APP_PASSWORD missing - notifications will not work.")

        # Check earnings APIs
        earnings_apis_configured = []
        api_map = {
            'finnhub': cls.FINNHUB_API_KEY, 
            'fmp': cls.FMP_API_KEY, 
            'polygon': cls.POLYGON_API_KEY,
            'tiingo': cls.TIINGO_API_KEY, 
            'alpha_vantage': cls.ALPHA_VANTAGE_API_KEY
        }
        
        for api_name in cls.EARNINGS_API_PRIORITY:
            if api_map.get(api_name):
                earnings_apis_configured.append(api_name.capitalize())
        
        if not earnings_apis_configured:
            issues.append("No earnings data APIs configured with API keys. Earnings data will be limited.")

        # Validate ranges
        if not (0 <= cls.MIN_CONFIDENCE_SCORE <= 1):
            issues.append(f"MIN_CONFIDENCE_SCORE ({cls.MIN_CONFIDENCE_SCORE}) must be between 0 and 1.")
        if not (0 < cls.MAX_POSITION_SIZE <= 1):
            issues.append(f"MAX_POSITION_SIZE ({cls.MAX_POSITION_SIZE}) must be > 0 and <= 1.")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'llm_providers_active': llm_providers_configured,
            'email_configured': bool(cls.GMAIL_EMAIL and cls.GMAIL_APP_PASSWORD),
            'earnings_apis_active': earnings_apis_configured
        }

    @classmethod
    def get_summary(cls):
        """Get configuration summary"""
        validation_results = cls.validate()
        return {
            'database_type': 'SQLite' if 'sqlite' in cls.DATABASE_URL else 'PostgreSQL/Other',
            'log_level': cls.LOG_LEVEL,
            'active_llm_providers': validation_results['llm_providers_active'],
            'email_notifications_setup': validation_results['email_configured'],
            'active_earnings_apis': validation_results['earnings_apis_active'],
            'critical_issues_found': len(validation_results['issues']) > 0
        }

    @classmethod
    def get_earnings_api_config(cls) -> dict:
        """Get earnings API configuration"""
        available = {
            'finnhub': bool(cls.FINNHUB_API_KEY),
            'fmp': bool(cls.FMP_API_KEY),
            'polygon': bool(cls.POLYGON_API_KEY),
            'tiingo': bool(cls.TIINGO_API_KEY),
            'alpha_vantage': bool(cls.ALPHA_VANTAGE_API_KEY),
            'yahoo': True  # Yahoo is always available as fallback
        }
        
        active_priority_order = [api for api in cls.EARNINGS_API_PRIORITY if available.get(api, False)]
        
        # Ensure Yahoo is included as fallback if no other APIs are available
        if not active_priority_order:
            active_priority_order = ['yahoo']
        
        return {
            'priority_order': active_priority_order,
            'available_apis': available
        }
