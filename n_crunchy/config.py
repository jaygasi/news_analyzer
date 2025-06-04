import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
# FIX: Correct import path for JsonFormatter
from pythonjsonlogger.jsonlogger import JsonFormatter 

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
FMP_API_KEY = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "YOUR_CLAUDE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY", "YOUR_GROK_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "YOUR_LLAMA_API_KEY")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "YOUR_TIINGO_API_KEY")

# --- Application Settings ---
NEWS_AGE_LIMIT_DAYS = 7
DECISION_CONFIDENCE_THRESHOLD = 0.75
WAIT_TIME_SECONDS = 300
MAX_CONCURRENT_API_CALLS = 5

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
LOG_DIR = os.path.join(BASE_DIR, "logs")

CSV_LOG_FILE = os.path.join(OUTPUT_DIR, "paper_trading_decisions.csv")
ARTICLE_TRACKER_DB = os.path.join(DATABASE_DIR, "articles.db")
APPLICATION_LOG_FILE = os.path.join(LOG_DIR, "app.log")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Configuration ---
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JsonFormatter,
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
        },
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": APPLICATION_LOG_FILE,
            "maxBytes": 10485760,
            "backupCount": 5,
            "level": "INFO"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
    "loggers": {
        "crunchy": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# --- Directional Keywords for News Analysis ---
DIRECTIONAL_KEYWORDS = {
    "positive": {
        "earnings beat": 0.8, "revenue surge": 0.7, "partnership announced": 0.6,
        "fda approval": 0.9, "new contract": 0.6, "strong outlook": 0.5,
        "acquisition": 0.7, "expansion": 0.5, "upgrade": 0.4,
        "record high": 0.7, "product launch": 0.6, "innovation": 0.5,
        "positive analyst report": 0.6, "dividend increase": 0.4,
        "stock buyback": 0.5, "patent granted": 0.5,
    },
    "negative": {
        "earnings miss": -0.8, "revenue decline": -0.7, "lawsuit": -0.7,
        "regulatory probe": -0.8, "bankruptcy": -0.9, "downgrade": -0.4,
        "recall": -0.6, "investigation": -0.7, "debt burden": -0.5,
        "labor dispute": -0.5, "supply chain disruption": -0.6,
        "cybersecurity breach": -0.7, "guidance cut": -0.6,
        "antitrust": -0.7, "management changes": -0.5, "market share loss": -0.5,
    },
    "neutral": {
        "restructuring": 0.1, "board meeting": 0.0, "analyst day": 0.0,
        "q3 results": 0.0, "annual report": 0.0,
    }
}

# --- FMP API Endpoints ---
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
FMP_ENDPOINTS = {
    "stock_news": "/stock_news",
    "press_releases": "/press-releases",
    "earnings_call_transcript": "/earning-call-transcript",
    "analyst_estimates": "/analyst-estimates",
    "historical_price": "/historical-price/{symbol}",
    "technical_indicator": "/technical_indicator/{symbol}",
    "quote": "/quote/{symbol}",
    "stock_list": "/stock/list"
}

# --- Technical Analysis Settings (Default, can be dynamic) ---
DEFAULT_TECHNICAL_INDICATORS_SETTINGS = {
    "RSI_PERIOD": 14,
    "MACD_FAST_PERIOD": 12,
    "MACD_SLOW_PERIOD": 26,
    "MACD_SIGNAL_PERIOD": 9,
    "BOLLINGER_BANDS_PERIOD": 20,
    "BOLLINGER_BANDS_DEV": 2,
    "MOVING_AVERAGE_SHORT": 10,
    "MOVING_AVERAGE_LONG": 50,
    "DATA_LOOKBACK_DAYS": 250
}

# --- Named Entity Recognition (NER) Settings ---
NER_MODEL_NAME = "en_core_web_sm"

# --- LLM Prompting ---
LLM_PROMPT_TEMPLATE = """
Analyze the following financial news article for {ticker} and determine its likely directional impact on the stock price (LONG, SHORT, or NEUTRAL).
Focus solely on the article's implications for stock movement, not general sentiment.
Provide a confidence score (0.0 to 1.0) for your prediction and a brief, concise reason (max 50 words).

Article Title: {title}
Article Content:
---
{content}
---

Your response MUST be a JSON object with the following structure:
{{
    "direction": "[LONG/SHORT/NEUTRAL]",
    "confidence": [0.0-1.0],
    "reason": "[brief explanation]"
}}
"""