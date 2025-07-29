import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Core API Configuration ---
FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable")

# --- News Source Configuration ---
# Set to True to enable fetching from the respective source.
ENABLE_FMP_NEWS_SOURCE = True
ENABLE_ALPHA_VANTAGE_NEWS_SOURCE = True

# --- Alpha Vantage Specific Settings ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_DEFAULT_KEY")
ALPHA_VANTAGE_NEWS_LIMIT = 1000 # Max limit for Alpha Vantage is 1000
ALPHA_VANTAGE_NEWS_SORT = "LATEST" # Can be LATEST or EARLIEST

# --- Alpha Vantage Technical Indicator Settings ---
ALPHA_VANTAGE_RSI_PERIOD = 14
ALPHA_VANTAGE_SMA_PERIOD = 50
ALPHA_VANTAGE_TECHNICALS_INTERVAL = 'daily' # e.g., '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'

# --- Database ---
# Get the absolute path of the directory where this config file is located
_project_dir = os.path.dirname(os.path.abspath(__file__))
# Define the database path relative to this project directory
DATABASE_FILE = os.path.join(_project_dir, "system.db")

# --- API Request Handling ---
API_REQUEST_DELAY_SECONDS = 0.5 # General delay between requests. Adjust as needed.
API_MAX_RETRIES = 5 # Max retry attempts for failed API calls
API_RETRY_BACKOFF_FACTOR = 2 # Factor by which retry delay increases
API_INITIAL_RETRY_DELAY = 1 # Initial delay before first retry (seconds)

# FMP API Requests Per Minute Limit (Set this based on your FMP subscription!)
# For example, a free tier might be 5, paid could be 100, 300, etc.
FMP_API_REQUESTS_PER_MINUTE = 300 # <--- SET THIS VALUE ACCORDING TO YOUR PAID FMP PLAN!
NEWS_FETCH_LIMIT = 1000  # Number of latest news articles to fetch per run
MAX_PAGES_PER_SOURCE = 20 # Number of pages to fetch from each news source
NEWS_FETCH_TIME_WINDOW_HOURS = 24 # How many hours back to look for news.
MAX_EMPTY_RESPONSE_LOGS_PER_ENDPOINT = 5
MAX_TICKER_FILTERING_LOGS = 10 # Max "Filtering out due to fundamentals" messages per run

# --- Ticker Filtering Criteria ---
MIN_MARKET_CAP_USD = 100_000_000  # $100 Million
MIN_AVG_DAILY_VOLUME = 100_000     # 100k shares
MIN_NORMALIZED_NAME_LENGTH = 5 # Minimum length for a normalized company name to be considered valid
ALLOWED_EXCHANGES = [
    # Add common short names and full names for robustness.
    # The code performs a case-insensitive check.
    "NASDAQ",       # Nasdaq
    "NYSE",         # New York Stock Exchange
    "AMEX",         # NYSE American
    "NYSE American",
    "ARCA",         # NYSE Arca
    "NYSE Arca",
    "BATS"          # Cboe BZX
]

ENABLE_TICKER_PRE_VALIDATION = True

# --- Article Filtering Keywords with Weights ---
KEYWORD_WEIGHTS = {
    # General Positive Indicators
    "earnings beat": 1.5, "revenue growth": 1.2, "profit increase": 1.2, "strong outlook": 1.3,
    "guidance raised": 1.5, "new product launch": 1.0, "patent granted": 1.2, "strategic partnership": 1.1,
    "collaboration": 1.0, "acquisition": 1.1, "merger": 1.1, "expansion plans": 0.8, "innovation": 1.2,
    "breakthrough": 1.5, "upgraded rating": 1.4, "dividend increase": 1.0, "share buyback": 1.1,
    "analyst upgrade": 1.4, "exceeds estimates": 1.5, "debt reduction": 0.9, "licensing agreement": 1.1,
    "positive results": 1.3, "record sales": 1.4, "market share gain": 1.2, "production increase": 0.7,
    "successful trial": 1.6, "approval": 1.7,

    # General Negative Indicators
    "earnings miss": -1.6, "revenue decline": -1.3, "profit warning": -1.7, "weak outlook": -1.4,
    "guidance lowered": -1.6, "product recall": -1.2, "regulatory fine": -1.5, "investigation": -1.4,
    "lawsuit": -1.3, "data breach": -1.5, "cyber attack": -1.5, "leadership change": -0.8,
    "bankruptcy": -2.0, "insolvency": -2.0, "downgraded rating": -1.4, "analyst downgrade": -1.4,
    "misses estimates": -1.5, "supply chain disruption": -1.1, "competition concerns": -1.0,
    "debt default": -1.8, "strike": -0.7, "worker dispute": -0.6, "fraud investigation": -1.8,
    "recession fears": -1.0, "market downturn": -0.9, "production cut": -0.8, "demand drop": -1.1,
    "class action": -1.3, "warning letter": -1.2, "sanctions": -1.4,

    # Sector-Specific Keywords (Lower weights as they are more contextual)
    # -- Biotech / Pharma (Expanded & Organized) --
    # --- Positive Regulatory & Milestones ---
    "fda approval": 1.8, "ema approval": 1.8, "marketing authorization": 1.8, "accelerated approval": 1.7,
    "breakthrough therapy designation": 1.6, "priority review": 1.4, "fast track designation": 1.3,
    "orphan drug designation": 1.2, "new drug application": 0.6, "nda": 0.6, "biologics license application": 0.6,
    "bla": 0.6, "pdufa date": 0.6, "ind filing": 0.5, "investigational new drug": 0.5, "upfront payment": 1.0,
    "milestone payment": 0.8,

    # --- Positive Trial Results & Data ---
    "statistically significant": 1.5, "p-value <": 1.5, "p < 0.05": 1.5, "clinically meaningful": 1.6,
    "positive data": 1.4, "positive topline": 0.9, "positive top-line": 0.9, "durable response": 1.3,
    "favorable safety profile": 1.1, "well-tolerated": 1.0, "good safety profile": 1.0, "overall survival": 1.0,
    "os": 1.0, "progression-free survival": 1.0, "pfs": 1.0, "topline results": 0.7, "top-line results": 0.7,

    # --- Trial Phases (with variations) ---
    "phase 3": 0.8, "phase iii": 0.8, "phase three": 0.8, "phase 2": 0.7, "phase ii": 0.7, "phase two": 0.7,
    "phase 1": 0.4, "phase i": 0.4, "phase one": 0.4, "clinical trial": 0.5,

    # --- Negative Outcomes & Setbacks ---
    "black box warning": -1.9, "patient death": -1.8, "failed trial": -1.7, "trial discontinuation": -1.7,
    "clinical hold": -1.6, "refusal to file": -1.6, "no better than placebo": -1.6,
    "complete response letter": -1.5, "crl": -1.5, "not statistically significant": -1.5,
    "serious adverse event": -1.5, "sae": -1.5, "safety concerns": -1.4, "adverse event": -1.2,
    "side effects": -1.1, "form 483": -1.1,

    # -- Energy --
    "oil price": 0.3, "gas price": 0.3, "opec": 0.4,
    # -- Finance --
    "interest rates": 0.2, "fintech": 0.6,
    # -- Tech --
    "artificial intelligence": 0.7, "ai": 0.7, "cloud computing": 0.5, "cybersecurity": 0.8
}

# --- Complex Keyword Rules for Contextual Matching ---
# These rules look for combinations of keywords within the same sentence.
COMPLEX_KEYWORD_RULES = {
    # --- Positive Biotech/Pharma Outcomes ---
    "positive_trial_endpoint": {
        "weight": 1.8,
        "sets": [
            ["met", "meets", "achieved"],
            ["primary endpoint", "secondary endpoint", "all primary endpoints"]
        ],
        "logic": "AND"  # Requires one from each set in the same sentence
    },
    "positive_survival_data": {
        "weight": 1.7,
        "sets": [
            ["prolonged", "improved", "superior", "significant", "meaningful"],
            ["progression-free survival", "pfs", "overall survival", "os", "durable response"]
        ],
        "logic": "AND"
    },

    # --- Negative Biotech/Pharma Outcomes ---
    "negative_trial_endpoint": {
        "weight": -1.8,
        "sets": [
            ["fails to meet", "did not meet", "failed to meet"],
            ["primary endpoint", "secondary endpoint"]
        ],
        "logic": "AND"
    },
}


# --- Keyword Evaluator Configuration ---
# The column in the 'trades' table used for keyword performance evaluation.
# This should be one of the 'perf_X_pct' columns from price_tracker.py.
KEYWORD_EVAL_PERFORMANCE_COLUMN = 'perf_eod_pct' # Using End-Of-Day performance for evaluation
KEYWORD_EVAL_MIN_TRADES_FOR_STATS = 5 # Minimum number of trades for a keyword to be statistically evaluated
KEYWORD_EVAL_POSITIVE_PERF_THRESHOLD = 1.0 # Percentage threshold for a keyword's average performance to be considered positive
KEYWORD_EVAL_NEGATIVE_PERF_THRESHOLD = -1.0 # Percentage threshold for a keyword's average performance to be considered negative
KEYWORD_EVAL_AI_EXPLANATION_THRESHOLD = 0.2 # Ratio of AI explanations mentioning a keyword for it to be considered 'AI-favored'
KEYWORD_WEIGHT_ADJUSTMENT_FACTOR = 0.1 # Factor by which keyword weights are adjusted during automated evaluation

# Performance checkpoints in minutes for intraday tracking.
PERFORMANCE_CHECKPOINTS = {"30_min": 30, "60_min": 60, "240_min": 240}
