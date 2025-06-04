import logging.config
import logging
import hashlib
import re
from datetime import datetime
from functools import wraps
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import spacy
from config import LOGGING_CONFIG, NER_MODEL_NAME
from .exceptions import FMPAPIError, LLMAPIError, TickerResolutionError, DataProcessingError

# Configure logging using the dictionary from config.py
logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name):
    """Returns a logger instance with configured settings."""
    return logging.getLogger(name)

logger = get_logger(__name__)

# --- Decorators for API Retries ---
def retry_api_call(func):
    """
    Decorator to apply exponential backoff and retries for API calls.
    Retries on specific exceptions (FMPAPIError, LLMAPIError)
    """
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60), # Wait 1, 2, 4, 8, 16, 32, 60 seconds
        stop=stop_after_attempt(5), # Max 5 attempts
        retry=retry_if_exception_type((FMPAPIError, LLMAPIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True # Re-raise the last exception if all retries fail
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# --- NER Model Loading ---
_nlp = None # Global variable to hold the SpaCy model

def load_ner_model():
    """Loads the SpaCy NER model only once."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(NER_MODEL_NAME)
            logger.info(f"SpaCy NER model '{NER_MODEL_NAME}' loaded successfully.")
        except OSError:
            logger.error(f"SpaCy model '{NER_MODEL_NAME}' not found. Please run 'python -m spacy download {NER_MODEL_NAME}'.")
            raise TickerResolutionError(f"SpaCy model '{NER_MODEL_NAME}' not found. Please download it.")
    return _nlp

# --- Utility Functions ---

def extract_timestamp_from_fmp_news(article: dict) -> datetime | None:
    """
    Extracts a datetime object from an FMP news article.
    FMP news typically has a 'publishedDate' field.
    """
    date_str = article.get('publishedDate')
    if date_str:
        try:
            # FMP date format: "YYYY-MM-DD HH:MM:SS"
            return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
        try:
            # Another common FMP date format: "YYYY-MM-DDTHH:MM:SSZ" (sometimes just date part)
            return datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
        except ValueError:
            pass
    return None

def normalize_ticker(ticker_symbol: str) -> str:
    """Normalizes a ticker symbol (e.g., converts to uppercase)."""
    return ticker_symbol.strip().upper()

def generate_article_hash(article_title: str, article_url: str) -> str:
    """Generates a SHA256 hash for an article using its title and URL for deduplication."""
    combined_string = f"{article_title}-{article_url}"
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()

def clean_text_for_analysis(text: str) -> str:
    """Removes special characters and extra spaces from text."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags if any
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters, keep letters, numbers and spaces
    # Allowing some punctuation potentially important for financial context, but strict removal for keyword matching
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_float(value, default=0.0):
    """Safely converts a value to float, returning default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default