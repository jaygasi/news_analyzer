import re
import hashlib
from typing import Dict, Any, List, Optional
from unidecode import unidecode
import logging


def _normalize_name(name: str) -> str:
    """Cleans and standardizes a company name for matching using a more robust regex approach."""
    if not name:
        return ""
    # Convert to ASCII, lowercase
    name = unidecode(name).lower()

    # More robust suffix removal using a single regex
    # This pattern looks for common company suffixes at the end of the string.
    # It handles optional punctuation (like ".") and ensures it's a whole word match.
    # Suffixes are sorted by length descending to match longer ones first (e.g., 'corporation' before 'corp').
    suffixes = sorted([
        'corporation', 'limited', 'inc', 'corp', 'ltd', 'plc', 'lp', 'co', 'group', 
        'holdings', 'bancorp', 'financial', 'international', 'technologies', 
        'entertainment', 'properties', 'partners', 'company', 'enterprises',
        'systems', 'solutions', 'services', 'industries', 'communications'
    ], key=len, reverse=True)
    suffix_pattern = r'\s*\b(' + '|'.join(re.escape(s) for s in suffixes) + r')\.?$'
    name = re.sub(suffix_pattern, '', name, flags=re.IGNORECASE)

    # Replace punctuation with a space to preserve word boundaries
    name = re.sub(r'[^\w\s]', ' ', name)
    return ' '.join(name.split())


def generate_news_hash(news_item: Dict[str, Any]) -> str:
    """
    Generates a SHA256 hash for a news item to uniquely identify it.
    This is more reliable than using content snippets which can be similar.
    """
    # The symbol is now INCLUDED in the hash. This is critical for sources like
    # Alpha Vantage where one article can apply to multiple tickers. This ensures
    # that an article about AAPL and MSFT can be processed as two unique items.
    symbol = news_item.get('symbol', '')
    title = news_item.get('title', '')
    published_date = news_item.get('publishedDate', '')

    # Combine the most unique elements to create the identifier.
    # Including the symbol is key.
    identifier = f"{symbol}{title}{published_date}"
    return hashlib.sha256(identifier.encode('utf-8', 'ignore')).hexdigest()


def batch_process_items(items: List[Any], batch_size: int = 50) -> List[List[Any]]:
    """
    Splits a list of items into batches of specified size.
    Useful for processing large datasets in chunks to manage memory usage.
    """
    if not items:
        return []
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    logging.debug(f"Split {len(items)} items into {len(batches)} batches of size {batch_size}")
    return batches


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely converts a value to float, returning a default if conversion fails.
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert {value} to float, using default {default}")
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely converts a value to int, returning a default if conversion fails.
    """
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not convert {value} to int, using default {default}")
        return default


def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Attempts to extract a ticker symbol from text using common patterns.
    Returns the first match found, or None if no ticker is detected.
    """
    if not text:
        return None
    
    # Common ticker patterns: $AAPL, (NASDAQ: AAPL), AAPL:, etc.
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',  # $AAPL
        r'\((?:NYSE|NASDAQ|AMEX):\s*([A-Z]{1,5})\)',  # (NYSE: AAPL)
        r'\b([A-Z]{1,5}):\s',  # AAPL: 
        r'\b([A-Z]{2,5})\s+(?:stock|shares|equity)',  # AAPL stock
    ]
    
    for pattern in ticker_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ticker = match.group(1).upper()
            # Basic validation: should be 1-5 characters, all letters
            if 1 <= len(ticker) <= 5 and ticker.isalpha():
                return ticker
    
    return None


def clean_text_for_analysis(text: str) -> str:
    """
    Cleans text for sentiment analysis and feature extraction.
    Removes excessive whitespace, normalizes quotes, etc.
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r'["""]', "'", text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    return text.strip()


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Formats a decimal value as a percentage string.
    """
    try:
        return f"{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncates a string to a maximum length, adding a suffix if truncated.
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
