"""
Optimized string utilities with improved performance and error handling
"""
import re
import unicodedata
from typing import List, Union, Optional
from functools import lru_cache

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    LangDetectException = Exception


def clean_string(s: Optional[str]) -> str:
    """
    Clean string by removing non-ASCII characters and trimming whitespace.
    
    Args:
        s: Input string to clean
        
    Returns:
        Cleaned string, empty string if input is None/empty
    """
    if not s:
        return ""
    
    # More efficient: normalize unicode first, then encode/decode
    try:
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', str(s))
        # Remove non-ASCII and decode
        ascii_string = normalized.encode('ascii', 'ignore').decode('ascii')
        return ascii_string.strip()
    except (UnicodeError, AttributeError):
        return ""


def join_items(item_list: List[str], separator: str = ",") -> str:
    """
    Join list items with specified separator.
    
    Args:
        item_list: List of strings to join
        separator: Separator character (default: comma)
        
    Returns:
        Joined string
    """
    if not item_list:
        return ""
    
    # Filter out None/empty values and convert to strings
    valid_items = [str(item).strip() for item in item_list if item is not None]
    return separator.join(valid_items)


def strtobool(val: Union[str, bool, int]) -> bool:
    """
    Convert string representation of truth to boolean.
    
    Args:
        val: Value to convert (string, bool, or int)
        
    Returns:
        Boolean value
        
    Raises:
        ValueError: If value cannot be converted to boolean
    """
    if isinstance(val, bool):
        return val
    
    if isinstance(val, int):
        return bool(val)
    
    if not isinstance(val, str):
        raise ValueError(f"Invalid type for boolean conversion: {type(val)}")
    
    val_lower = val.lower().strip()
    
    if val_lower in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val_lower in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"Invalid truth value: '{val}'")


def language_detection(text: str, method: str = "single") -> Union[str, List]:
    """
    Detect language of text with fallback handling.
    
    Args:
        text: Text to analyze
        method: Detection method ("single" or "multiple")
        
    Returns:
        Language code(s) or "unknown" if detection fails
    """
    if not LANGDETECT_AVAILABLE:
        return "unknown" if method == "single" else [("unknown", 1.0)]
    
    if not text or not text.strip():
        return "unknown" if method == "single" else [("unknown", 1.0)]
    
    try:
        if method.lower() != "single":
            return detect_langs(text)
        else:
            return detect(text)
    except (LangDetectException, Exception):
        return "unknown" if method == "single" else [("unknown", 1.0)]


def clean_text(text: str) -> str:
    """
    Clean text by trimming and ensuring proper sentence ending.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with proper sentence ending
    """
    if not text:
        return ""
    
    text = text.strip()
    if not text:
        return text
    
    # Find last sentence-ending punctuation
    last_punct_index = max(
        text.rfind('.'),
        text.rfind('!'),
        text.rfind('?')
    )
    
    if last_punct_index > 0:
        # Cut off text after last sentence-ending punctuation
        output = text[:last_punct_index + 1]
    else:
        # No sentence-ending punctuation found, add period
        output = text.rstrip('.!?') + '.'
    
    return output


@lru_cache(maxsize=256)
def camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case with caching for performance.
    
    Args:
        name: CamelCase string
        
    Returns:
        snake_case string
    """
    if not name:
        return ""
    
    # More efficient regex approach
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def capitalize_first_word(sentence: str) -> str:
    """
    Capitalize first word of each sentence with improved regex.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Sentence with properly capitalized first words
    """
    if not sentence:
        return ""
    
    # Convert to lowercase first
    lower_sentence = sentence.lower()
    
    # Capitalize first letter of the entire string
    if lower_sentence:
        lower_sentence = lower_sentence[0].upper() + lower_sentence[1:]
    
    # Capitalize first letter after sentence-ending punctuation followed by space
    capitalized = re.sub(
        r'([.!?]\s+)([a-z])', 
        lambda m: m.group(1) + m.group(2).upper(), 
        lower_sentence
    )
    
    return capitalized


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text with suffix if needed
    """
    if not text or max_length <= 0:
        return ""
    
    if len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace in text to single spaces.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return ""
    
    # Replace all whitespace sequences with single spaces
    return re.sub(r'\s+', ' ', text.strip())


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of numbers found in text
    """
    if not text:
        return []
    
    # Pattern to match integers and floats
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            # Try to convert to float, then to int if it's a whole number
            num = float(match)
            if num.is_integer():
                numbers.append(int(num))
            else:
                numbers.append(num)
        except ValueError:
            continue
    
    return numbers