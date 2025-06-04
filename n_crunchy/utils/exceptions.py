class FMPAPIError(Exception):
    """Custom exception for FMP API errors."""
    pass

class LLMAPIError(Exception):
    """Custom exception for LLM API errors (e.g., rate limit, invalid response)."""
    pass

class TickerResolutionError(Exception):
    """Custom exception for errors in resolving company names to tickers."""
    pass

class DataProcessingError(Exception):
    """Custom exception for general data processing issues."""
    pass

class MissingConfigError(Exception):
    """Custom exception for missing or invalid configuration."""
    pass