# biomed_analyzer/analyzer/config.py
import os
import sys
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class ConfigError(Exception):
    """Raised when configuration is invalid"""
    pass

class Config:
    """Production-ready configuration management"""
    
    def __init__(self):
        self._validate_and_load()
    
    def _validate_and_load(self):
        """Validate and load all configuration"""
        # API Keys
        self.FINNHUB_API_KEY = self._get_api_key("FINNHUB_API_KEY", required=False)
        self.GEMINI_API_KEY = self._get_api_key("GEMINI_API_KEY", required=True)
        self.FMP_API_KEY = self._get_api_key("FMP_API_KEY", required=False)
        
        # Validate at least one news API is available
        if not self.FINNHUB_API_KEY and not self.FMP_API_KEY:
            raise ConfigError(
                "At least one news API key must be provided (FINNHUB_API_KEY or FMP_API_KEY). "
                "Please check your .env file."
            )
        
        # LLM Configuration
        self.DEFAULT_LLM = "gemini"
        self.LLM_REQUEST_TIMEOUT = 120
        
        # News Configuration
        self.FINNHUB_NEWS_COUNT = 100
        self.FMP_NEWS_COUNT = 50
        self.FMP_NEWS_PAGES = 2
        self.NEWS_LOOKBACK_DAYS = 2
        self.FDA_NEWS_WINDOW_PAST_DAYS = 7
        self.FDA_NEWS_WINDOW_FUTURE_DAYS = 3
        
        # Analysis Configuration
        self.MIN_CONFIDENCE_THRESHOLD = 5
        
        # Biomedical Classifications
        self.BIOMEDICAL_GICS_SECTORS = ["Health Care"]
        self.BIOMEDICAL_GICS_INDUSTRY_GROUPS = ["Pharmaceuticals, Biotechnology & Life Sciences"]
        self.BIOMEDICAL_GICS_INDUSTRIES = [
            "Biotechnology", "Pharmaceuticals", "Life Sciences Tools & Services", "Health Care Technology"
        ]
        
        # Research Keywords
        self.RESEARCH_NEWS_KEYWORDS = [
            "study", "trial", "clinical", "fda", "pdufa", "adcom", "advisory committee",
            "phase i", "phase 1", "phase ii", "phase 2", "phase iii", "phase 3",
            "results", "data", "findings", "breakthrough", "positive results", "negative results",
            "approval", "approved", "rejected", "rejection", "nda", "bla", "filing", "submitted",
            "investigational", "efficacy", "safety", "effective", "ineffective",
            "enrollment", "topline", "pivotal", "regulatory", "submission",
            "molecule", "compound", "therapeutic", "drug", "treatment", "vaccine", "biologic",
            "fast track", "orphan drug", "priority review", "breakthrough therapy designation"
        ]
    
    def _get_api_key(self, key_name: str, required: bool = True) -> Optional[str]:
        """Get and validate API key"""
        key = os.getenv(key_name)
        if not key:
            if required:
                raise ConfigError(f"Required API key {key_name} not found in environment variables")
            return None
        
        key = key.strip()
        if not key:
            if required:
                raise ConfigError(f"API key {key_name} is empty")
            return None
        
        # Basic validation - API keys should be alphanumeric and reasonable length
        if len(key) < 10:
            if required:
                raise ConfigError(f"API key {key_name} appears to be too short (less than 10 characters)")
            return None
        
        return key
    
    def has_finnhub(self) -> bool:
        """Check if Finnhub API is available"""
        return self.FINNHUB_API_KEY is not None
    
    def has_fmp(self) -> bool:
        """Check if FMP API is available"""
        return self.FMP_API_KEY is not None
    
    def has_gemini(self) -> bool:
        """Check if Gemini API is available"""
        return self.GEMINI_API_KEY is not None
    
    def print_config_status(self):
        """Print configuration status for debugging"""
        print("=== Configuration Status ===")
        print(f"Finnhub API: {'✓ Available' if self.has_finnhub() else '✗ Missing'}")
        print(f"FMP API: {'✓ Available' if self.has_fmp() else '✗ Missing'}")
        print(f"Gemini API: {'✓ Available' if self.has_gemini() else '✗ Missing'}")
        print(f"News sources: {sum([self.has_finnhub(), self.has_fmp()])} available")
        print("===========================")

# Create global config instance
try:
    config = Config()
except ConfigError as e:
    print(f"❌ Configuration Error: {e}")
    print("\nPlease ensure your .env file contains the required API keys:")
    print("GEMINI_API_KEY=your_gemini_key_here")
    print("FINNHUB_API_KEY=your_finnhub_key_here  # Optional if you have FMP")
    print("FMP_API_KEY=your_fmp_key_here          # Optional if you have Finnhub")
    sys.exit(1)

# Export commonly used values for backward compatibility
FINNHUB_API_KEY = config.FINNHUB_API_KEY
GEMINI_API_KEY = config.GEMINI_API_KEY
FMP_API_KEY = config.FMP_API_KEY
DEFAULT_LLM = config.DEFAULT_LLM
MIN_CONFIDENCE_THRESHOLD = config.MIN_CONFIDENCE_THRESHOLD
RESEARCH_NEWS_KEYWORDS = config.RESEARCH_NEWS_KEYWORDS
FINNHUB_NEWS_COUNT = config.FINNHUB_NEWS_COUNT
FMP_NEWS_COUNT = config.FMP_NEWS_COUNT
FMP_NEWS_PAGES = config.FMP_NEWS_PAGES
NEWS_LOOKBACK_DAYS = config.NEWS_LOOKBACK_DAYS
FDA_NEWS_WINDOW_PAST_DAYS = config.FDA_NEWS_WINDOW_PAST_DAYS
FDA_NEWS_WINDOW_FUTURE_DAYS = config.FDA_NEWS_WINDOW_FUTURE_DAYS
LLM_REQUEST_TIMEOUT = config.LLM_REQUEST_TIMEOUT
BIOMEDICAL_GICS_SECTORS = config.BIOMEDICAL_GICS_SECTORS
BIOMEDICAL_GICS_INDUSTRY_GROUPS = config.BIOMEDICAL_GICS_INDUSTRY_GROUPS
BIOMEDICAL_GICS_INDUSTRIES = config.BIOMEDICAL_GICS_INDUSTRIES