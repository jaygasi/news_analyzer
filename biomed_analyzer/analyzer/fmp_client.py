# biomed_analyzer/analyzer/fmp_client.py
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
import re

from .config import (
    FMP_API_KEY, FMP_NEWS_COUNT, FMP_NEWS_PAGES, NEWS_LOOKBACK_DAYS,
    BIOMEDICAL_GICS_INDUSTRIES, BIOMEDICAL_GICS_INDUSTRY_GROUPS, BIOMEDICAL_GICS_SECTORS
)
from .utils import logger

# Define regex constant at module level
VALID_SYMBOL_REGEX = r'^[A-Z0-9\.]{1,10}$'

class FMPClient:
    def __init__(self, api_key: Optional[str] = FMP_API_KEY):
        if not api_key:
            logger.error("FMP API key not found. Please set FMP_API_KEY environment variable.")
            raise ValueError("FMP API key is required.")
        
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api"
        
        # Test API key with a simple request
        try:
            test_response = requests.get(
                f"{self.base_url}/v3/stock/list?apikey={self.api_key}",
                timeout=10,
                params={'limit': 1}
            )
            if test_response.status_code == 401:
                raise ValueError("FMP API key is invalid or unauthorized")
            elif test_response.status_code != 200:
                raise ValueError(f"FMP API returned status code: {test_response.status_code}")
            
            logger.info("FMP client initialized and API key validated.")
        except requests.RequestException as e:
            logger.error(f"Failed to validate FMP API key: {e}")
            raise ValueError(f"FMP API key validation failed: {e}") from e
        
        self._company_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._biomed_symbols_cache: Optional[Set[str]] = None

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Make a request to FMP API with error handling."""
        if params is None:
            params = {}
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            time.sleep(0.12)  # Rate limiting: ~8 requests per second
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("FMP API rate limit hit, waiting...")
                time.sleep(1)
                return None
            else:
                logger.error(f"FMP API error: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request error for FMP endpoint {endpoint}: {e}")
            return None

    def _fetch_and_cache_biomed_symbols(self) -> Set[str]:
        """Fetch and cache biomedical symbols from FMP."""
        logger.info("Fetching biomedical symbols from FMP...")
        symbols_set: Set[str] = set()
        
        try:
            # Get all US stocks with company profiles
            stocks_data = self._make_request("/v3/stock/list")
            if not stocks_data:
                self._biomed_symbols_cache = symbols_set
                return symbols_set
            
            processed_count = 0
            biomed_count = 0
            
            for stock in stocks_data:
                processed_count += 1
                symbol = stock.get('symbol', '').strip().upper()
                
                if not symbol or not re.match(VALID_SYMBOL_REGEX, symbol):
                    continue
                
                # Filter to US exchanges and common stock types
                exchange = stock.get('exchangeShortName', '').upper()
                stock_type = stock.get('type', '').lower()
                
                if exchange not in ['NASDAQ', 'NYSE', 'AMEX'] or stock_type not in ['stock', 'common stock', '']:
                    continue
                
                # Check if it's biomedical by getting company profile
                profile = self.get_company_profile(symbol)
                if profile and self._is_biomedical_from_profile(profile):
                    symbols_set.add(symbol)
                    biomed_count += 1
                
                # Limit processing to avoid excessive API calls
                if processed_count >= 1000:  # Reasonable limit for caching
                    break
            
            logger.info(f"Processed {processed_count} symbols from FMP. Cached {biomed_count} biomedical symbols.")
            
        except Exception as e:
            logger.error(f"Error fetching biomedical symbols from FMP: {e}")
        
        self._biomed_symbols_cache = symbols_set
        return symbols_set

    def get_cached_biomed_symbols(self) -> Set[str]:
        """Get cached biomedical symbols, fetching if not already cached."""
        if self._biomed_symbols_cache is None:
            self._fetch_and_cache_biomed_symbols()
        return self._biomed_symbols_cache if self._biomed_symbols_cache is not None else set()

    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile from FMP API."""
        symbol = symbol.upper()
        if symbol in self._company_profile_cache:
            return self._company_profile_cache[symbol] if self._company_profile_cache[symbol] else None
        
        try:
            profile_data = self._make_request(f"/v3/profile/{symbol}")
            if profile_data and isinstance(profile_data, list) and len(profile_data) > 0:
                profile = profile_data[0]
                self._company_profile_cache[symbol] = profile
                return profile
            else:
                self._company_profile_cache[symbol] = {}
                return None
                
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol} from FMP: {e}")
            self._company_profile_cache[symbol] = {}
            return None

    def _is_biomedical_from_profile(self, profile: Dict[str, Any]) -> bool:
        """Check if a company is biomedical based on its profile."""
        sector = profile.get('sector', '').strip()
        industry = profile.get('industry', '').strip()
        
        # FMP uses slightly different naming conventions
        biomedical_sectors = ['Healthcare', 'Health Care']
        biomedical_industries = [
            'Biotechnology', 'Pharmaceuticals', 'Drug Manufacturers', 
            'Medical Care Facilities', 'Medical Devices', 'Medical Instruments',
            'Diagnostics & Research', 'Health Information Services'
        ]
        
        return (sector in biomedical_sectors or 
                any(bio_ind.lower() in industry.lower() for bio_ind in biomedical_industries))

    def is_biomedical_company_from_profile(self, symbol: str) -> bool:
        """Check if a symbol represents a biomedical company."""
        profile = self.get_company_profile(symbol)
        if not profile:
            return False
        return self._is_biomedical_from_profile(profile)

    def get_general_market_news(self) -> List[Dict[str, Any]]:
        """Get general market news from FMP."""
        try:
            all_news = []
            cutoff_timestamp = (datetime.now() - timedelta(days=NEWS_LOOKBACK_DAYS)).timestamp()
            
            # Fetch multiple pages of general news
            for page in range(FMP_NEWS_PAGES):
                news_data = self._make_request("/v3/stock_news", {'page': page})
                if not news_data:
                    break
                
                for item in news_data:
                    # Convert FMP news format to our standard format
                    published_date = item.get('publishedDate', '')
                    if published_date:
                        try:
                            # FMP typically uses ISO format: 2024-01-15 10:30:00
                            dt = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                            item_timestamp = dt.timestamp()
                            
                            if item_timestamp < cutoff_timestamp:
                                continue
                                
                        except (ValueError, AttributeError):
                            continue
                    
                    # Extract symbols from the news
                    symbols = item.get('symbol', '')
                    if symbols:
                        if isinstance(symbols, str):
                            symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
                        else:
                            symbol_list = [str(symbols).upper()]
                        
                        # Filter for valid symbols
                        valid_symbols = [s for s in symbol_list if re.match(VALID_SYMBOL_REGEX, s)]
                        
                        if valid_symbols:
                            # Standardize the news item format
                            standardized_item = {
                                'headline': item.get('title', ''),
                                'summary': item.get('text', ''),
                                'url': item.get('url', ''),
                                'source': item.get('site', 'FMP'),
                                'datetime': item_timestamp,
                                'id': f"fmp_{item.get('url', '')}",  # Create unique ID
                                'identified_symbols': valid_symbols
                            }
                            all_news.append(standardized_item)
            
            # Sort by timestamp (newest first) and limit results
            all_news.sort(key=lambda x: x.get('datetime', 0), reverse=True)
            return all_news[:FMP_NEWS_COUNT]
            
        except Exception as e:
            logger.error(f"Error fetching general market news from FMP: {e}")
            return []

    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get company-specific news from FMP."""
        try:
            news_data = self._make_request(f"/v3/stock_news", {
                'tickers': symbol.upper(),
                'from': start_date,
                'to': end_date,
                'limit': 50
            })
            
            if not news_data:
                return []
            
            processed_news = []
            for item in news_data:
                # Standardize the news item format
                standardized_item = {
                    'headline': item.get('title', ''),
                    'summary': item.get('text', ''),
                    'url': item.get('url', ''),
                    'source': item.get('site', 'FMP'),
                    'datetime': 0,  # Will be set below
                    'id': f"fmp_company_{item.get('url', '')}",
                    'identified_symbols': [symbol.upper()]
                }
                
                # Parse datetime
                published_date = item.get('publishedDate', '')
                if published_date:
                    try:
                        dt = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        standardized_item['datetime'] = dt.timestamp()
                    except (ValueError, AttributeError):
                        standardized_item['datetime'] = datetime.now().timestamp()
                
                processed_news.append(standardized_item)
            
            return processed_news
            
        except Exception as e:
            logger.error(f"Error fetching company news for {symbol} from FMP: {e}")
            return []