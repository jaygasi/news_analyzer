# biomed_analyzer/analyzer/finnhub_client.py
import finnhub
from datetime import datetime, timedelta, date 
from typing import List, Dict, Optional, Any, Set
import time
import re

from .config import config
from .utils import logger

# Define regex constant at module level
VALID_SYMBOL_REGEX = r'^[A-Z0-9\.]{1,10}$'

class FinnhubClientError(Exception):
    """Custom exception for Finnhub client errors"""
    pass

class FinnhubClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.FINNHUB_API_KEY
        
        if not self.api_key:
            raise FinnhubClientError("Finnhub API key is required but not provided")
        
        # Initialize client
        try:
            self.client = finnhub.Client(api_key=self.api_key)
        except Exception as e:
            raise FinnhubClientError(f"Failed to initialize Finnhub client: {e}")
        
        # Initialize rate limiting and caches BEFORE validation
        self._last_api_call = 0
        self._company_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._biomed_symbols_cache: Optional[Set[str]] = None
        
        # Validate API key with proper endpoint
        self._validate_api_key()
        
        logger.info("Finnhub client initialized and validated successfully")
    
    def _rate_limit(self):
        """Enforce rate limiting: max 60 calls/minute for free tier"""
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call
        min_interval = 1.1  # Slightly over 1 second to stay under 60/minute
        
        if time_since_last_call < min_interval:
            sleep_time = min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_api_call = time.time()
    
    def _validate_api_key(self):
        """Validate API key with a proper test call"""
        try:
            # Use company_profile2 for AAPL as test - this is documented and reliable
            self._rate_limit()
            response = self.client.company_profile2(symbol='AAPL')
            
            # Check if we got a proper response
            if not response or not isinstance(response, dict):
                raise FinnhubClientError("API key validation failed - received invalid response")
            
            # Check for expected fields in response
            if 'name' not in response or not response.get('name'):
                raise FinnhubClientError("API key validation failed - incomplete response")
            
            logger.debug(f"Finnhub API key validated successfully - test company: {response.get('name')}")
            
        except finnhub.FinnhubAPIException as e:
            error_msg = str(e)
            if "403" in error_msg or "Forbidden" in error_msg:
                raise FinnhubClientError(
                    "Finnhub API key is invalid or doesn't have required permissions. "
                    "Please check your API key at https://finnhub.io/dashboard"
                )
            elif "429" in error_msg or "rate" in error_msg.lower():
                raise FinnhubClientError(
                    "Finnhub API rate limit exceeded. Free tier allows 60 calls/minute. "
                    "Please wait or upgrade your plan."
                )
            else:
                raise FinnhubClientError(f"Finnhub API validation failed: {e}")
        except finnhub.FinnhubRequestException as e:
            error_msg = str(e)
            if "Invalid Response" in error_msg and "<!DOCTYPE html>" in error_msg:
                raise FinnhubClientError(
                    "Invalid Finnhub API key - receiving HTML instead of JSON. "
                    "Please verify your API key at https://finnhub.io/dashboard"
                )
            else:
                raise FinnhubClientError(f"Finnhub request failed: {e}")
        except Exception as e:
            raise FinnhubClientError(f"Unexpected error during API validation: {e}")
    
    def _make_api_call(self, method_name: str, *args, **kwargs):
        """Wrapper for API calls with proper error handling and rate limiting"""
        self._rate_limit()
        
        try:
            method = getattr(self.client, method_name)
            result = method(*args, **kwargs)
            return result
        except finnhub.FinnhubAPIException as e:
            error_msg = str(e)
            if "403" in error_msg:
                logger.error(f"Finnhub API access denied for {method_name}")
                raise FinnhubClientError("API access denied - check your API key permissions")
            elif "429" in error_msg:
                logger.warning("Finnhub API rate limit hit - waiting...")
                time.sleep(60)  # Wait 1 minute for rate limit reset
                return None
            else:
                logger.error(f"Finnhub API error in {method_name}: {e}")
                return None
        except finnhub.FinnhubRequestException as e:
            error_msg = str(e)
            if "Invalid Response" in error_msg and "<!DOCTYPE html>" in error_msg:
                logger.error("Finnhub API key became invalid during operation")
                raise FinnhubClientError("API key invalid - please check your Finnhub dashboard")
            else:
                logger.error(f"Finnhub request error in {method_name}: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error in Finnhub {method_name}: {e}")
            return None

    def _fetch_and_cache_biomed_symbols(self, exchange: str = 'US') -> Set[str]:
        """Fetch and cache biomedical symbols with robust error handling"""
        logger.info(f"Fetching biomedical symbols for exchange: {exchange}")
        symbols_set: Set[str] = set()
        
        try:
            all_symbols = self._make_api_call('stock_symbols', exchange)
            
            if not all_symbols:
                logger.warning(f"No symbols returned from Finnhub for exchange {exchange}")
                self._biomed_symbols_cache = symbols_set
                return symbols_set

            processed_count = 0
            biomed_count = 0
            
            for stock_info in all_symbols:
                processed_count += 1
                symbol = stock_info.get('symbol')
                
                if not symbol or not re.match(VALID_SYMBOL_REGEX, symbol):
                    continue

                stock_type = stock_info.get('type', '').lower()
                if stock_type not in ['common stock', 'adrc', '']: 
                    continue

                gics_sector = stock_info.get('gsector', '').strip()
                gics_industry_group = stock_info.get('ggroup', '').strip()
                gics_industry = stock_info.get('gind', '').strip()

                # Check against our biomedical classifications
                is_biomed = (
                    gics_sector in config.BIOMEDICAL_GICS_SECTORS or
                    gics_industry_group in config.BIOMEDICAL_GICS_INDUSTRY_GROUPS or
                    gics_industry in config.BIOMEDICAL_GICS_INDUSTRIES
                )
                
                if is_biomed:
                    symbols_set.add(symbol)
                    biomed_count += 1
            
            logger.info(f"Processed {processed_count} symbols, found {biomed_count} biomedical companies")
            
        except FinnhubClientError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"Unexpected error fetching biomedical symbols: {e}")
            # Don't fail completely - return empty set
        
        self._biomed_symbols_cache = symbols_set
        return symbols_set

    def get_cached_biomed_symbols(self, exchange: str = 'US') -> Set[str]:
        """Get cached biomedical symbols"""
        if self._biomed_symbols_cache is None:
            try:
                self._fetch_and_cache_biomed_symbols(exchange)
            except Exception as e:
                logger.error(f"Failed to fetch biomedical symbols: {e}")
                self._biomed_symbols_cache = set()
        
        return self._biomed_symbols_cache if self._biomed_symbols_cache is not None else set()

    def _extract_symbols_from_related(self, related_str: Optional[str]) -> List[str]:
        """Helper to extract and validate symbols from news 'related' string"""
        item_symbols = []
        if related_str and isinstance(related_str, str) and related_str.strip():
            symbols_from_related = re.split(r'[,\s]+', related_str)
            for s_rel in symbols_from_related:
                s_rel = s_rel.strip().upper()
                if s_rel and re.match(VALID_SYMBOL_REGEX, s_rel):
                    item_symbols.append(s_rel)
        return list(set(item_symbols))

    def get_general_market_news(self, category: str = 'general') -> List[Dict[str, Any]]:
        """Get general market news with robust error handling"""
        try:
            news_items = self._make_api_call('general_news', category, min_id=0)
            
            if not news_items:
                logger.info("No general market news items returned from Finnhub")
                return []
            
            cutoff_timestamp = (datetime.now() - timedelta(days=config.NEWS_LOOKBACK_DAYS)).timestamp()
            
            processed_news = []
            for item in news_items:
                if not ('datetime' in item and item['datetime'] >= cutoff_timestamp):
                    continue

                item_symbols = self._extract_symbols_from_related(item.get('related'))
                
                if item_symbols:
                    item['identified_symbols'] = item_symbols
                    processed_news.append(item)
            
            processed_news.sort(key=lambda x: x['datetime'], reverse=True)
            result = processed_news[:config.FINNHUB_NEWS_COUNT]
            logger.info(f"Retrieved {len(result)} relevant general market news items from Finnhub")
            return result
            
        except FinnhubClientError:
            logger.error("Failed to get general market news from Finnhub")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting general market news: {e}")
            return []

    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile with caching and error handling"""
        symbol = symbol.upper()
        if symbol in self._company_profile_cache:
            cached = self._company_profile_cache[symbol]
            return cached if cached else None
        
        try:
            # Use company_profile2 as per official documentation
            profile = self._make_api_call('company_profile2', symbol=symbol)
            
            if profile and profile.get('ticker'): 
                self._company_profile_cache[symbol] = profile
                return profile
            else:
                self._company_profile_cache[symbol] = {}
                return None
                
        except FinnhubClientError:
            self._company_profile_cache[symbol] = {}
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching company profile for {symbol}: {e}")
            self._company_profile_cache[symbol] = {}
            return None

    def is_biomedical_company_from_profile(self, symbol: str) -> bool:
        """Check if company is biomedical based on profile"""
        profile = self.get_company_profile(symbol)
        if not profile: 
            return False

        gics_sector = profile.get('gsector', '').strip()
        gics_industry_group = profile.get('ggroup', '').strip()
        gics_industry = profile.get('gind', '').strip()

        return (
            gics_sector in config.BIOMEDICAL_GICS_SECTORS or
            gics_industry_group in config.BIOMEDICAL_GICS_INDUSTRY_GROUPS or
            gics_industry in config.BIOMEDICAL_GICS_INDUSTRIES
        )

    def get_company_news(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get company-specific news"""
        try:
            # Use _from parameter as per documentation (from is Python keyword)
            news = self._make_api_call('company_news', symbol, _from=start_date, to=end_date)
            
            if news:
                for item in news:
                    item['identified_symbols'] = [symbol.upper()]
                logger.debug(f"Retrieved {len(news)} news items for {symbol}")
            
            return news or []
            
        except FinnhubClientError:
            logger.error(f"Failed to get company news for {symbol}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting company news for {symbol}: {e}")
            return []

    def get_fda_calendar_events(self) -> List[Dict[str, Any]]:
        """Get FDA calendar events"""
        try:
            events = self._make_api_call('fda_calendar')
            
            if not events:
                logger.info("No FDA calendar events returned")
                return []

            relevant_events = []
            today = date.today()
            min_event_date = today - timedelta(days=config.FDA_NEWS_WINDOW_PAST_DAYS + 7)
            max_event_date = today + timedelta(days=config.FDA_NEWS_WINDOW_FUTURE_DAYS + 30)

            for event in events:
                event_date_str = event.get('date')
                symbol = event.get('symbol', '').strip().upper()

                if not (event_date_str and symbol and isinstance(event_date_str, str) and 
                        re.match(VALID_SYMBOL_REGEX, symbol)):
                    continue

                try:
                    event_dt = datetime.strptime(event_date_str, '%Y-%m-%d').date()
                    if min_event_date <= event_dt <= max_event_date:
                        relevant_events.append(event)
                except ValueError:
                    logger.debug(f"Could not parse date '{event_date_str}' for FDA event for {symbol}")
            
            logger.info(f"Retrieved {len(events)} FDA events, {len(relevant_events)} relevant")
            return relevant_events
            
        except FinnhubClientError:
            logger.error("Failed to get FDA calendar events")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting FDA calendar: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test if the connection is working"""
        try:
            self._validate_api_key()
            return True
        except Exception:
            return False