import asyncio
import aiohttp
import logging
import time
from collections import deque, defaultdict
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import config
from config import (
    API_REQUEST_DELAY_SECONDS,
    FMP_API_KEY,
    FMP_BASE_URL,
    API_MAX_RETRIES,
    API_INITIAL_RETRY_DELAY,
    API_RETRY_BACKOFF_FACTOR,
    FMP_API_REQUESTS_PER_MINUTE,
    MAX_PAGES_PER_SOURCE,
    NEWS_FETCH_TIME_WINDOW_HOURS,
    MIN_NORMALIZED_NAME_LENGTH,
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_NEWS_LIMIT,
    ALPHA_VANTAGE_NEWS_SORT,
)
from utils import _normalize_name, generate_news_hash


class RateLimiter:
    """A robust, asynchronous rate limiter using the sliding window algorithm."""
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.time_window = 60.0
        self.request_timestamps = deque()
        self.lock = asyncio.Lock()
        logging.debug(f"RateLimiter initialized for {requests_per_minute} req/min.")

    async def wait_for_permission(self):
        """
        Waits until a request can be made without exceeding the rate limit.
        This uses a sliding window of timestamps to enforce the rate limit.
        """
        async with self.lock:
            while True:
                now = time.monotonic()
                # Remove timestamps that are older than the time window
                while self.request_timestamps and self.request_timestamps[0] <= now - self.time_window:
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.requests_per_minute:
                    self.request_timestamps.append(now)
                    break
                
                # Calculate how long to wait until the oldest request expires from the window
                oldest_request_time = self.request_timestamps[0]
                wait_time = (oldest_request_time + self.time_window) - now
                if wait_time > 0:
                    logging.warning(f"Rate limit nearly reached. Waiting for {wait_time:.2f}s to send next request.")
                    await asyncio.sleep(wait_time)


class FMPDataLoader:
    def __init__(self, api_key: str = FMP_API_KEY, base_url: str = FMP_BASE_URL):
        self.fmp_api_key = api_key
        self.alpha_vantage_api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = base_url
        self.session = None
        self.rate_limiter = RateLimiter(FMP_API_REQUESTS_PER_MINUTE)
        logging.info(f"FMPDataLoader initialized to use base URL: {self.base_url}")

    def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "EventDrivenNewsAnalyzer/1.1 (Proactive Rate Limit)"},
            )
        return self.session

    async def _perform_request_with_retries(self, session, url, params):
        retries = 0

        # Create a log-safe version of the full URL for transparent logging.
        params_for_log = params.copy()
        if 'apikey' in params_for_log:
            params_for_log['apikey'] = 'REDACTED'
        full_url_for_logging = f"{url}?{urlencode(params_for_log)}"

        while retries < API_MAX_RETRIES:
            try:
                await self.rate_limiter.wait_for_permission()
                async with session.get(url, params=params) as response:
                    logging.debug(f"Requesting URL: {response.url}")
                    response.raise_for_status()
                    data = await response.json()

                    if isinstance(data, dict):
                        if "Note" in data and "API call frequency" in data["Note"]:
                            raise aiohttp.ClientResponseError(request_info=response.request_info, history=response.history, status=429, message="Internal rate limit detected")
                        if "Error Message" in data:
                            raise aiohttp.ClientResponseError(request_info=response.request_info, history=response.history, status=503, message="API error message in payload")
                    
                    return data

            except aiohttp.ClientResponseError as e:
                if e.status in [401, 404]:
                    logging.error(f"HTTP Error for {full_url_for_logging}: {e.status}. Not retrying.")
                    return None
                
                retries += 1
                if retries >= API_MAX_RETRIES:
                    logging.error(f"Max retries reached for {full_url_for_logging}. Giving up.")
                    return None
                
                delay = API_INITIAL_RETRY_DELAY * (API_RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.warning(f"HTTP Error {e.status} for {full_url_for_logging}. Retrying in {delay:.2f}s (attempt {retries}/{API_MAX_RETRIES})")
                await asyncio.sleep(delay)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retries += 1
                if retries >= API_MAX_RETRIES:
                    logging.error(f"Max retries reached for {full_url_for_logging}. Giving up. Error: {e}")
                    return None
                
                delay = API_INITIAL_RETRY_DELAY * (API_RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.warning(f"Request error for {full_url_for_logging}: {e}. Retrying in {delay:.2f}s (attempt {retries}/{API_MAX_RETRIES})")
                await asyncio.sleep(delay)

        return None

    async def get_all_activity(self, ticker_filter: "TickerFilterEngine", processed_cache: Set[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches news from multiple sources and returns a unified list.
        Optimized to skip already processed news items.
        """
        if processed_cache is None:
            processed_cache = set()

        session = self._get_session()
        try:
            # Fetch from both sources concurrently
            fmp_task = self._get_fmp_news(session, ticker_filter, processed_cache)
            alpha_task = self._get_alpha_vantage_news(session, ticker_filter, processed_cache)
            
            fmp_news, alpha_news = await asyncio.gather(fmp_task, alpha_task)
            
            all_news = []
            if fmp_news:
                all_news.extend(fmp_news)
            if alpha_news:
                all_news.extend(alpha_news)
            
            # Remove duplicates based on news hash
            seen_hashes = set()
            unique_news = []
            for item in all_news:
                news_hash = generate_news_hash(item)
                if news_hash not in seen_hashes and news_hash not in processed_cache:
                    seen_hashes.add(news_hash)
                    unique_news.append(item)
            
            logging.info(f"Collected {len(unique_news)} unique, unprocessed news items from all sources.")
            return unique_news
            
        except Exception as e:
            logging.error(f"Error in get_all_activity: {e}")
            return []

    async def _get_fmp_news(self, session: aiohttp.ClientSession, ticker_filter: "TickerFilterEngine", processed_cache: Set[str]) -> List[Dict[str, Any]]:
        """Fetches news from FMP API with pagination support."""
        all_news = []
        page = 0
        
        while page < MAX_PAGES_PER_SOURCE:
            url = f"{self.base_url}/news/stock-latest"
            params = {
                "apikey": self.fmp_api_key,
                "limit": 2000,
                "page": page,
            }
            
            data = await self._perform_request_with_retries(session, url, params)
            if not data or not isinstance(data, list):
                break
            
            new_items = []
            for item in data:
                if not ticker_filter.is_valid_ticker_format(item.get("symbol")):
                    continue
                news_hash = generate_news_hash(item)
                if news_hash not in processed_cache:
                    new_items.append(item)
            
            all_news.extend(new_items)
            
            if len(data) < 2000:
                break
                
            page += 1
        
        logging.info(f"Fetched {len(all_news)} new items from FMP (pages: {page + 1})")
        return all_news

    async def _get_alpha_vantage_news(self, session: aiohttp.ClientSession, ticker_filter: "TickerFilterEngine", processed_cache: Set[str]) -> List[Dict[str, Any]]:
        """Fetches news from Alpha Vantage API."""
        # --- FIX: Add a guard clause to check for a valid API key ---
        if not self.alpha_vantage_api_key or self.alpha_vantage_api_key == "YOUR_DEFAULT_KEY":
            logging.warning("Alpha Vantage API key is not configured in the .env file. Skipping this news source.")
            return []

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.alpha_vantage_api_key,
            "limit": ALPHA_VANTAGE_NEWS_LIMIT,
            "sort": ALPHA_VANTAGE_NEWS_SORT,
        }
        
        data = await self._perform_request_with_retries(session, url, params)
        if not data or "feed" not in data:
            # --- FIX: Provide more detailed logging on failure ---
            if data:
                logging.warning(f"Alpha Vantage API returned data but no 'feed' key. Response: {data}")
            else:
                logging.warning("No data returned from Alpha Vantage API after retries. Check API key and plan limits.")
            return []
        
        transformed_news = []
        for item in data.get("feed", []):
            for ticker_mention in item.get("ticker_sentiment", []):
                ticker = ticker_mention.get("ticker")
                if not ticker or not ticker_filter.is_valid_ticker_format(ticker):
                    continue

                transformed_item = {
                    "symbol": ticker,
                    "publishedDate": item.get("time_published"),
                    "title": item.get("title"),
                    "image": item.get("banner_image"),
                    "site": item.get("source"),
                    "text": item.get("summary"),
                    "url": item.get("url"),
                    "alpha_vantage_relevance": float(ticker_mention.get("relevance_score", 0.0)),
                    "alpha_vantage_sentiment": float(ticker_mention.get("sentiment_score", 0.0)),
                }

                news_hash = generate_news_hash(transformed_item)
                
                if news_hash not in processed_cache:
                    transformed_news.append(transformed_item)
        
        logging.info(f"Fetched and transformed {len(transformed_news)} new items from Alpha Vantage")
        return transformed_news

    async def get_ticker_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetches company profile for a given ticker."""
        session = self._get_session()
        # FIX: Reverted URL to use query parameters instead of path parameters.
        url = f"{self.base_url}/profile"
        params = {"symbol": ticker, "apikey": self.fmp_api_key}
        
        data = await self._perform_request_with_retries(session, url, params)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    async def get_single_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetches a real-time quote for a single ticker."""
        session = self._get_session()
        url = f"{self.base_url}/quote"
        params = {"symbol": ticker, "apikey": self.fmp_api_key}
        
        data = await self._perform_request_with_retries(session, url, params)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    async def get_batch_quotes(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetches real-time quotes for multiple tickers by making concurrent single requests."""
        if not tickers:
            return {}
        
        tasks = [self.get_single_quote(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        
        quotes_dict = {}
        for quote in results:
            if quote and "symbol" in quote:
                quotes_dict[quote["symbol"]] = quote
        
        logging.info(f"Fetched quotes for {len(quotes_dict)} tickers")
        return quotes_dict

    async def get_batch_profiles(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches company profiles for multiple tickers.
        """
        if not tickers:
            return {}

        profiles_dict = {}
        tasks = [self.get_ticker_details(ticker) for ticker in tickers]

        results = await asyncio.gather(*tasks)
        for profile in results:
            if profile and "symbol" in profile:
                profiles_dict[profile["symbol"]] = profile

        return profiles_dict

    async def get_intraday_historical_data(
        self, ticker: str, date_str: str
    ) -> Optional[List[Dict]]:
        """
        Fetches 1-minute historical intraday data for a specific ticker and date.
        """
        session = self._get_session()
        # FIX: Reverted URL to use query parameters instead of path parameters.
        url = f"{self.base_url}/historical-chart/1min"
        params = {
            "symbol": ticker,
            "from": date_str,
            "to": date_str,
            "apikey": self.fmp_api_key,
        }
        data = await self._perform_request_with_retries(session, url, params)
        return data if data and isinstance(data, list) else None

    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


class TickerFilterEngine:
    """
    Handles ticker validation and filtering based on various criteria.
    """
    
    def __init__(self):
        self.valid_ticker_cache = set()
        self.invalid_ticker_cache = set()
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=24)
    
    def is_valid_ticker_format(self, ticker: str) -> bool:
        """
        Basic ticker format validation.
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip().upper()
        
        if len(ticker) < 1 or len(ticker) > 5:
            return False
        
        if not ticker.replace(".", "").replace("-", "").isalnum():
            return False
        
        invalid_patterns = ["TEST", "XXXX", "NULL", "NONE", "N/A"]
        if ticker in invalid_patterns:
            return False
        
        return True
    
    def is_ticker_cached(self, ticker: str) -> Optional[bool]:
        """
        Checks if ticker validity is cached and still valid.
        """
        now = datetime.now()
        
        if ticker in self.cache_expiry:
            if now > self.cache_expiry[ticker]:
                self.valid_ticker_cache.discard(ticker)
                self.invalid_ticker_cache.discard(ticker)
                del self.cache_expiry[ticker]
                return None
        
        if ticker in self.valid_ticker_cache:
            return True
        elif ticker in self.invalid_ticker_cache:
            return False
        
        return None
    
    def cache_ticker_validity(self, ticker: str, is_valid: bool):
        """Caches the validity of a ticker."""
        expiry_time = datetime.now() + self.cache_duration
        self.cache_expiry[ticker] = expiry_time
        
        if is_valid:
            self.valid_ticker_cache.add(ticker)
            self.invalid_ticker_cache.discard(ticker)
        else:
            self.invalid_ticker_cache.add(ticker)
            self.valid_ticker_cache.discard(ticker)
    
    def pre_filter_tickers(self, tickers: List[str]) -> List[str]:
        """
        Pre-filters a list of tickers based on format and cache.
        """
        valid_tickers = []
        
        for ticker in tickers:
            if not self.is_valid_ticker_format(ticker):
                continue
            
            cached_result = self.is_ticker_cached(ticker)
            if cached_result is True:
                valid_tickers.append(ticker)
            elif cached_result is False:
                continue
            else:
                valid_tickers.append(ticker)
        
        return valid_tickers