"""
Enhanced FMP data loader with optimized news endpoints and error handling
"""
import requests
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import time
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug, log_warning


class SimpleFMPLoader:
    """Enhanced FMP data loader with comprehensive news sources and error handling."""
    
    # Class-level constants for better performance
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    TIMEOUT = 30
    MIN_REQUEST_INTERVAL = 0.5  # Increased from 0.1 to reduce API load
    
    def __init__(self, api_key: str) -> None:
        """Initialize FMP loader with validated API key."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self._setup_session()
        self._setup_rate_limiting()
    
    def _setup_session(self) -> None:
        """Setup optimized HTTP session."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('https://', adapter)
    
    def _setup_rate_limiting(self) -> None:
        """Initialize rate limiting."""
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict, List]]:
        """Make API request with comprehensive error handling."""
        self._rate_limit()
        
        request_params = params or {}
        request_params['apikey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=request_params, timeout=self.TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                log_debug(f"Empty response from {endpoint}")
                return None
            
            return data
            
        except requests.exceptions.Timeout:
            log_error(f"Timeout requesting {endpoint}")
        except requests.exceptions.ConnectionError:
            log_error(f"Connection error for {endpoint}")
        except requests.exceptions.HTTPError as e:
            log_error(f"HTTP error {e.response.status_code} for {endpoint}")
        except requests.exceptions.RequestException as e:
            log_error(f"Request error for {endpoint}: {e}")
        except ValueError as e:
            log_error(f"JSON decode error for {endpoint}: {e}")
        except Exception as e:
            log_error(f"Unexpected error for {endpoint}: {e}")
        
        return None
    
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener with enhanced parameters for news-heavy stocks."""
        if limit <= 0:
            raise ValueError("Limit must be positive")
        
        # Enhanced screener parameters for better news coverage
        params = {
            "marketCapMoreThan": CONFIG.min_market_cap,
            "priceMoreThan": CONFIG.min_price,
            "priceLowerThan": CONFIG.max_price,
            "volumeMoreThan": CONFIG.min_volume,
            "isActivelyTrading": "true",
            "exchange": "NASDAQ,NYSE",  # Focus on main exchanges for better news coverage
            "sector": "Technology,Healthcare,Financial Services,Consumer Cyclical,Industrials,Communication Services,Consumer Defensive,Energy,Real Estate",  # Expanded sectors
            "limit": min(limit, 1000)
        }
        
        log_info(f"Requesting stock screener with enhanced parameters, limit: {limit}")
        data = self._make_request("stock-screener", params)
        
        if not self._validate_response(data, "stock screener"):
            return None
        
        return self._process_screener_data(data)
    
    def _validate_response(self, data: Any, endpoint_name: str) -> bool:
        """Validate API response."""
        if not data:
            log_error(f"No data from {endpoint_name}")
            return False
        
        if not isinstance(data, list):
            log_error(f"Expected list from {endpoint_name}, got: {type(data)}")
            return False
        
        log_info(f"{endpoint_name} returned {len(data)} results")
        return True
    
    def _process_screener_data(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        """Process screener data with enhanced filtering."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                log_error("Invalid screener data structure")
                return None
            
            log_info(f"Processing {len(df)} screener results")
            
            # Enhanced symbol filtering for news relevance
            filtered_df = self._filter_symbols_for_news(df)
            
            return filtered_df.reset_index(drop=True) if not filtered_df.empty else None
            
        except Exception as e:
            log_error(f"Error processing screener data: {e}")
            return None
    
    def _filter_symbols_for_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter symbols optimized for news coverage."""
        initial_count = len(df)
        
        # Basic filters
        df = df[df['symbol'].notna() & (df['symbol'].str.strip() != '')]
        
        # News-optimized filters
        filters = (
            (df['symbol'].str.len() <= 5) &  # Shorter symbols get more news
            (df['symbol'].str.match(r'^[A-Z]+$')) &  # Only alphabetic
            (~df['symbol'].str.contains(r'[.]')) &  # No dots (preferred shares)
            (df['symbol'].str.len() >= 1)  # Minimum length
        )
        
        df_filtered = df[filters]
        
        # If too restrictive, apply lenient filtering
        if len(df_filtered) < 100:  # Increased from 50
            log_info("Applying lenient filtering for better coverage")
            lenient_filters = (
                (df['symbol'].str.len() <= 6) &
                (df['symbol'].str.match(r'^[A-Z][A-Z0-9]*$'))
            )
            df_filtered = df[lenient_filters]
        
        log_info(f"Symbol filtering: {initial_count} -> {len(df_filtered)}")
        return df_filtered
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Clean and deduplicate
        clean_symbols = list(dict.fromkeys(
            symbol.strip().upper() for symbol in symbols if symbol.strip()
        ))
        
        if not clean_symbols:
            return None
        
        # Process in optimized batches
        symbol_batch = clean_symbols[:100]  # Reduced from 150 to avoid errors
        symbol_string = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_string}")
        return self._process_price_data(data) if data else None
    
    def _process_price_data(self, data: Union[Dict, List]) -> Optional[pd.DataFrame]:
        """Process price data with type conversion."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return None
            
            # Optimize numeric conversion
            numeric_cols = ['lastSalePrice', 'bidPrice', 'askPrice', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            log_error(f"Error processing price data: {e}")
            return None
    
    def get_comprehensive_news(self) -> Optional[pd.DataFrame]:
        """Get enhanced news from multiple FMP endpoints with better error handling."""
        try:
            all_articles = []
            
            # Primary RSS feed with optimized parameters
            rss_articles = self._get_enhanced_rss_news()
            if rss_articles:
                all_articles.extend(rss_articles)
                log_debug(f"RSS feed: {len(rss_articles)} articles")
            
            # Stock news endpoint
            stock_news = self._get_stock_news()
            if stock_news:
                all_articles.extend(stock_news)
                log_debug(f"Stock news: {len(stock_news)} articles")
            
            # Earnings news
            earnings_articles = self._get_earnings_news()
            if earnings_articles:
                all_articles.extend(earnings_articles)
                log_debug(f"Earnings news: {len(earnings_articles)} articles")
            
            # Press releases
            press_releases = self._get_press_releases()
            if press_releases:
                all_articles.extend(press_releases)
                log_debug(f"Press releases: {len(press_releases)} articles")
            
            if not all_articles:
                log_warning("No articles from any news endpoint")
                return None
            
            # Enhanced deduplication and processing
            df = pd.DataFrame(all_articles)
            df = self._enhanced_deduplicate_news(df)
            
            log_info(f"Total comprehensive news: {len(df)} articles")
            return self._process_news_data(df)
            
        except Exception as e:
            log_error(f"Error fetching comprehensive news: {e}")
            return self.get_news_rss()  # Fallback
    
    def _get_enhanced_rss_news(self) -> List[Dict[str, Any]]:
        """Get RSS news with enhanced parameters."""
        try:
            articles = []
            
            for page in range(CONFIG.news_page_limit):
                params = {
                    "page": page,
                    "limit": CONFIG.news_per_page_limit,
                }
                
                data = self._make_request("../v4/stock-news-sentiments-rss-feed", params)
                
                if not data or not isinstance(data, list):
                    break
                
                articles.extend(data)
                
                # Stop if we got fewer than requested (end of data)
                if len(data) < CONFIG.news_per_page_limit:
                    break
                
                time.sleep(0.5)  # Rate limiting between pages
            
            return articles
            
        except Exception as e:
            log_error(f"Error fetching RSS news: {e}")
            return []
    
    def _get_stock_news(self) -> List[Dict[str, Any]]:
        """Get general stock news."""
        try:
            articles = []
            
            for page in range(min(3, CONFIG.news_page_limit)):
                params = {
                    "page": page,
                    "limit": min(30, CONFIG.news_per_page_limit)
                }
                
                data = self._make_request("stock_news", params)
                
                if not data or not isinstance(data, list):
                    break
                
                # Transform and filter for relevance
                for article in data:
                    if self._is_relevant_news(article):
                        transformed = self._transform_news_article(article, 'stock_news')
                        articles.append(transformed)
                
                if len(data) < params['limit']:
                    break
                
                time.sleep(0.3)
            
            return articles
            
        except Exception as e:
            log_debug(f"Stock news not available: {e}")
            return []
    
    def _get_earnings_news(self) -> List[Dict[str, Any]]:
        """Get earnings-specific news."""
        try:
            # Get recent earnings calendar for relevant symbols
            data = self._make_request("earning_calendar", {"limit": 50})
            
            if not data:
                return []
            
            articles = []
            for earning in data[:20]:  # Limit processing
                if 'symbol' in earning:
                    article = {
                        'symbol': earning['symbol'],
                        'title': f"Earnings Report: {earning['symbol']}",
                        'text': f"Earnings scheduled for {earning.get('date', 'TBD')}",
                        'url': '',
                        'publishedDate': earning.get('date', ''),
                        'site': 'earnings_calendar',
                        'source': 'earnings'
                    }
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            log_debug(f"Earnings news not available: {e}")
            return []
    
    def _get_press_releases(self) -> List[Dict[str, Any]]:
        """Get press releases."""
        try:
            articles = []
            
            for page in range(min(2, CONFIG.news_page_limit)):
                params = {
                    "page": page,
                    "limit": 20
                }
                
                data = self._make_request("press-releases", params)
                
                if not data or not isinstance(data, list):
                    break
                
                for article in data:
                    if 'symbol' in article and article['symbol']:
                        transformed = self._transform_news_article(article, 'press_release')
                        articles.append(transformed)
                
                time.sleep(0.3)
            
            return articles
            
        except Exception as e:
            log_debug(f"Press releases not available: {e}")
            return []
    
    def _is_relevant_news(self, article: Dict) -> bool:
        """Filter for relevant news based on content."""
        if not isinstance(article, dict):
            return False
        
        title = str(article.get('title', '')).lower()
        text = str(article.get('text', '')).lower()
        content = f"{title} {text}"
        
        # High-value keywords
        relevant_keywords = [
            'earnings', 'revenue', 'profit', 'guidance', 'acquisition', 'merger',
            'fda', 'approval', 'partnership', 'contract', 'breakthrough',
            'upgrade', 'downgrade', 'target', 'analyst', 'dividend',
            'buyback', 'spinoff', 'ipo', 'secondary offering',
            'beats', 'misses', 'exceeds', 'disappoints'
        ]
        
        return any(keyword in content for keyword in relevant_keywords)
    
    def _transform_news_article(self, article: Dict, source: str) -> Dict[str, Any]:
        """Transform article to standard format."""
        return {
            'symbol': article.get('symbol', ''),
            'title': article.get('title', ''),
            'text': article.get('text', ''),
            'url': article.get('url', ''),
            'publishedDate': article.get('publishedDate', ''),
            'site': article.get('site', ''),
            'source': source
        }
    
    def _enhanced_deduplicate_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced deduplication with content similarity."""
        if df.empty:
            return df
        
        try:
            # Remove exact duplicates
            df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # Enhanced similarity removal for manageable datasets
            if len(df) <= 800:  # Increased from 1000
                df = self._remove_similar_content(df)
            
            return df
            
        except Exception as e:
            log_error(f"Error in deduplication: {e}")
            return df
    
    def _remove_similar_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with similar content."""
        if 'title' not in df.columns:
            return df
        
        try:
            unique_indices = []
            seen_content = set()
            
            for idx, row in df.iterrows():
                title = str(row['title']).lower().strip()
                symbol = str(row.get('symbol', '')).upper()
                
                # Create content fingerprint
                title_words = set(word for word in title.split() if len(word) > 3)
                content_key = (symbol, frozenset(title_words))
                
                # Check similarity with existing content
                is_similar = False
                for existing_symbol, existing_words in seen_content:
                    if (existing_symbol == symbol and 
                        title_words and existing_words and
                        len(title_words & existing_words) / len(title_words | existing_words) > 0.6):  # Reduced threshold
                        is_similar = True
                        break
                
                if not is_similar:
                    unique_indices.append(idx)
                    seen_content.add(content_key)
                    
                    # Prevent memory growth
                    if len(seen_content) > 300:  # Reduced from 500
                        seen_content = set(list(seen_content)[-150:])
            
            return df.loc[unique_indices]
            
        except Exception as e:
            log_error(f"Error removing similar content: {e}")
            return df
    
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Backward compatibility method."""
        return self.get_comprehensive_news()
    
    def _process_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process news data with optimized transformations."""
        if df.empty:
            return df
        
        # Optimize datetime conversion
        if 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce', utc=True)
        
        # Create combined content field
        if 'title' in df.columns and 'text' in df.columns:
            df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
        
        # Ensure source field
        if 'source' not in df.columns:
            df['source'] = 'rss_feed'
        
        # Filter essential data
        essential_filters = (
            df['symbol'].notna() & 
            (df['symbol'].str.strip() != '') &
            df['title'].notna() & 
            (df['title'].str.strip() != '')
        )
        
        return df[essential_filters]
    
    @lru_cache(maxsize=100)
    def get_historical_data_cached(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Cached historical data retrieval."""
        return self.get_historical_data(symbol, days)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        return self._process_historical_data(data, symbol) if data else None
    
    def _process_historical_data(self, data: Union[Dict, List], symbol: str) -> Optional[pd.DataFrame]:
        """Process historical data with validation."""
        try:
            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                log_error(f"Unexpected historical data format for {symbol}")
                return None
            
            if not historical_data:
                return None
            
            df = pd.DataFrame(historical_data)
            
            if 'date' not in df.columns:
                log_error(f"Missing date column for {symbol}")
                return None
            
            # Process and validate data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def __del__(self):
        """Clean up session resources."""
        if hasattr(self, 'session'):
            self.session.close()