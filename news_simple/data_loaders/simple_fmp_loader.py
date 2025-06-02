"""
Enhanced FMP data loader with multiple news endpoints and improved error handling
"""
import requests
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import time
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug, log_warning


class SimpleFMPLoader:
    """Enhanced FMP data loader with multiple news sources and comprehensive error handling."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize FMP loader with API key validation."""
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")
        
        self.api_key = api_key.strip()
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self._setup_session()
        self._setup_rate_limiting()
    
    def _setup_session(self) -> None:
        """Setup optimized session with connection pooling."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingSystem/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
    
    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting configuration."""
        self.last_request_time = 0.0
        self.min_request_interval = 0.1
    
    def _rate_limit(self) -> None:
        """Implement rate limiting with precise timing."""
        elapsed = time.time() - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict, List]]:
        """Make API request with comprehensive error handling."""
        self._rate_limit()
        
        params = params or {}
        params['apikey'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                log_debug(f"Empty response from {endpoint}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            log_error(f"Request error for {endpoint}: {e}")
            return None
        except ValueError as e:
            log_error(f"JSON decode error for {endpoint}: {e}")
            return None
        except Exception as e:
            log_error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_stock_screener(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Get stock screener results with proper debugging and filtering."""
        if limit <= 0:
            log_error("Limit must be positive")
            return None
        
        params = self._build_screener_params(limit)
        
        log_info(f"Requesting stock screener with limit: {limit}")
        api_response_data = self._make_request("stock-screener", params)
        
        if not self._validate_screener_response(api_response_data):
            return None
        
        assert isinstance(api_response_data, list), "Screener data should be a list after validation"
        
        return self._process_screener_data(api_response_data)
    
    def _build_screener_params(self, limit: int) -> Dict[str, Any]:
        """Build screener request parameters."""
        return {
            "marketCapMoreThan": 100_000_000,
            "priceMoreThan": 2,
            "priceLowerThan": 500,
            "volumeMoreThan": 50_000,
            "isActivelyTrading": "true",
            "exchange": "NYSE,NASDAQ",
            "limit": min(limit, 1000)
        }
    
    def _validate_screener_response(self, data: Any) -> bool:
        """Validate screener API response."""
        if not data:
            log_error("No data received from stock screener API")
            return False
        
        if not isinstance(data, list):
            log_error(f"Expected list response, got: {type(data)}")
            return False
        
        log_info(f"Stock screener API returned {len(data)} results")
        return True
    
    def _process_screener_data(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        """Process screener data with filtering and validation."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                log_error("Empty DataFrame from stock screener")
                return None
            
            if 'symbol' not in df.columns:
                log_error(f"No 'symbol' column in response. Columns: {df.columns.tolist()}")
                return None
            
            log_info(f"DataFrame created with {len(df)} rows")
            
            # Apply symbol filtering
            filtered_df = self._filter_symbols(df)
            
            if filtered_df.empty:
                log_error("All symbols filtered out!")
                return None
            
            return filtered_df.reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing screener data: {e}")
            return None
    
    def _filter_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter symbols with optimized logic."""
        initial_count = len(df)
        
        # Remove null/empty symbols
        df = df[df['symbol'].notna()]
        df = df[df['symbol'].astype(str).str.strip() != '']
        log_info(f"After removing null/empty symbols: {len(df)} from {initial_count}")
        
        # Apply length filter
        length_mask = df['symbol'].str.len() <= 6
        df_length = df[length_mask]
        log_info(f"After length filter (<=6 chars): {len(df_length)} from {len(df)}")
        
        # Apply alphabetic filter
        alpha_mask = df_length['symbol'].str.match(r'^[A-Z][A-Z0-9\-\.]*$')
        df_filtered = df_length[alpha_mask]
        log_info(f"After alpha filter: {len(df_filtered)} from {len(df_length)}")
        
        # Use lenient filtering if too few results
        if len(df_filtered) < 50:
            return self._apply_lenient_filter(df)
        
        return df_filtered
    
    def _apply_lenient_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply more lenient filtering when strict filtering yields too few results."""
        log_info("Applying lenient filtering...")
        
        lenient_mask = (
            (df['symbol'].str.len() >= 1) & 
            (df['symbol'].str.len() <= 8) &
            (~df['symbol'].str.contains(r'[^A-Z0-9\-\.]', na=False))
        )
        
        df_lenient = df[lenient_mask]
        log_info(f"Lenient filtering: {len(df_lenient)} symbols")
        
        return df_lenient
    
    def get_real_time_prices(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Get real-time prices with batch optimization."""
        if not symbols:
            return None
        
        # Clean and deduplicate symbols
        unique_symbols = list(dict.fromkeys(
            sym.strip().upper() for sym in symbols if sym.strip()
        ))
        
        if not unique_symbols:
            return None
        
        # Process in batches
        symbol_batch = unique_symbols[:100]
        symbol_str = ",".join(symbol_batch)
        
        data = self._make_request(f"stock/full/real-time-price/{symbol_str}")
        
        if not data:
            return None
        
        return self._process_price_data(data)
    
    def _process_price_data(self, data: Union[Dict, List]) -> Optional[pd.DataFrame]:
        """Process price data with error handling."""
        try:
            df = pd.DataFrame(data)
            if df.empty or 'symbol' not in df.columns:
                return None
            
            # Convert numeric columns
            numeric_columns = ['lastSalePrice', 'bidPrice', 'askPrice', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            log_error(f"Error processing price data: {e}")
            return None
    
    def get_comprehensive_news(self) -> Optional[pd.DataFrame]:
        """Get news from multiple FMP endpoints for comprehensive coverage."""
        try:
            all_articles = []
            
            # 1. Primary RSS feed (current method)
            log_debug("Fetching from RSS sentiment feed...")
            rss_articles = self._get_rss_news()
            if rss_articles:
                all_articles.extend(rss_articles)
                log_debug(f"RSS feed: {len(rss_articles)} articles")
            
            # 2. General stock news (new endpoint)
            log_debug("Fetching from general stock news...")
            general_articles = self._get_general_stock_news()
            if general_articles:
                all_articles.extend(general_articles)
                log_debug(f"General news: {len(general_articles)} articles")
            
            # 3. Social sentiment (if available)
            log_debug("Fetching social sentiment data...")
            social_articles = self._get_social_sentiment_news()
            if social_articles:
                all_articles.extend(social_articles)
                log_debug(f"Social sentiment: {len(social_articles)} articles")
            
            if not all_articles:
                log_warning("No articles retrieved from any news endpoint")
                return None
            
            # Remove duplicates and process
            df = pd.DataFrame(all_articles)
            df = self._deduplicate_news_articles(df)
            
            log_info(f"Total comprehensive news articles: {len(df)}")
            return self._process_news_data(df)
            
        except Exception as e:
            log_error(f"Error in comprehensive news fetching: {e}")
            # Fallback to original method
            return self.get_news_rss()
    
    def _get_rss_news(self) -> List[Dict[str, Any]]:
        """Get news from RSS sentiment feed."""
        try:
            all_articles = []
            
            # Fetch multiple pages
            for page in range(CONFIG.news_page_limit):
                params = {
                    "page": page,
                    "limit": CONFIG.news_per_page_limit
                }
                
                data = self._make_request("../v4/stock-news-sentiments-rss-feed", params)
                
                if not data or not isinstance(data, list):
                    break
                
                all_articles.extend(data)
                
                if len(data) < CONFIG.news_per_page_limit:
                    break
                
                time.sleep(0.2)  # Rate limiting
            
            return all_articles
            
        except Exception as e:
            log_error(f"Error fetching RSS news: {e}")
            return []
    
    def _get_general_stock_news(self) -> List[Dict[str, Any]]:
        """Get news from general stock news endpoint."""
        try:
            all_articles = []
            
            # Fetch multiple pages
            for page in range(min(3, CONFIG.news_page_limit)):  # Limit to 3 pages for this endpoint
                params = {
                    "page": page,
                    "limit": min(50, CONFIG.news_per_page_limit)  # Smaller limit for this endpoint
                }
                
                data = self._make_request("stock_news", params)
                
                if not data or not isinstance(data, list):
                    break
                
                # Transform data to match expected format
                transformed_articles = []
                for article in data:
                    if isinstance(article, dict):
                        # Map fields to expected format
                        transformed = {
                            'symbol': article.get('symbol', ''),
                            'title': article.get('title', ''),
                            'text': article.get('text', ''),
                            'url': article.get('url', ''),
                            'publishedDate': article.get('publishedDate', ''),
                            'site': article.get('site', ''),
                            'source': 'general_news'
                        }
                        transformed_articles.append(transformed)
                
                all_articles.extend(transformed_articles)
                
                if len(data) < params['limit']:
                    break
                
                time.sleep(0.3)  # Slightly longer delay for this endpoint
            
            return all_articles
            
        except Exception as e:
            log_debug(f"General stock news not available or error: {e}")
            return []
    
    def _get_social_sentiment_news(self) -> List[Dict[str, Any]]:
        """Get social sentiment data."""
        try:
            # Fetch for popular symbols
            popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
            all_articles = []
            
            for symbol in popular_symbols[:5]:  # Limit to 5 symbols to avoid rate limits
                try:
                    data = self._make_request(f"../v4/social-sentiments", {'symbol': symbol})
                    
                    if data and isinstance(data, list):
                        # Transform social sentiment to news-like format
                        for item in data[:5]:  # Limit items per symbol
                            if isinstance(item, dict):
                                transformed = {
                                    'symbol': symbol,
                                    'title': f"Social Sentiment Update: {symbol}",
                                    'text': f"Social sentiment analysis shows {item.get('sentiment', 'neutral')} sentiment for {symbol}",
                                    'url': '',
                                    'publishedDate': item.get('date', ''),
                                    'site': 'social_sentiment',
                                    'source': 'social_sentiment',
                                    'sentiment': item.get('sentiment', 'neutral')
                                }
                                all_articles.append(transformed)
                    
                    time.sleep(0.5)  # Longer delay for social sentiment
                    
                except Exception as e:
                    log_debug(f"Error fetching social sentiment for {symbol}: {e}")
                    continue
            
            return all_articles
            
        except Exception as e:
            log_debug(f"Social sentiment not available or error: {e}")
            return []
    
    def _deduplicate_news_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles from multiple sources."""
        if df.empty:
            return df
        
        try:
            # Remove exact duplicates by title and symbol
            df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # Remove very similar titles
            if len(df) <= 1000:  # Only for manageable sizes
                df = self._remove_similar_titles(df)
            
            return df
            
        except Exception as e:
            log_error(f"Error deduplicating news articles: {e}")
            return df
    
    def _remove_similar_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with very similar titles."""
        if 'title' not in df.columns:
            return df
        
        try:
            # Simple similarity check
            seen_titles = set()
            keep_indices = []
            
            for idx, row in df.iterrows():
                title = str(row['title']).lower().strip()
                title_words = set(title.split())
                
                # Check for high similarity with seen titles
                is_similar = False
                for seen_title in seen_titles:
                    seen_words = set(seen_title.split())
                    if title_words and seen_words:
                        intersection = len(title_words & seen_words)
                        union = len(title_words | seen_words)
                        if union > 0 and intersection / union > 0.8:  # 80% similarity
                            is_similar = True
                            break
                
                if not is_similar:
                    keep_indices.append(idx)
                    seen_titles.add(title)
                    
                    # Limit memory usage
                    if len(seen_titles) > 500:
                        seen_titles = set(list(seen_titles)[-250:])
            
            return df.loc[keep_indices]
            
        except Exception as e:
            log_error(f"Error removing similar titles: {e}")
            return df
    
    def get_news_rss(self) -> Optional[pd.DataFrame]:
        """Get news using comprehensive method (backwards compatibility)."""
        return self.get_comprehensive_news()
    
    def _process_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process news data with optimized handling."""
        if df.empty:
            return df
        
        # Handle datetime
        if 'publishedDate' in df.columns:
            df['publishedDate'] = pd.to_datetime(
                df['publishedDate'], 
                errors='coerce',
                utc=True
            )
        
        # Create content field
        if 'title' in df.columns and 'text' in df.columns:
            df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
        
        # Ensure source field exists
        if 'source' not in df.columns:
            df['source'] = 'rss_feed'
        
        # Filter essential data with more permissive requirements
        essential_columns = ['symbol', 'title']
        for col in essential_columns:
            if col in df.columns:
                df = df[df[col].notna() & (df[col].str.strip() != '')]
        
        # Remove completely empty text
        if 'text' in df.columns:
            df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        
        return df
    
    @lru_cache(maxsize=100)
    def get_historical_data_cached(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical data with caching for repeated requests."""
        return self.get_historical_data(symbol, days)
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data without caching."""
        from datetime import datetime, timedelta
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self._make_request(f"historical-price-full/{symbol}", {
            'from': start_date,
            'to': end_date
        })
        
        if not data:
            return None
        
        return self._process_historical_data(data, symbol)
    
    def _process_historical_data(self, data: Union[Dict, List], symbol: str) -> Optional[pd.DataFrame]:
        """Process historical data with error handling."""
        try:
            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                historical_data = data['historical']
            elif isinstance(data, list):
                historical_data = data
            else:
                log_error(f"Unexpected data format for {symbol}: {type(data)}")
                return None
            
            if not historical_data:
                log_debug(f"Empty historical data for {symbol}")
                return None
            
            df = pd.DataFrame(historical_data)
            
            if 'date' not in df.columns:
                log_error(f"Missing 'date' column in historical data for {symbol}")
                return None
            
            # Process data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(subset=['close']).reset_index(drop=True)
            
        except Exception as e:
            log_error(f"Error processing historical data for {symbol}: {e}")
            return None
    
    def __del__(self):
        """Cleanup session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()