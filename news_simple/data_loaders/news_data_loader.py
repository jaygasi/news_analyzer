"""
News data loader for FMP API - handles all news-related endpoints with date-based filtering
"""
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from config import CONFIG
from utils.simple_logger import log_debug, log_error, log_info
from .base_fmp_loader import BaseFMPLoader


class NewsDataLoader(BaseFMPLoader):
    """Specialized loader for news-related FMP endpoints with date-based filtering."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize with optimized settings for fresh news."""
        super().__init__(api_key)
        self._universe_cache: List[str] = []
        self._last_fetch_time = 0.0
        self._fetch_cooldown = 5.0  # Minimum seconds between fetches to avoid rate limits
        
        # Track last successful fetch time for incremental updates
        self._last_successful_fetch = None
    
    def get_comprehensive_news(self) -> List[Dict[str, Any]]:
        """Get comprehensive news with date-based filtering for efficiency."""
        current_time = time.time()
        
        # Rate limiting check
        if current_time - self._last_fetch_time < self._fetch_cooldown:
            time.sleep(self._fetch_cooldown - (current_time - self._last_fetch_time))
        
        all_articles = []
        
        # Calculate time window for API filtering
        now = datetime.now(timezone.utc)
        
        # Since we scan every 25 seconds, only fetch articles from last 3 minutes
        # This accounts for:
        # 1. Potential API delays/caching
        # 2. Clock drift between systems  
        # 3. Small overlap to prevent missing articles
        time_window_minutes = 3
        from_time = now - timedelta(minutes=time_window_minutes)
        
        log_debug(f"Fetching news from {from_time.isoformat()} to {now.isoformat()}")
        
        # Prioritize sources by freshness and reliability with date filtering
        news_sources = [
            ("RSS feed", lambda: self._get_rss_news(from_time, now), 1.0),
            ("Stock news", lambda: self._get_stock_news(from_time, now), 0.9),
            ("Company news", lambda: self._get_company_specific_news(from_time, now), 0.8),
            ("Press releases", lambda: self._get_press_releases(from_time, now), 0.7),
            ("FMP articles", lambda: self._get_fmp_articles(from_time, now), 0.6),
        ]
        
        for source_name, source_func, priority in news_sources:
            try:
                articles = source_func()
                if articles:
                    # Add priority metadata for sorting
                    for article in articles:
                        article['_priority'] = priority
                    all_articles.extend(articles)
                    log_debug(f"{source_name}: {len(articles)} articles (priority: {priority})")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        self._last_fetch_time = time.time()
        self._last_successful_fetch = now
        
        # Sort by priority and freshness
        if all_articles:
            all_articles = self._sort_articles_by_freshness(all_articles)
            log_info(f"Total new articles fetched: {len(all_articles)}")
        
        return all_articles
    
    def _sort_articles_by_freshness(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort articles by freshness and priority."""
        try:
            import pandas as pd
            
            # Convert to DataFrame for easier sorting
            df = pd.DataFrame(articles)
            
            # Parse published dates
            if 'publishedDate' in df.columns:
                df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce', utc=True)
                # Fill NaN dates with a very old date for sorting
                df['publishedDate'] = df['publishedDate'].fillna(pd.Timestamp('2020-01-01', tz='UTC'))
            else:
                # If no date, assign current time
                df['publishedDate'] = pd.Timestamp.now(tz='UTC')
            
            # Sort by priority (descending) and publishedDate (descending - newest first)
            df = df.sort_values(['_priority', 'publishedDate'], ascending=[False, False])
            
            # Remove priority metadata before returning
            df = df.drop('_priority', axis=1, errors='ignore')
            
            return df.to_dict('records')
            
        except Exception as e:
            log_debug(f"Error sorting articles by freshness: {e}")
            return articles
    
    def _get_rss_news(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Get RSS news feed with date-based filtering."""
        try:
            articles = []
            
            # Format times for API (try different formats as FMP might be picky)
            from_str = from_time.strftime('%Y-%m-%d %H:%M:%S')
            to_str = to_time.strftime('%Y-%m-%d %H:%M:%S')
            from_iso = from_time.isoformat()
            
            # Reduced page limit since we're date filtering
            for page in range(min(3, CONFIG.news_page_limit)):
                
                # Try with date parameters first
                params_with_date = {
                    "page": page,
                    "limit": CONFIG.news_per_page_limit,
                    "from": from_str,
                    "to": to_str
                }
                
                data = None
                
                # Try multiple date parameter formats
                for date_params in [
                    {"from": from_str, "to": to_str},
                    {"from": from_iso, "to": to_time.isoformat()},
                    {"fromDate": from_str, "toDate": to_str},
                    {"startDate": from_str, "endDate": to_str}
                ]:
                    try:
                        test_params = {
                            "page": page,
                            "limit": CONFIG.news_per_page_limit,
                            **date_params
                        }
                        data = self._make_request("stock-news-sentiments-rss-feed", test_params, use_v4=True)
                        if data and isinstance(data, list):
                            log_debug(f"RSS: Date filtering working with params: {list(date_params.keys())}")
                            break
                    except Exception as e:
                        log_debug(f"RSS: Date params {list(date_params.keys())} failed: {e}")
                        continue
                
                # Fallback to no date filtering if all attempts failed
                if not data or not isinstance(data, list):
                    log_debug("RSS: Falling back to no date filtering")
                    fallback_params = {
                        "page": page,
                        "limit": CONFIG.news_per_page_limit
                    }
                    data = self._make_request("stock-news-sentiments-rss-feed", fallback_params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                # Client-side date filtering as backup/verification
                filtered_articles = []
                for article in data:
                    try:
                        pub_date_str = article.get('publishedDate', '')
                        if pub_date_str:
                            pub_date = pd.to_datetime(pub_date_str, utc=True)
                            if pd.notna(pub_date):
                                # Only keep articles from our time window (with small buffer)
                                age_minutes = (to_time - pub_date).total_seconds() / 60
                                if age_minutes <= 10:  # 10 minute buffer for clock drift
                                    article['source'] = 'rss_feed'
                                    filtered_articles.append(article)
                                    continue
                        
                        # Keep articles with no date or unparseable dates (might be fresh)
                        article['source'] = 'rss_feed'
                        filtered_articles.append(article)
                        
                    except Exception as e:
                        log_debug(f"Error filtering article by date: {e}")
                        # Keep article if date parsing fails
                        article['source'] = 'rss_feed'
                        filtered_articles.append(article)
                
                articles.extend(filtered_articles)
                
                if len(data) < CONFIG.news_per_page_limit:
                    break
                
                time.sleep(0.1)
            
            log_debug(f"RSS: Fetched {len(articles)} articles with date filtering")
            return articles
            
        except Exception as e:
            log_error(f"Error fetching RSS news: {e}")
            return []
    
    def _get_stock_news(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Get general stock news with date filtering."""
        try:
            articles = []
            from_str = from_time.strftime('%Y-%m-%d')
            to_str = to_time.strftime('%Y-%m-%d')
            
            for page in range(min(2, CONFIG.news_page_limit)):  # Reduced since date filtered
                
                # Try with date parameters
                params_with_date = {
                    "page": page,
                    "limit": min(30, CONFIG.news_per_page_limit),
                    "from": from_str,
                    "to": to_str
                }
                
                try:
                    data = self._make_request("stock_news", params_with_date)
                    if not data or not isinstance(data, list):
                        # Fallback without dates
                        params_fallback = {
                            "page": page,
                            "limit": min(30, CONFIG.news_per_page_limit)
                        }
                        data = self._make_request("stock_news", params_fallback)
                except:
                    # Fallback without dates
                    params_fallback = {
                        "page": page,
                        "limit": min(30, CONFIG.news_per_page_limit)
                    }
                    data = self._make_request("stock_news", params_fallback)
                
                if not data or not isinstance(data, list):
                    break
                
                for article in data:
                    if self._is_relevant_and_recent(article, from_time, to_time):
                        transformed = self._transform_article(article, 'stock_news')
                        articles.append(transformed)
                
                if len(data) < min(30, CONFIG.news_per_page_limit):
                    break
                
                time.sleep(0.1)
            
            return articles
            
        except Exception as e:
            log_debug(f"Stock news not available: {e}")
            return []
    
    def _get_press_releases(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Get press releases with date filtering."""
        try:
            articles = []
            from_str = from_time.strftime('%Y-%m-%d')
            to_str = to_time.strftime('%Y-%m-%d')
            
            for page in range(min(2, CONFIG.news_page_limit)):
                
                # Try with date parameters
                params = {
                    "page": page,
                    "limit": 20,
                    "from": from_str,
                    "to": to_str
                }
                
                try:
                    data = self._make_request("press-releases", params)
                except:
                    # Fallback without dates
                    params = {"page": page, "limit": 20}
                    data = self._make_request("press-releases", params)
                
                if not data or not isinstance(data, list):
                    break
                
                for article in data:
                    if ('symbol' in article and article['symbol'] and 
                        self._is_recent_article(article, from_time, to_time)):
                        transformed = self._transform_article(article, 'press_release')
                        articles.append(transformed)
                
                time.sleep(0.1)
            
            return articles
            
        except Exception as e:
            log_debug(f"Press releases not available: {e}")
            return []
    
    def _get_fmp_articles(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Get FMP curated articles with date filtering."""
        try:
            articles = []
            from_str = from_time.strftime('%Y-%m-%d')
            to_str = to_time.strftime('%Y-%m-%d')
            
            for page in range(min(2, CONFIG.news_page_limit)):
                
                # Try with date parameters
                params = {
                    "page": page,
                    "size": 20,
                    "from": from_str,
                    "to": to_str
                }
                
                try:
                    data = self._make_request("fmp/articles", params, use_v4=True)
                except:
                    # Fallback without dates
                    params = {"page": page, "size": 20}
                    data = self._make_request("fmp/articles", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                for article in data:
                    if ('tickers' in article and article['tickers'] and
                        self._is_recent_article(article, from_time, to_time)):
                        
                        # Process each ticker mentioned in the article
                        for ticker in article['tickers'][:3]:  # Limit to 3 for efficiency
                            transformed = {
                                'symbol': ticker.strip().upper(),
                                'title': article.get('title', ''),
                                'text': article.get('content', '')[:1000],
                                'url': article.get('url', ''),
                                'publishedDate': article.get('date', ''),
                                'site': 'fmp_articles',
                                'source': 'fmp_articles'
                            }
                            articles.append(transformed)
                
                if len(data) < 20:
                    break
                
                time.sleep(0.1)
            
            return articles
            
        except Exception as e:
            log_debug(f"FMP articles not available: {e}")
            return []
    
    def _get_company_specific_news(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Get company-specific news with date filtering."""
        try:
            articles = []
            
            # Get universe symbols - focus on top performers
            universe_symbols = getattr(self, '_universe_cache', [])[:50]  # Reduced for efficiency
            
            from_str = from_time.strftime('%Y-%m-%d')
            to_str = to_time.strftime('%Y-%m-%d')
            
            # Process in smaller batches for efficiency
            for symbol in universe_symbols:
                try:
                    # Try with date parameters
                    params = {
                        "limit": 5,
                        "from": from_str,
                        "to": to_str
                    }
                    
                    try:
                        data = self._make_request(f"stock_news/{symbol}", params)
                    except:
                        # Fallback without dates
                        params = {"limit": 5}
                        data = self._make_request(f"stock_news/{symbol}", params)
                    
                    if data and isinstance(data, list):
                        for article in data:
                            if self._is_recent_article(article, from_time, to_time):
                                transformed = self._transform_article(article, 'company_news')
                                transformed['symbol'] = symbol  # Ensure symbol is set
                                articles.append(transformed)
                    
                    time.sleep(0.05)  # Very short delay
                    
                except Exception as e:
                    log_debug(f"Error getting news for {symbol}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            log_debug(f"Company-specific news not available: {e}")
            return []
    
    def _is_recent_article(self, article: Dict, from_time: datetime, to_time: datetime) -> bool:
        """Check if article is within our time window."""
        try:
            pub_date_str = article.get('publishedDate') or article.get('date', '')
            if not pub_date_str:
                return True  # Include articles with no date
            
            pub_date = pd.to_datetime(pub_date_str, utc=True)
            if pd.isna(pub_date):
                return True  # Include articles with unparseable dates
            
            # Check if within time window (with small buffer)
            buffer_minutes = 15  # 15 minute buffer
            from_time_buffered = from_time - timedelta(minutes=buffer_minutes)
            to_time_buffered = to_time + timedelta(minutes=buffer_minutes)
            
            return from_time_buffered <= pub_date <= to_time_buffered
            
        except Exception:
            return True  # Include on error
    
    def _is_relevant_and_recent(self, article: Dict, from_time: datetime, to_time: datetime) -> bool:
        """Check if article is both relevant and recent."""
        return self._is_relevant_news(article) and self._is_recent_article(article, from_time, to_time)
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for company-specific news fetching."""
        self._universe_cache = universe[:100]  # Cache top 100 symbols
        log_debug(f"Updated universe cache with {len(self._universe_cache)} symbols")
    
    def _is_relevant_news(self, article: Dict) -> bool:
        """Enhanced relevance filtering for news articles."""
        if not isinstance(article, dict):
            return False
        
        title = str(article.get('title', '')).lower()
        text = str(article.get('text', '')).lower()
        content = f"{title} {text}"
        
        # High-value keywords (expanded list)
        relevant_keywords = [
            # Financial results
            'earnings', 'revenue', 'profit', 'guidance', 'sales', 'income',
            'eps', 'quarterly', 'annual', 'results', 'beats', 'misses', 'exceeds', 'disappoints',
            
            # Corporate actions
            'acquisition', 'merger', 'buyout', 'takeover', 'deal', 'agreement',
            'partnership', 'joint venture', 'collaboration', 'alliance',
            'dividend', 'buyback', 'repurchase', 'spinoff', 'split',
            
            # Regulatory & approvals
            'fda', 'approval', 'approved', 'rejected', 'trial', 'study',
            'patent', 'license', 'regulatory', 'investigation',
            
            # Market moving events
            'breakthrough', 'innovation', 'launch', 'announces', 'reports',
            'upgrade', 'downgrade', 'target', 'analyst', 'recommendation',
            'ipo', 'listing', 'offering', 'secondary',
            
            # Leadership & strategy
            'ceo', 'cfo', 'executive', 'management', 'appointed', 'resigned',
            'strategy', 'restructuring', 'expansion', 'investment',
            
            # Financial health
            'debt', 'funding', 'financing', 'loan', 'credit', 'bankruptcy',
            'cash', 'balance sheet', 'financial position'
        ]
        
        # Check for keyword matches
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in content)
        
        # Must have at least 1 relevant keyword
        has_keywords = keyword_matches >= 1
        
        # Additional quality checks
        has_symbol = 'symbol' in article and str(article['symbol']).strip()
        has_reasonable_length = len(content.strip()) >= 20
        
        # Filter out obvious spam/promotional content
        spam_indicators = ['click here', 'visit our website', 'subscribe now', 'advertisement', 'promo']
        is_not_spam = not any(spam in content for spam in spam_indicators)
        
        return has_keywords and has_symbol and has_reasonable_length and is_not_spam