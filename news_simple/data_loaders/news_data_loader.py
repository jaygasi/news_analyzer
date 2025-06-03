"""
News data loader for FMP API - Fixed date filtering and cache logic
"""
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from config import CONFIG
from utils.simple_logger import log_debug, log_error, log_info
from .base_fmp_loader import BaseFMPLoader


class NewsDataLoader(BaseFMPLoader):
    """Specialized loader for news-related FMP endpoints with fixed date filtering."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize with optimized settings for fresh news."""
        super().__init__(api_key)
        self._universe_cache: List[str] = []
        self._last_fetch_time = 0.0
        self._fetch_cooldown = 3.0  # Reduced cooldown for 25s scans
        
        # Track successful requests by endpoint to optimize API calls
        self._endpoint_success: Dict[str, bool] = {}
        
    def get_comprehensive_news(self) -> List[Dict[str, Any]]:
        """Get truly fresh news with improved filtering."""
        current_time = time.time()
        
        # Rate limiting check
        if current_time - self._last_fetch_time < self._fetch_cooldown:
            return []  # Return empty rather than wait
        
        all_articles = []
        
        # For 25-second scans, only get articles from last 5 minutes to account for API delays
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(minutes=5)
        
        log_debug(f"Fetching news newer than {from_time.isoformat()}")
        
        # Prioritize by speed and freshness for 25s scans
        news_sources = [
            ("RSS feed", self._get_rss_news_fast, 1.0),
            ("Stock news", self._get_stock_news_fast, 0.9),
            ("Press releases", self._get_press_releases_fast, 0.8),
        ]
        
        for source_name, source_func, priority in news_sources:
            try:
                articles = source_func(from_time, now)
                if articles:
                    for article in articles:
                        article['_priority'] = priority
                        article['_fetch_time'] = current_time
                    all_articles.extend(articles)
                    log_debug(f"{source_name}: {len(articles)} fresh articles")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        self._last_fetch_time = current_time
        
        if all_articles:
            # Sort by freshness first, then priority
            all_articles = self._sort_by_freshness_strict(all_articles, from_time)
            log_info(f"Fresh articles found: {len(all_articles)}")
        
        return all_articles
    
    def _sort_by_freshness_strict(self, articles: List[Dict[str, Any]], 
                                 cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Strict freshness sorting with cutoff filtering."""
        try:
            fresh_articles = []
            
            for article in articles:
                # Parse date and apply strict cutoff
                pub_date_str = article.get('publishedDate', '')
                if pub_date_str:
                    try:
                        pub_date = pd.to_datetime(pub_date_str, utc=True)
                        if pd.notna(pub_date) and pub_date >= cutoff_time:
                            article['_parsed_date'] = pub_date
                            fresh_articles.append(article)
                    except:
                        # Include articles with unparseable dates as potentially fresh
                        article['_parsed_date'] = datetime.now(timezone.utc)
                        fresh_articles.append(article)
                else:
                    # Include articles with no date as potentially fresh
                    article['_parsed_date'] = datetime.now(timezone.utc)
                    fresh_articles.append(article)
            
            # Sort by parsed date (newest first), then by priority
            fresh_articles.sort(
                key=lambda x: (x['_parsed_date'], x.get('_priority', 0)), 
                reverse=True
            )
            
            # Clean up temporary fields
            for article in fresh_articles:
                article.pop('_parsed_date', None)
                article.pop('_priority', None)
            
            return fresh_articles
            
        except Exception as e:
            log_debug(f"Error in strict freshness sorting: {e}")
            return articles
    
    def _get_rss_news_fast(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Fast RSS news with minimal pages for 25s scans."""
        try:
            articles = []
            
            # Only fetch first 2 pages for speed with 25s scans
            for page in range(2):
                # Try simple request first (fastest)
                params = {
                    "page": page,
                    "limit": min(30, CONFIG.news_per_page_limit)
                }
                
                data = self._make_request("stock-news-sentiments-rss-feed", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                # Client-side filtering for freshness
                page_fresh_count = 0
                for article in data:
                    if self._is_article_fresh(article, from_time):
                        article['source'] = 'rss_feed'
                        articles.append(article)
                        page_fresh_count += 1
                
                # If no fresh articles on this page, stop pagination
                if page_fresh_count == 0:
                    break
                
                # Stop if we got fewer articles than requested (end of data)
                if len(data) < min(30, CONFIG.news_per_page_limit):
                    break
                
                time.sleep(0.1)
            
            return articles
            
        except Exception as e:
            log_debug(f"Error fetching RSS news: {e}")
            return []
    
    def _get_stock_news_fast(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Fast stock news with freshness filtering."""
        try:
            articles = []
            
            # Only get first page for speed in 25s scans
            params = {"limit": 20}
            data = self._make_request("stock_news", params)
            
            if data and isinstance(data, list):
                for article in data:
                    if (self._is_article_fresh(article, from_time) and 
                        self._is_relevant_news(article)):
                        transformed = self._transform_article(article, 'stock_news')
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Stock news error: {e}")
            return []
    
    def _get_press_releases_fast(self, from_time: datetime, to_time: datetime) -> List[Dict[str, Any]]:
        """Fast press releases with freshness filtering."""
        try:
            articles = []
            
            params = {"limit": 15}
            data = self._make_request("press-releases", params)
            
            if data and isinstance(data, list):
                for article in data:
                    if (article.get('symbol') and 
                        self._is_article_fresh(article, from_time)):
                        transformed = self._transform_article(article, 'press_release')
                        articles.append(transformed)
            
            return articles
            
        except Exception as e:
            log_debug(f"Press releases error: {e}")
            return []
    
    def _is_article_fresh(self, article: Dict, cutoff_time: datetime) -> bool:
        """Check if article is fresh (after cutoff time)."""
        try:
            pub_date_str = (article.get('publishedDate') or 
                           article.get('date') or 
                           article.get('timestamp', ''))
            
            if not pub_date_str:
                return True  # Assume fresh if no date
            
            pub_date = pd.to_datetime(pub_date_str, utc=True)
            if pd.isna(pub_date):
                return True  # Assume fresh if unparseable
            
            # Add small buffer (1 minute) for clock drift
            return pub_date >= (cutoff_time - timedelta(minutes=1))
            
        except:
            return True  # Assume fresh on error
    
    def _is_relevant_news(self, article: Dict) -> bool:
        """Fast relevance check for news articles."""
        if not isinstance(article, dict):
            return False
        
        # Must have symbol and title
        if not (article.get('symbol') and article.get('title')):
            return False
        
        title = str(article.get('title', '')).lower()
        text = str(article.get('text', '')).lower()
        content = f"{title} {text}"
        
        # Fast keyword check - only high-value terms
        high_value_keywords = [
            'earnings', 'revenue', 'guidance', 'beats', 'misses',
            'acquisition', 'merger', 'deal', 'approval', 'fda',
            'upgrade', 'downgrade', 'target', 'breakthrough'
        ]
        
        # Quick check for any high-value keyword
        has_keywords = any(keyword in content for keyword in high_value_keywords)
        
        # Basic quality check
        has_content = len(content.strip()) >= 20
        
        # Quick spam check
        not_spam = not any(spam in content for spam in ['click here', 'advertisement'])
        
        return has_keywords and has_content and not_spam
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for targeted fetching."""
        self._universe_cache = universe[:50]  # Reduced for 25s scans
        log_debug(f"Universe cache updated: {len(self._universe_cache)} symbols")