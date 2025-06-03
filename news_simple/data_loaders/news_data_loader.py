"""
News data loader for FMP API - handles all news-related endpoints with incremental fetching
"""
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import CONFIG
from utils.simple_logger import log_debug, log_error, log_info
from .base_fmp_loader import BaseFMPLoader


class NewsDataLoader(BaseFMPLoader):
    """Specialized loader for news-related FMP endpoints with incremental fetching."""
    
    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        self.last_fetch_file = CONFIG.cache_dir / "last_fetch_timestamps.json"
        self.last_fetch_times = self._load_last_fetch_times()
        self._universe_cache = []
    
    def _load_last_fetch_times(self) -> Dict[str, str]:
        """Load last fetch timestamps from file."""
        try:
            if self.last_fetch_file.exists():
                with open(self.last_fetch_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            log_debug(f"Could not load last fetch times: {e}")
        
        # Default to 6 hours ago for initial fetch
        default_time = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        return {
            "rss_feed": default_time,
            "stock_news": default_time,
            "fmp_articles": default_time,
            "press_releases": default_time,
            "company_news": default_time
        }
    
    def _save_last_fetch_times(self) -> None:
        """Save last fetch timestamps to file."""
        try:
            with open(self.last_fetch_file, 'w') as f:
                json.dump(self.last_fetch_times, f, indent=2)
        except Exception as e:
            log_debug(f"Could not save last fetch times: {e}")
    
    def _update_last_fetch_time(self, source: str, timestamp: Optional[str] = None) -> None:
        """Update last fetch time for a source."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        self.last_fetch_times[source] = timestamp
        self._save_last_fetch_times()

    def get_comprehensive_news(self) -> List[Dict[str, Any]]:
        """Get comprehensive news from all sources with incremental fetching."""
        all_articles = []
        current_time = datetime.now(timezone.utc)
        
        news_sources = [
            ("RSS feed", "rss_feed", self._get_rss_news_incremental),
            ("Stock news", "stock_news", self._get_stock_news_incremental),
            ("FMP articles", "fmp_articles", self._get_fmp_articles_incremental),
            ("Press releases", "press_releases", self._get_press_releases_incremental),
            ("Company news", "company_news", self._get_company_specific_news_incremental),
        ]
        
        for source_name, source_key, source_func in news_sources:
            try:
                articles = source_func()
                if articles:
                    all_articles.extend(articles)
                    log_debug(f"{source_name}: {len(articles)} new articles")
                    # Update last fetch time with current time
                    self._update_last_fetch_time(source_key)
                else:
                    log_debug(f"{source_name}: No new articles")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        log_info(f"Total new articles fetched: {len(all_articles)}")
        return all_articles
    
    def _get_rss_news_incremental(self) -> List[Dict[str, Any]]:
        """Get RSS news feed with incremental fetching."""
        try:
            articles = []
            last_fetch = self.last_fetch_times.get("rss_feed")
            
            # Convert to FMP API date format
            since_date = self._format_date_for_api(last_fetch)
            
            for page in range(min(CONFIG.news_page_limit, 3)):  # Limit pages for incremental
                params = {
                    "page": page,
                    "limit": CONFIG.news_per_page_limit,
                }
                
                # Add date filter if API supports it
                if since_date:
                    params["from"] = since_date
                
                data = self._make_request("stock-news-sentiments-rss-feed", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                # Filter articles newer than last fetch
                new_articles = self._filter_articles_by_date(data, last_fetch)
                articles.extend(new_articles)
                
                # If we get fewer articles than requested, we've reached the end
                if len(data) < CONFIG.news_per_page_limit:
                    break
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_error(f"Error fetching incremental RSS news: {e}")
            return []
    
    def _get_stock_news_incremental(self) -> List[Dict[str, Any]]:
        """Get stock news with incremental fetching."""
        try:
            articles = []
            last_fetch = self.last_fetch_times.get("stock_news")
            since_date = self._format_date_for_api(last_fetch)
            
            for page in range(min(2, CONFIG.news_page_limit)):  # Fewer pages for incremental
                params = {
                    "page": page,
                    "limit": min(30, CONFIG.news_per_page_limit)
                }
                
                if since_date:
                    params["from"] = since_date
                
                data = self._make_request("stock_news", params)
                
                if not data or not isinstance(data, list):
                    break
                
                # Filter and transform articles
                new_articles = self._filter_articles_by_date(data, last_fetch)
                for article in new_articles:
                    if self._is_relevant_news(article):
                        transformed = self._transform_article(article, 'stock_news')
                        articles.append(transformed)
                
                if len(data) < params['limit']:
                    break
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_debug(f"Incremental stock news not available: {e}")
            return []
    
    def _get_fmp_articles_incremental(self) -> List[Dict[str, Any]]:
        """Get FMP articles with incremental fetching."""
        try:
            articles = []
            last_fetch = self.last_fetch_times.get("fmp_articles")
            since_date = self._format_date_for_api(last_fetch)
            
            for page in range(min(2, CONFIG.news_page_limit)):
                params = {
                    "page": page,
                    "size": min(20, CONFIG.news_per_page_limit)
                }
                
                if since_date:
                    params["from"] = since_date
                
                data = self._make_request("fmp/articles", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                new_articles = self._filter_articles_by_date(data, last_fetch, date_field='date')
                for article in new_articles:
                    if 'tickers' in article and article['tickers']:
                        # Process each ticker mentioned in the article
                        for ticker in article['tickers'][:3]:  # Limit to first 3 tickers
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
                
                if len(data) < params['size']:
                    break
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_debug(f"Incremental FMP articles not available: {e}")
            return []
    
    def _get_press_releases_incremental(self) -> List[Dict[str, Any]]:
        """Get press releases with incremental fetching."""
        try:
            articles = []
            last_fetch = self.last_fetch_times.get("press_releases")
            since_date = self._format_date_for_api(last_fetch)
            
            for page in range(min(2, CONFIG.news_page_limit)):
                params = {
                    "page": page,
                    "limit": 20
                }
                
                if since_date:
                    params["from"] = since_date
                
                data = self._make_request("press-releases", params)
                
                if not data or not isinstance(data, list):
                    break
                
                new_articles = self._filter_articles_by_date(data, last_fetch)
                for article in new_articles:
                    if 'symbol' in article and article['symbol']:
                        transformed = self._transform_article(article, 'press_release')
                        articles.append(transformed)
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_debug(f"Incremental press releases not available: {e}")
            return []
    
    def _get_company_specific_news_incremental(self) -> List[Dict[str, Any]]:
        """Get company-specific news with incremental fetching."""
        try:
            articles = []
            last_fetch = self.last_fetch_times.get("company_news")
            since_date = self._format_date_for_api(last_fetch)
            
            # Get universe symbols (limit to top 20 for API efficiency in incremental mode)
            universe_symbols = self._universe_cache[:20] if self._universe_cache else []
            
            for symbol in universe_symbols:
                try:
                    params = {"limit": 3}  # Reduce limit for incremental
                    
                    if since_date:
                        params["from"] = since_date
                    
                    data = self._make_request(f"stock_news/{symbol}", params)
                    
                    if data and isinstance(data, list):
                        new_articles = self._filter_articles_by_date(data, last_fetch)
                        for article in new_articles:
                            transformed = self._transform_article(article, 'company_news')
                            transformed['symbol'] = symbol  # Ensure symbol is set
                            articles.append(transformed)
                    
                    time.sleep(0.1)  # Small delay between symbol requests
                    
                except Exception as e:
                    log_debug(f"Error getting incremental news for {symbol}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            log_debug(f"Incremental company-specific news not available: {e}")
            return []
    
    def _format_date_for_api(self, timestamp_str: Optional[str]) -> Optional[str]:
        """Format timestamp for FMP API date parameters."""
        if not timestamp_str:
            return None
        
        try:
            # Parse ISO timestamp and convert to API format (YYYY-MM-DD)
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except Exception as e:
            log_debug(f"Error formatting date {timestamp_str}: {e}")
            return None
    
    def _filter_articles_by_date(self, articles: List[Dict], last_fetch: Optional[str], 
                                date_field: str = 'publishedDate') -> List[Dict]:
        """Filter articles to only include those newer than last fetch."""
        if not last_fetch or not articles:
            return articles
        
        try:
            last_fetch_dt = datetime.fromisoformat(last_fetch.replace('Z', '+00:00'))
            filtered_articles = []
            
            for article in articles:
                article_date_str = article.get(date_field, '')
                if not article_date_str:
                    # Include articles without dates
                    filtered_articles.append(article)
                    continue
                
                try:
                    # Parse article date
                    if 'T' in article_date_str:
                        article_dt = datetime.fromisoformat(article_date_str.replace('Z', '+00:00'))
                    else:
                        # Handle date-only format
                        article_dt = datetime.strptime(article_date_str, '%Y-%m-%d')
                        article_dt = article_dt.replace(tzinfo=timezone.utc)
                    
                    # Include if newer than last fetch
                    if article_dt > last_fetch_dt:
                        filtered_articles.append(article)
                        
                except Exception as e:
                    log_debug(f"Error parsing article date {article_date_str}: {e}")
                    # Include articles with unparseable dates
                    filtered_articles.append(article)
            
            return filtered_articles
            
        except Exception as e:
            log_debug(f"Error filtering articles by date: {e}")
            return articles
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for company-specific news fetching."""
        self._universe_cache = universe[:50]  # Cache top 50 symbols for incremental fetching
    
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
    
    def force_refresh_from_time(self, hours_back: int = 1) -> None:
        """Force refresh last fetch times to pull more recent data."""
        refresh_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        
        for source in self.last_fetch_times:
            self.last_fetch_times[source] = refresh_time
        
        self._save_last_fetch_times()
        log_info(f"Forced refresh: last fetch times reset to {hours_back} hours ago")