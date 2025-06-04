"""
Simplified news fetcher using FMP APIs
Python 3.13.3 compatible
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug
from config import Config


class NewsFetcher(BaseFMPLoader):
    """Fetch financial news from FMP APIs"""
    
    def __init__(self, api_key: str) -> None:
        """Initialize news fetcher"""
        super().__init__(api_key)
        self.max_articles = Config.MAX_NEWS_ARTICLES
    
    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all available FMP sources"""
        all_news = []
        
        # Define news sources and their fetch methods
        news_sources = [
            ("General Stock News", self._fetch_stock_news),
            ("Press Releases", self._fetch_press_releases),
            ("Earnings News", self._fetch_earnings_news),
            ("Market News", self._fetch_market_news)
        ]
        
        for source_name, fetch_method in news_sources:
            try:
                articles = fetch_method()
                if articles:
                    all_news.extend(articles)
                    log_info(f"Fetched {len(articles)} articles from {source_name}")
            except Exception as e:
                log_error(f"Error fetching from {source_name}: {e}")
        
        # Remove duplicates and sort by date
        all_news = self._deduplicate_articles(all_news)
        all_news = self._filter_recent_articles(all_news)
        
        log_info(f"Total articles fetched: {len(all_news)}")
        return all_news[:self.max_articles]
    
    def _fetch_stock_news(self) -> List[Dict[str, Any]]:
        """Fetch general stock news"""
        try:
            data = self.make_request("stock_news", {"limit": 200})
            
            if not data or not isinstance(data, list):
                return []
            
            articles = []
            for item in data:
                if self._is_valid_article(item):
                    articles.append(self._normalize_article(item, "stock_news"))
            
            return articles
            
        except Exception as e:
            log_error(f"Error fetching stock news: {e}")
            return []
    
    def _fetch_press_releases(self) -> List[Dict[str, Any]]:
        """Fetch press releases"""
        try:
            data = self.make_request("press-releases", {"limit": 100})
            
            if not data or not isinstance(data, list):
                return []
            
            articles = []
            for item in data:
                if self._is_valid_article(item) and item.get('symbol'):
                    articles.append(self._normalize_article(item, "press_release"))
            
            return articles
            
        except Exception as e:
            log_error(f"Error fetching press releases: {e}")
            return []
    
    def _fetch_earnings_news(self) -> List[Dict[str, Any]]:
        """Fetch earnings-related news"""
        try:
            # Get earnings calendar first
            earnings_data = self.make_request("earning_calendar", {"limit": 50})
            
            if not earnings_data or not isinstance(earnings_data, list):
                return []
            
            articles = []
            for item in earnings_data:
                if item.get('symbol') and item.get('date'):
                    # Create synthetic news article from earnings data
                    article = {
                        'symbol': item['symbol'],
                        'title': f"Earnings Call: {item['symbol']} Q{item.get('quarter', '?')} {item.get('year', '')}",
                        'text': f"Earnings call scheduled for {item.get('date', 'TBD')}. EPS estimate: {item.get('epsEstimated', 'N/A')}",
                        'url': '',
                        'publishedDate': item.get('date', datetime.now(timezone.utc).isoformat()),
                        'source': 'earnings_calendar'
                    }
                    articles.append(self._normalize_article(article, "earnings"))
            
            return articles
            
        except Exception as e:
            log_error(f"Error fetching earnings news: {e}")
            return []
    
    def _fetch_market_news(self) -> List[Dict[str, Any]]:
        """Fetch general market news"""
        try:
            data = self.make_request("general-news", {"limit": 50}, use_v4=True)
            
            if not data or not isinstance(data, list):
                return []
            
            articles = []
            for item in data:
                if self._is_valid_article(item):
                    # For general market news, assign to major market symbols
                    item['symbol'] = 'SPY'  # Use SPY as proxy for market news
                    articles.append(self._normalize_article(item, "market_news"))
            
            return articles
            
        except Exception as e:
            log_debug(f"General news not available: {e}")
            return []
    
    def _is_valid_article(self, article: Dict[str, Any]) -> bool:
        """Validate article has required fields"""
        required_fields = ['title']
        
        for field in required_fields:
            if not article.get(field):
                return False
        
        # Check minimum content length
        title = str(article.get('title', ''))
        text = str(article.get('text', ''))
        
        if len(title + text) < Config.MIN_NEWS_LENGTH:
            return False
        
        return True
    
    def _normalize_article(self, article: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize article to standard format"""
        return {
            'symbol': str(article.get('symbol', '')).upper().strip(),
            'title': str(article.get('title', '')).strip(),
            'text': str(article.get('text', '')).strip(),
            'url': str(article.get('url', '')).strip(),
            'publishedDate': article.get('publishedDate', datetime.now(timezone.utc).isoformat()),
            'source': source,
            'original_data': article  # Keep original for debugging
        }
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title and symbol"""
        seen = set()
        unique_articles = []
        
        for article in articles:
            # Create unique key from symbol and title
            key = f"{article['symbol']}:{article['title'][:100]}"
            
            if key not in seen:
                seen.add(key)
                unique_articles.append(article)
        
        log_debug(f"Deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles
    
    def _filter_recent_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles to only include recent ones"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=Config.MAX_NEWS_AGE_HOURS)
        recent_articles = []
        
        for article in articles:
            try:
                pub_date = pd.to_datetime(article['publishedDate'], utc=True)
                if pd.notna(pub_date) and pub_date >= cutoff_time:
                    recent_articles.append(article)
            except Exception:
                # Include articles with unparseable dates
                recent_articles.append(article)
        
        log_debug(f"Time filter: {len(articles)} -> {len(recent_articles)} recent articles")
        return recent_articles
    
    def get_news_for_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch news for specific symbols"""
        if not symbols:
            return []
        
        symbol_news = []
        
        # Fetch news for each symbol individually
        for symbol in symbols[:50]:  # Limit to 50 symbols to avoid API overload
            try:
                data = self.make_request(f"stock_news", {
                    "tickers": symbol,
                    "limit": 10
                })
                
                if data and isinstance(data, list):
                    for item in data:
                        if self._is_valid_article(item):
                            item['symbol'] = symbol  # Ensure symbol is set
                            symbol_news.append(self._normalize_article(item, "targeted_news"))
                            
            except Exception as e:
                log_debug(f"Error fetching news for {symbol}: {e}")
        
        return symbol_news