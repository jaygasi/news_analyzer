"""
News data loader for FMP API - handles all news-related endpoints
"""
import time
from typing import List, Dict, Any
from config import CONFIG
from utils.simple_logger import log_debug, log_error
from .base_fmp_loader import BaseFMPLoader


class NewsDataLoader(BaseFMPLoader):
    """Specialized loader for news-related FMP endpoints."""
    
    def get_comprehensive_news(self) -> List[Dict[str, Any]]:
        """Get comprehensive news from all news endpoints."""
        all_articles = []
        
        news_sources = [
            ("RSS feed", self._get_rss_news),
            ("Stock news", self._get_stock_news),
            ("FMP articles", self._get_fmp_articles),
            ("Press releases", self._get_press_releases),
            ("Company news", self._get_company_specific_news),
        ]
        
        for source_name, source_func in news_sources:
            try:
                articles = source_func()
                if articles:
                    all_articles.extend(articles)
                    log_debug(f"{source_name}: {len(articles)} articles")
            except Exception as e:
                log_debug(f"Error getting {source_name}: {e}")
        
        return all_articles
    
    def _get_rss_news(self) -> List[Dict[str, Any]]:
        """Get RSS news feed."""
        try:
            articles = []
            
            for page in range(CONFIG.news_page_limit):
                params = {
                    "page": page,
                    "limit": CONFIG.news_per_page_limit,
                }
                
                data = self._make_request("stock-news-sentiments-rss-feed", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                articles.extend(data)
                
                if len(data) < CONFIG.news_per_page_limit:
                    break
                
                time.sleep(0.2)
            
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
                
                for article in data:
                    if self._is_relevant_news(article):
                        transformed = self._transform_article(article, 'stock_news')
                        articles.append(transformed)
                
                if len(data) < params['limit']:
                    break
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_debug(f"Stock news not available: {e}")
            return []
    
    def _get_fmp_articles(self) -> List[Dict[str, Any]]:
        """Get FMP curated articles."""
        try:
            articles = []
            
            for page in range(min(3, CONFIG.news_page_limit)):
                params = {
                    "page": page,
                    "size": min(20, CONFIG.news_per_page_limit)
                }
                
                data = self._make_request("fmp/articles", params, use_v4=True)
                
                if not data or not isinstance(data, list):
                    break
                
                for article in data:
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
            log_debug(f"FMP articles not available: {e}")
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
                        transformed = self._transform_article(article, 'press_release')
                        articles.append(transformed)
                
                time.sleep(0.2)
            
            return articles
            
        except Exception as e:
            log_debug(f"Press releases not available: {e}")
            return []
    
    def _get_company_specific_news(self) -> List[Dict[str, Any]]:
        """Get company-specific news for universe symbols."""
        try:
            articles = []
            
            # Get universe symbols (limit to top 50 for API efficiency)
            universe_symbols = getattr(self, '_universe_cache', [])[:50]
            
            for symbol in universe_symbols:
                try:
                    params = {"limit": 5}  # Limit per symbol
                    data = self._make_request(f"stock_news/{symbol}", params)
                    
                    if data and isinstance(data, list):
                        for article in data:
                            transformed = self._transform_article(article, 'company_news')
                            transformed['symbol'] = symbol  # Ensure symbol is set
                            articles.append(transformed)
                    
                    time.sleep(0.1)  # Small delay between symbol requests
                    
                except Exception as e:
                    log_debug(f"Error getting news for {symbol}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            log_debug(f"Company-specific news not available: {e}")
            return []
    
    def set_universe_cache(self, universe: List[str]) -> None:
        """Set universe cache for company-specific news fetching."""
        self._universe_cache = universe[:100]  # Cache top 100 symbols
    
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