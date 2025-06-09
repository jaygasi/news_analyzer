"""
Enhanced news fetcher with earnings transcript analysis capability
Python 3.13.3 compatible - FIXED VERSION
"""
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import time
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config


class NewsFetcher(BaseFMPLoader):
    """Enhanced news fetcher with comprehensive debugging and earnings transcript integration"""

    def __init__(self, api_key: str) -> None:
        """Initialize news fetcher (simplified)"""
        super().__init__(api_key)
        self.lookback_days = 3  # Increased from 2 to 3 days for better coverage
        log_info("✅ News fetcher initialized")

    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all sources including earnings transcripts with comprehensive error handling"""
        all_news = []
        
        # Calculate date range
        to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        from_date = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        log_info(f"📰 Fetching news from {from_date} to {to_date} ({self.lookback_days} day window)")

        # Define news sources with multiple approaches for each
        news_source_mapping = {
            'stock-news': ("General Stock News", lambda: self._fetch_stock_news_multi_approach(from_date, to_date)),
            'press-releases': ("Press Releases", lambda: self._fetch_press_releases_multi_approach(from_date, to_date)),
            'earnings': ("Earnings Calendar", lambda: self._fetch_earnings_news(from_date, to_date)),
            'general-news': ("Market News", lambda: self._fetch_market_news_multi_approach(from_date, to_date)),
        }

        news_sources = []
        for source in Config.NEWS_SOURCES:
            source = source.strip()
            if source in news_source_mapping:
                news_sources.append(news_source_mapping[source])
            else:
                log_warning(f"Unknown news source in config: {source}")

        if not news_sources:
            log_warning("No valid news sources configured, using defaults")
            news_sources = list(news_source_mapping.values())

        # Fetch from each source with detailed logging
        for source_name, fetch_method in news_sources:
            try:
                log_info(f"🔍 Fetching from {source_name}...")
                start_time = time.time()
                
                articles = fetch_method()
                
                fetch_time = time.time() - start_time
                
                if articles and len(articles) > 0:
                    log_info(f"✅ {source_name}: {len(articles)} articles ({fetch_time:.1f}s)")
                    all_news.extend(articles)
                else:
                    log_info(f"📭 {source_name}: No articles found ({fetch_time:.1f}s)")
                    
            except Exception as e:
                log_error(f"❌ Error fetching {source_name}: {e}")
                continue
        
        # Final comprehensive summary by source type
        source_summary = {}
        for article in all_news:
            source = article.get('source', 'unknown')
            source_summary[source] = source_summary.get(source, 0) + 1
        
        log_info(f"📊 Final summary: {len(all_news)} articles from {len(news_sources)} sources")
        for source, count in source_summary.items():
            log_info(f"   📰 {source}: {count} articles")
        
        return all_news

    # === STOCK NEWS (ENHANCED) ===
    
    def _fetch_stock_news_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Enhanced stock news fetching with multiple API approaches"""
        all_articles = []
        
        # Approach 1: Direct stock news API (most reliable)
        try:
            log_debug("Trying stock_news API...")
            stock_data = self.make_request("stock_news", {
                "from": from_date,
                "to": to_date,
                "limit": 1000
            })
            
            if stock_data and isinstance(stock_data, list):
                articles = self._process_stock_news_data(stock_data, "stock_news")
                if articles:
                    log_info(f"   ✅ Stock News success with stock_news: {len(articles)} articles")
                    all_articles.extend(articles)
            
        except Exception as e:
            log_debug(f"Stock news API failed: {e}")
        
        # Approach 2: Alternative stock news endpoint
        if not all_articles:
            try:
                log_debug("Trying stock_news_v3_dated API...")
                alt_data = self.make_request("stock_news_v3_dated", {
                    "from": from_date,
                    "to": to_date
                })
                
                if alt_data and isinstance(alt_data, list):
                    articles = self._process_stock_news_data(alt_data, "stock_news")
                    if articles:
                        log_info(f"   ✅ Stock News success with stock_news_v3_dated: {len(articles)} articles")
                        all_articles.extend(articles)
                
            except Exception as e:
                log_debug(f"Alternative stock news API failed: {e}")
        
        return all_articles
    
    def _process_stock_news_data(self, data: Any, source_type: str) -> List[Dict[str, Any]]:
        """Process stock news data with validation"""
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data:
            if self._is_valid_article(item) and self._is_article_recent(item):
                articles.append(self._normalize_article(item, source_type))
        
        return articles

    # === PRESS RELEASES (ENHANCED) ===
    
    def _fetch_press_releases_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Enhanced press release fetching with multiple API approaches"""
        all_articles = []
        
        # Approach 1: Direct press releases API
        try:
            log_debug("Trying press_releases API...")
            pr_data = self.make_request("press-releases", {
                "from": from_date,
                "to": to_date,
                "limit": 500
            })
            
            if pr_data and isinstance(pr_data, list):
                articles = self._process_press_release_data(pr_data, "press_release")
                if articles:
                    log_info(f"   ✅ Press Releases success with press_releases: {len(articles)} articles")
                    all_articles.extend(articles)
                    
        except Exception as e:
            log_debug(f"Press releases API failed: {e}")
        
        # Approach 2: Alternative press releases endpoint
        if not all_articles:
            try:
                log_debug("Trying press_releases_v3_dated API...")
                alt_data = self.make_request("press_releases_v3_dated", {
                    "from": from_date,
                    "to": to_date
                })
                
                if alt_data and isinstance(alt_data, list):
                    articles = self._process_press_release_data(alt_data, "press_release")
                    if articles:
                        log_info(f"   ✅ Press Releases success with press_releases_v3_dated: {len(articles)} articles")
                        all_articles.extend(articles)
                
            except Exception as e:
                log_debug(f"Alternative press releases API failed: {e}")
        
        return all_articles
    
    def _process_press_release_data(self, data: Any, source_type: str) -> List[Dict[str, Any]]:
        """Process press release data with validation"""
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data:
            if self._is_valid_article(item) and self._is_article_recent(item):
                articles.append(self._normalize_article(item, source_type))
        
        return articles

    # === MARKET NEWS (ENHANCED) ===
    
    def _fetch_market_news_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Enhanced general market news fetching"""
        all_articles = []
        
        # Approach 1: General news with ticker assignment
        try:
            log_debug("Trying general_news API...")
            general_data = self.make_request("general_news", {
                "from": from_date,
                "to": to_date,
                "limit": 200
            })
            
            if general_data and isinstance(general_data, list):
                articles = self._process_general_news_data(general_data, "market_news")
                if articles:
                    log_info(f"   ✅ Market News success with general_news: {len(articles)} articles")
                    all_articles.extend(articles)
                    
        except Exception as e:
            log_debug(f"General news API failed: {e}")
        
        # Approach 2: Alternative general news endpoint  
        if not all_articles:
            try:
                log_debug("Trying general_news_v4_dated API...")
                alt_data = self.make_request("general_news_v4_dated", {
                    "from": from_date,
                    "to": to_date
                })
                
                if alt_data and isinstance(alt_data, list):
                    articles = self._process_general_news_data(alt_data, "market_news")
                    if articles:
                        log_info(f"   ✅ Market News success with general_news_v4_dated: {len(articles)} articles")
                        all_articles.extend(articles)
                
            except Exception as e:
                log_debug(f"Alternative general news API failed: {e}")
        
        # Approach 3: News for specific major tickers
        major_tickers = ['SPY', 'QQQ', 'DIA', 'XLF', 'XLK', 'XLV', 'XLE', 'XLY', 'XRT']
        for symbol in major_tickers:
            try:
                ticker_data = self.make_request(f"stock_news", {
                    "tickers": symbol,
                    "from": from_date,
                    "to": to_date,
                    "limit": 50
                })
                
                if ticker_data and isinstance(ticker_data, list):
                    for item in ticker_data:
                        if self._is_valid_article(item) and self._is_article_recent(item):
                            # Override symbol to ensure market news gets proper ticker
                            item['symbol'] = symbol
                            articles.append(self._normalize_article(item, "market_news"))
                
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                log_debug(f"Failed to fetch news for {symbol}: {e}")
                continue
        
        return all_articles
    
    def _process_general_news_data(self, data: Any, source_type: str) -> List[Dict[str, Any]]:
        """Process general news data and assign tickers intelligently"""
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data:
            if self._is_valid_article(item) and self._is_article_recent(item):
                # Try to assign a ticker based on content
                ticker = self._assign_market_news_ticker(item)
                if ticker:
                    item['symbol'] = ticker
                    articles.append(self._normalize_article(item, "market_news"))
        
        return articles

    # === EARNINGS NEWS (EXISTING - WORKING) ===
    
    def _fetch_earnings_news(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch earnings calendar events (this is working fine)"""
        try:
            log_debug(f"Fetching earnings calendar from {from_date} to {to_date}")
            
            earnings_data = self.make_request("earning_calendar", {
                "from": from_date,
                "to": to_date
            })

            if not earnings_data or not isinstance(earnings_data, list):
                log_debug("No earnings calendar data returned")
                return []

            log_info(f"FMP returned {len(earnings_data)} earnings events for date range")

            articles = []
            for item in earnings_data:
                if not item.get('symbol') or not item.get('date'):
                    continue

                symbol = str(item['symbol']).upper().strip()
                if not symbol or len(symbol) > 5:
                    continue

                article = {
                    'symbol': symbol,
                    'title': f"Earnings Call: {symbol} Q{item.get('quarter', '?')} {item.get('year', '')}",
                    'text': f"Earnings call scheduled for {item.get('date', 'TBD')}. "
                           f"Expected EPS: {item.get('epsEstimated', 'N/A')}, "
                           f"Revenue Estimate: {item.get('revenueEstimated', 'N/A')}. "
                           f"Time: {item.get('time', 'TBD')}.",
                    'url': f"earnings_calendar_{symbol}_{item.get('date', 'unknown')}",
                    'publishedDate': item.get('date', datetime.now(timezone.utc).isoformat()),
                    'source': 'earnings'
                }
                articles.append(article)

            log_info(f"Created {len(articles)} earnings articles")
            return articles

        except Exception as e:
            log_error(f"Error fetching earnings calendar: {e}")
            return []

    # === UTILITY METHODS ===
    
    def _assign_market_news_ticker(self, article: Dict[str, Any]) -> str:
        """Intelligently assign ticker symbols to general market news"""
        title = article.get('title', '').upper()
        text = article.get('text', '').upper()
        content = f"{title} {text}"
        
        # Market-wide indicators
        market_indicators = {
            'SPY': ['S&P 500', 'S&P500', 'SP500', 'MARKET INDEX', 'BROAD MARKET'],
            'QQQ': ['NASDAQ', 'TECH INDEX', 'TECHNOLOGY SECTOR'],
            'DIA': ['DOW JONES', 'DJIA', 'DOW INDUSTRIAL'],
            'XLF': ['FINANCIAL SECTOR', 'BANK SECTOR', 'FINANCE'],
            'XLK': ['TECHNOLOGY SECTOR', 'TECH SECTOR'],
            'XLV': ['HEALTHCARE SECTOR', 'HEALTH SECTOR'],
            'XLE': ['ENERGY SECTOR', 'OIL SECTOR'],
            'XLY': ['CONSUMER SECTOR', 'RETAIL SECTOR'],
            'XRT': ['RETAIL SECTOR', 'RETAIL ETF']
        }
        
        # Check for market indicators
        for ticker, indicators in market_indicators.items():
            if any(indicator in content for indicator in indicators):
                return ticker
        
        # Default to SPY for general market news
        market_keywords = ['MARKET', 'STOCK', 'TRADING', 'WALL STREET', 'INVESTOR']
        if any(keyword in content for keyword in market_keywords):
            return 'SPY'
        
        return None  # No ticker assigned
    
    def _is_valid_article(self, article: Dict[str, Any]) -> bool:
        """Validate article has required fields"""
        required_fields = ['title']
        
        for field in required_fields:
            if not article.get(field):
                return False
        
        # Check minimum content length
        title = article.get('title', '')
        text = article.get('text', '')
        total_content = len(title) + len(text)
        
        return total_content >= 20  # Minimum content threshold
    
    def _is_article_recent(self, article: Dict[str, Any]) -> bool:
        """Check if article is within the desired time window"""
        try:
            published_date = article.get('publishedDate')
            if not published_date:
                return True  # Include if no date available
            
            # Parse the date
            if isinstance(published_date, str):
                # Handle different date formats
                try:
                    pub_datetime = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                except:
                    # Try parsing as date only
                    pub_datetime = datetime.strptime(published_date[:10], '%Y-%m-%d')
                    pub_datetime = pub_datetime.replace(tzinfo=timezone.utc)
            else:
                return True
            
            # Check if within lookback window
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            return pub_datetime >= cutoff_time
            
        except Exception as e:
            log_debug(f"Date parsing error for article: {e}")
            return True  # Include if date parsing fails
    
    def _normalize_article(self, article: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Normalize article format across different API responses"""
        return {
            'symbol': article.get('symbol', article.get('ticker', 'UNKNOWN')),
            'title': article.get('title', ''),
            'text': article.get('text', article.get('summary', article.get('description', ''))),
            'url': article.get('url', article.get('link', '')),
            'publishedDate': article.get('publishedDate', article.get('date', datetime.now(timezone.utc).isoformat())),
            'source': source_type
        }
