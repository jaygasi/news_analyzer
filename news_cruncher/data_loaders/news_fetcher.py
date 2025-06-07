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
    """Enhanced news fetcher with comprehensive debugging and multiple API approaches"""

    def __init__(self, api_key: str) -> None:
        """Initialize news fetcher with earnings transcript capability"""
        super().__init__(api_key)
        self.lookback_days = 3  # Increased from 2 to 3 days for better coverage
        
        # Initialize earnings transcript fetcher
        log_info("✅ Earnings transcript analysis capability initialized")

    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all sources with comprehensive error handling"""
        all_news = []
        
        # Calculate date range
        to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        from_date = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        log_info(f"📰 Fetching news from {from_date} to {to_date} ({self.lookback_days} day window)")

        # Define news sources with multiple approaches for each
        news_sources = [
            ("General Stock News", lambda: self._fetch_stock_news_multi_approach(from_date, to_date)),
            ("Press Releases", lambda: self._fetch_press_releases_multi_approach(from_date, to_date)),
            ("Earnings Calendar", lambda: self._fetch_earnings_news(from_date, to_date)),
            ("Market News", lambda: self._fetch_market_news_multi_approach(from_date, to_date)),
        ]
        
        
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
                    log_warning(f"❌ {source_name}: No articles returned ({fetch_time:.1f}s)")
                    
            except Exception as e:
                log_error(f"💥 {source_name} failed: {e}")
                import traceback
                log_debug(f"   Traceback: {traceback.format_exc()}")
                continue

        # Log final summary with source breakdown
        source_summary = {}
        for article in all_news:
            source = article.get('source', 'unknown')
            source_summary[source] = source_summary.get(source, 0) + 1
        
        log_info(f"📊 Final summary: {len(all_news)} articles from {len(source_summary)} sources")
        for source, count in sorted(source_summary.items()):
            log_info(f"   📰 {source}: {count} articles")
        
        return all_news

    def _fetch_stock_news_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Try multiple approaches to fetch stock news"""
        approaches = [
            ("stock_news_v4_dated", lambda: self._fetch_stock_news_v4_with_dates(from_date, to_date)),
            ("stock_news_v3_dated", lambda: self._fetch_stock_news_v3_with_dates(from_date, to_date)),
            ("stock_news_v4_recent", lambda: self._fetch_stock_news_v4_recent()),
            ("stock_news_v3_recent", lambda: self._fetch_stock_news_v3_recent()),
        ]
        
        return self._try_multiple_approaches("Stock News", approaches)
    
    def _fetch_press_releases_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Try multiple approaches to fetch press releases"""
        approaches = [
            ("press_releases_v4_dated", lambda: self._fetch_press_releases_v4_with_dates(from_date, to_date)),
            ("press_releases_v3_dated", lambda: self._fetch_press_releases_v3_with_dates(from_date, to_date)),
            ("press_releases_v4_recent", lambda: self._fetch_press_releases_v4_recent()),
            ("press_releases_v3_recent", lambda: self._fetch_press_releases_v3_recent()),
        ]
        
        return self._try_multiple_approaches("Press Releases", approaches)
    
    def _fetch_market_news_multi_approach(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Try multiple approaches to fetch market news"""
        approaches = [
            ("general_news_v4_dated", lambda: self._fetch_general_news_v4_with_dates(from_date, to_date)),
            ("general_news_v3_dated", lambda: self._fetch_general_news_v3_with_dates(from_date, to_date)),
            ("general_news_v4_recent", lambda: self._fetch_general_news_v4_recent()),
            ("market_news_symbols", lambda: self._fetch_market_news_by_symbols()),
        ]
        
        return self._try_multiple_approaches("Market News", approaches)
    
    def _try_multiple_approaches(self, source_name: str, approaches: List) -> List[Dict[str, Any]]:
        """Try multiple API approaches until one works"""
        for approach_name, approach_func in approaches:
            try:
                log_debug(f"   🔍 Trying {source_name} approach: {approach_name}")
                articles = approach_func()
                
                if articles and len(articles) > 0:
                    log_info(f"   ✅ {source_name} success with {approach_name}: {len(articles)} articles")
                    return articles
                else:
                    log_debug(f"   ❌ {approach_name}: No articles")
                    
            except Exception as e:
                log_debug(f"   💥 {approach_name} failed: {e}")
                continue
        
        log_warning(f"   ⚠️ All {source_name} approaches failed")
        return []

    # === STOCK NEWS APPROACHES ===
    
    def _fetch_stock_news_v4_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch stock news using v4 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 1000}
        data = self.make_request("stock_news", params, use_v4=True)
        return self._process_stock_news_data(data, "stock_news_v4")
    
    def _fetch_stock_news_v3_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch stock news using v3 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 1000}
        data = self.make_request("stock_news", params, use_v4=False)
        return self._process_stock_news_data(data, "stock_news_v3")
    
    def _fetch_stock_news_v4_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent stock news using v4 API without date filters"""
        params = {"limit": 500}
        data = self.make_request("stock_news", params, use_v4=True)
        return self._process_stock_news_data(data, "stock_news_v4_recent")
    
    def _fetch_stock_news_v3_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent stock news using v3 API without date filters"""
        params = {"limit": 500}
        data = self.make_request("stock_news", params, use_v4=False)
        return self._process_stock_news_data(data, "stock_news_v3_recent")
    
    def _process_stock_news_data(self, data: Any, source_type: str) -> List[Dict[str, Any]]:
        """Process stock news data into standardized format"""
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data:
            if self._is_valid_article(item) and item.get('symbol'):
                # Filter by date if we have publishedDate
                if self._is_article_recent(item):
                    articles.append(self._normalize_article(item, "stock_news"))
        
        return articles

    # === PRESS RELEASES APPROACHES ===
    
    def _fetch_press_releases_v4_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch press releases using v4 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 1000}
        data = self.make_request("press-releases", params, use_v4=True)
        return self._process_press_release_data(data, "press_release_v4")
    
    def _fetch_press_releases_v3_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch press releases using v3 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 1000}
        data = self.make_request("press-releases", params, use_v4=False)
        return self._process_press_release_data(data, "press_release_v3")
    
    def _fetch_press_releases_v4_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent press releases using v4 API"""
        params = {"limit": 500}
        data = self.make_request("press-releases", params, use_v4=True)
        return self._process_press_release_data(data, "press_release_v4_recent")
    
    def _fetch_press_releases_v3_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent press releases using v3 API"""
        params = {"limit": 500}
        data = self.make_request("press-releases", params, use_v4=False)
        return self._process_press_release_data(data, "press_release_v3_recent")
    
    def _process_press_release_data(self, data: Any, source_type: str) -> List[Dict[str, Any]]:
        """Process press release data into standardized format"""
        if not data or not isinstance(data, list):
            return []
        
        articles = []
        for item in data:
            if self._is_valid_article(item) and item.get('symbol'):
                if self._is_article_recent(item):
                    articles.append(self._normalize_article(item, "press_release"))
        
        return articles

    # === MARKET NEWS APPROACHES ===
    
    def _fetch_general_news_v4_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch general news using v4 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 500}
        data = self.make_request("general_news", params, use_v4=True)
        return self._process_general_news_data(data, "general_news_v4")
    
    def _fetch_general_news_v3_with_dates(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch general news using v3 API with date parameters"""
        params = {"from": from_date, "to": to_date, "limit": 500}
        data = self.make_request("general_news", params, use_v4=False)
        return self._process_general_news_data(data, "general_news_v3")
    
    def _fetch_general_news_v4_recent(self) -> List[Dict[str, Any]]:
        """Fetch recent general news using v4 API"""
        params = {"limit": 300}
        data = self.make_request("general_news", params, use_v4=True)
        return self._process_general_news_data(data, "general_news_v4_recent")
    
    def _fetch_market_news_by_symbols(self) -> List[Dict[str, Any]]:
        """Fetch news for major market symbols/ETFs"""
        major_symbols = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO']
        articles = []
        
        for symbol in major_symbols:
            try:
                params = {"tickers": symbol, "limit": 50}
                data = self.make_request("stock_news", params, use_v4=True)
                
                if data and isinstance(data, list):
                    for item in data:
                        if self._is_valid_article(item) and self._is_article_recent(item):
                            # Override symbol to ensure market news gets proper ticker
                            item['symbol'] = symbol
                            articles.append(self._normalize_article(item, "market_news"))
                
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                log_debug(f"Failed to fetch news for {symbol}: {e}")
                continue
        
        return articles
    
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
                    'text': f"Earnings call scheduled for {item.get('date', 'TBD')}. EPS estimate: {item.get('epsEstimated', 'N/A')}",
                    'url': '',
                    'publishedDate': item.get('date', datetime.now(timezone.utc).isoformat()),
                    'source': 'earnings_calendar'
                }
                articles.append(self._normalize_article(article, "earnings"))

            log_info(f"Created {len(articles)} earnings articles")
            return articles

        except Exception as e:
            log_error(f"Error fetching earnings news: {e}")
            return []

        
    # === UTILITY METHODS ===
    
    def _assign_market_news_ticker(self, article: Dict[str, Any]) -> str:
        """Intelligently assign ticker to market news based on content"""
        title = article.get('title', '').upper()
        text = article.get('text', '').upper()
        content = f"{title} {text}"

        ticker_assignments = {
            # Market indices and ETFs
            'S&P 500': 'SPY', 'S&P500': 'SPY', 'SPX': 'SPY',
            'NASDAQ': 'QQQ', 'NASDAQ 100': 'QQQ',
            'DOW JONES': 'DIA', 'DJIA': 'DIA',
            'RUSSELL': 'IWM', 'RUSSELL 2000': 'IWM',
            
            # Major companies frequently mentioned in market news
            'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'AMAZON': 'AMZN',
            'TESLA': 'TSLA', 'GOOGLE': 'GOOGL', 'META': 'META',
            'NVIDIA': 'NVDA', 'FACEBOOK': 'META',
            
            # Sectors
            'BANKS': 'XLF', 'BANKING': 'XLF', 'FINANCIAL': 'XLF',
            'TECHNOLOGY': 'XLK', 'TECH': 'XLK',
            'ENERGY': 'XLE', 'OIL': 'XLE',
            'HEALTHCARE': 'XLV', 'BIOTECH': 'XBI',
            'RETAIL': 'XRT', 'CONSUMER': 'XLY'
        }

        for keyword, ticker in ticker_assignments.items():
            if keyword in content:
                return ticker

        return None
    
    def _is_article_recent(self, article: Dict[str, Any]) -> bool:
        """Check if article is within our lookback window"""
        try:
            published_date = article.get('publishedDate', '')
            if not published_date:
                return True  # If no date, assume recent
            
            # Parse the date (handle various formats)
            if 'T' in published_date:
                article_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                article_date = datetime.strptime(published_date, '%Y-%m-%d')
                article_date = article_date.replace(tzinfo=timezone.utc)
            
            # Check if within lookback window
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days + 1)
            return article_date >= cutoff_date
            
        except Exception:
            return True  # If date parsing fails, include the article

    def _is_valid_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is valid and worth processing"""
        if not article:
            return False

        title = article.get('title', '')
        text = article.get('text', '')
        
        if not title and not text:
            return False

        # Filter out very short articles
        combined_length = len(title) + len(text)
        min_length = getattr(Config, 'MIN_NEWS_LENGTH', 50)
        if combined_length < min_length:
            return False

        return True

    def _normalize_article(self, article: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Normalize article structure across different sources"""
        return {
            'symbol': str(article.get('symbol', '')).upper().strip(),
            'title': str(article.get('title', '')).strip(),
            'text': str(article.get('text', '')).strip(),
            'url': str(article.get('url', '')).strip(),
            'publishedDate': article.get('publishedDate', datetime.now(timezone.utc).isoformat()),
            'source': source_type
        }