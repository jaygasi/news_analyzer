"""
Simplified news fetcher using date-based filtering instead of complex time tracking
Python 3.13.3 compatible
"""
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import pandas as pd
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug
from config import Config


class NewsFetcher(BaseFMPLoader):
    """Fetch financial news using simple date-based filtering"""

    def __init__(self, api_key: str) -> None:
        """Initialize news fetcher"""
        super().__init__(api_key)
        # Use a rolling window approach instead of complex time tracking
        self.lookback_days = 2  # Get news from last 2 days to ensure we don't miss anything

    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all sources using date-based filtering"""
        all_news = []
        
        # Calculate date range (simple approach)
        to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        from_date = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        log_info(f"📰 Fetching news from {from_date} to {to_date}")

        # Define news sources with date filtering and NO limits
        news_sources = [
            ("General Stock News", lambda: self._fetch_stock_news(from_date, to_date)),
            ("Press Releases", lambda: self._fetch_press_releases(from_date, to_date)),
            ("Earnings News", lambda: self._fetch_earnings_news(from_date, to_date)),
            ("Market News", lambda: self._fetch_market_news(from_date, to_date))
        ]

        for source_name, fetch_method in news_sources:
            try:
                articles = fetch_method()
                if articles:
                    all_news.extend(articles)
                    log_info(f"Fetched {len(articles)} articles from {source_name}")
            except Exception as e:
                log_error(f"Error fetching from {source_name}: {e}")

        # Remove duplicates
        all_news = self._deduplicate_articles(all_news)

        log_info(f"📰 Total articles before processing filter: {len(all_news)}")
        return all_news

    def _fetch_stock_news(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch general stock news with date filtering"""
        try:
            # Remove limit, add date filtering
            params = {
                "from": from_date,
                "to": to_date
            }
            data = self.make_request("stock_news", params)

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

    def _fetch_press_releases(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch press releases with date filtering"""
        try:
            # Remove limit, add date filtering
            params = {
                "from": from_date,
                "to": to_date
            }
            data = self.make_request("press-releases", params)

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

    def _fetch_earnings_news(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch earnings-related news with date filtering"""
        try:
            log_debug(f"Fetching earnings calendar from {from_date} to {to_date}")

            # Remove limit, use date filtering
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

                # Validate symbol
                symbol = str(item['symbol']).upper().strip()
                if not symbol or len(symbol) > 5:
                    continue

                # Create synthetic news article from earnings data
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

    def _fetch_market_news(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch general market news with date filtering"""
        try:
            # Remove limit, add date filtering if API supports it
            # Note: Check if general-news API supports date filtering
            params = {
                "from": from_date,
                "to": to_date
            }
            
            data = self.make_request("general-news", params, use_v4=True)

            if not data or not isinstance(data, list):
                return []

            articles = []
            for item in data:
                if self._is_valid_article(item):
                    # Intelligent ticker assignment for market news
                    ticker = self._assign_market_news_ticker(item)
                    if ticker:
                        item['symbol'] = ticker
                        articles.append(self._normalize_article(item, "market_news"))

            return articles

        except Exception as e:
            log_debug(f"General news not available or doesn't support date filtering: {e}")
            return []

    def _assign_market_news_ticker(self, article: Dict[str, Any]) -> str:
        """Intelligently assign ticker to market news based on content"""
        title = str(article.get('title', '')).lower()
        text = str(article.get('text', '')).lower()
        content = f"{title} {text}"

        # Look for explicit ticker mentions first
        import re
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, article.get('title', '') + ' ' + article.get('text', ''))

        # Filter to valid stock tickers
        valid_tickers = [t for t in potential_tickers if len(t) <= 5 and t not in ['NYSE', 'NASDAQ', 'SEC', 'FDA', 'CEO', 'CFO']]
        if valid_tickers:
            return valid_tickers[0]

        # Category-based assignment for general market news
        if any(word in content for word in ['federal reserve', 'fed', 'interest rate', 'monetary policy']):
            return 'TLT'  # Treasury bonds for Fed news
        elif any(word in content for word in ['s&p 500', 'market index', 'broad market']):
            return 'SPY'  # S&P 500 ETF
        elif any(word in content for word in ['technology', 'tech stocks', 'nasdaq']):
            return 'QQQ'  # Tech-heavy NASDAQ ETF
        elif any(word in content for word in ['small cap', 'russell']):
            return 'IWM'  # Small cap ETF
        elif any(word in content for word in ['volatility', 'vix', 'fear']):
            return 'VIX'  # Volatility index
        elif any(word in content for word in ['oil', 'energy', 'crude']):
            return 'XLE'  # Energy sector ETF
        elif any(word in content for word in ['gold', 'precious metals']):
            return 'GLD'  # Gold ETF
        else:
            # If no specific category, skip this article
            log_debug(f"Skipping market news without clear ticker assignment: {title[:50]}...")
            return None

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
            'original_data': article
        }

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title and symbol"""
        seen = set()
        unique_articles = []

        for article in articles:
            key = f"{article['symbol']}:{article['title'][:100]}"

            if key not in seen:
                seen.add(key)
                unique_articles.append(article)

        log_debug(f"Deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles