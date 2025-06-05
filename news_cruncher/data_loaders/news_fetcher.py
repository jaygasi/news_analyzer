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
        # Remove max_articles limit from here - will be applied after filtering processed articles

    def fetch_all_news_since(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Fetch news from all sources since specific cutoff time"""
        all_news = []

        # Define news sources and their fetch methods - now pass cutoff_time to earnings
        news_sources = [
            ("General Stock News", lambda: self._fetch_stock_news()),
            ("Press Releases", lambda: self._fetch_press_releases()),
            ("Earnings News", lambda: self._fetch_earnings_news(cutoff_time)),  # Pass cutoff_time
            ("Market News", lambda: self._fetch_market_news())
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

        # Filter by dynamic cutoff time with more flexible logic
        all_news = self._filter_articles_since(all_news, cutoff_time)

        log_info(f"📰 Total articles after filtering: {len(all_news)} (fetched since {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')})")
        return all_news

    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all available FMP sources"""
        all_news = []

        # Define news sources and their fetch methods
        news_sources = [
            ("General Stock News", self._fetch_stock_news),
            ("Press Releases", self._fetch_press_releases),
            ("Earnings News", lambda: self._fetch_earnings_news()),  # No cutoff for legacy method
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

        log_info(f"Total articles fetched (before processing filter): {len(all_news)}")
        return all_news

    def _filter_articles_since(self, articles: List[Dict[str, Any]], cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Filter articles to only include those since cutoff time with debugging"""
        if not articles:
            return []

        filtered_articles = []
        too_old_count = 0
        unparseable_or_missing_date_included_count = 0
        kept_by_date_count = 0

        date_samples = []

        # More flexible cutoff - use 2 hours minimum instead of 30 minutes for news that might be delayed
        flexible_cutoff = min(cutoff_time, datetime.now(timezone.utc) - timedelta(hours=2))

        # Check if the original cutoff_time is very recent (within 1 hour)
        is_cutoff_very_recent = (datetime.now(timezone.utc) - cutoff_time).total_seconds() < 3600

        # SonarLint S3457 fix: Changed f-string to .format()
        log_debug("Time filtering: strict cutoff={}, flexible cutoff={}, is_cutoff_very_recent={}".format(
            cutoff_time, flexible_cutoff, is_cutoff_very_recent))

        for article in articles:
            pub_date_str = article.get('publishedDate', '')
            pub_date = None # Initialize to None
            # Initialize formatted string for logging/debugging, handles unparseable cases
            formatted_pub_date = 'N/A (unparseable)'

            if pub_date_str:
                try:
                    parsed_dt = pd.to_datetime(pub_date_str, utc=True)
                    if pd.notna(parsed_dt):
                        pub_date = parsed_dt # Assign only if valid
                        # Pylance fix: Ensure pub_date is not None before calling strftime
                        formatted_pub_date = pub_date.strftime('%Y-%m-%d %H:%M:%S UTC')
                except Exception:
                    pass # pub_date remains None, formatted_pub_date remains 'N/A'

            # --- Cognitive Complexity Fix: Streamlined inclusion logic ---
            should_add_article = False

            if pub_date is None: # Date was missing or unparseable
                if is_cutoff_very_recent:
                    should_add_article = True
                    unparseable_or_missing_date_included_count += 1
                # else: article is old and unparseable, so it's skipped
            elif pub_date >= flexible_cutoff: # Date was parsed and is recent enough
                should_add_article = True
                kept_by_date_count += 1
            else: # Date was parsed but is too old
                too_old_count += 1
                # should_add_article remains False, so article is skipped

            if should_add_article:
                filtered_articles.append(article)
                if len(date_samples) < 5: # Only collect samples for included articles
                    date_samples.append({
                        'title': article.get('title', '')[:50],
                        'pub_date_raw': pub_date_str,
                        'pub_date_parsed': formatted_pub_date, # Use the safely formatted date
                        'source': article.get('source', '')
                    })

        # SonarLint S3457 fix: Changed f-strings to .format()
        log_info("📅 Time filtering results:")
        log_info("   Input: {} articles".format(len(articles)))
        log_info("   Kept by date: {} (published since {})".format(kept_by_date_count, flexible_cutoff.strftime('%H:%M:%S')))
        log_info("   Too old: {}".format(too_old_count))
        log_info("   Missing/Unparseable dates (included if cutoff recent): {}".format(unparseable_or_missing_date_included_count))
        log_info("   Output: {} articles".format(len(filtered_articles)))

        # Show sample publication dates for debugging
        if date_samples:
            log_debug("📅 Sample article dates (from included articles):")
            for sample in date_samples:
                log_debug("   '{title}' - Raw: {pub_date_raw}, Parsed: {pub_date_parsed} ({source})".format(**sample))

        return filtered_articles

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

    def _fetch_earnings_news(self, cutoff_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch earnings-related news filtered by date"""
        try:
            # Use cutoff time if provided, otherwise default to 24 hours ago
            if cutoff_time is None:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=Config.DEFAULT_NEWS_AGE_HOURS)

            # Format dates for FMP API (YYYY-MM-DD format)
            from_date = cutoff_time.strftime('%Y-%m-%d')
            to_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')

            log_debug(f"Fetching earnings calendar from {from_date} to {to_date}")

            # Get earnings calendar with date filtering
            earnings_data = self.make_request("earning_calendar", {
                "from": from_date,
                "to": to_date,
                "limit": 50
            })

            if not earnings_data or not isinstance(earnings_data, list):
                log_debug("No earnings calendar data returned")
                return []

            log_info(f"FMP returned {len(earnings_data)} earnings events for date range {from_date} to {to_date}")

            articles = []
            processed_count = 0
            skipped_count = 0

            for item in earnings_data:
                processed_count += 1

                if not item.get('symbol') or not item.get('date'):
                    skipped_count += 1
                    continue

                # Validate symbol before creating article
                symbol = str(item['symbol']).upper().strip()
                if not symbol or len(symbol) > 5:
                    skipped_count += 1
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

            log_info(f"Earnings processing: {len(earnings_data)} events → {len(articles)} articles (processed: {processed_count}, skipped: {skipped_count})")
            return articles

        except Exception as e:
            log_error(f"Error fetching earnings news: {e}")
            return []

    def _fetch_market_news(self) -> List[Dict[str, Any]]:
        """Fetch general market news with intelligent ticker assignment"""
        try:
            data = self.make_request("general-news", {"limit": 50}, use_v4=True)

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
            log_debug(f"General news not available: {e}")
            return []

    def _assign_market_news_ticker(self, article: Dict[str, Any]) -> Optional[str]:
        """Intelligently assign ticker to market news based on content"""
        title = str(article.get('title', '')).lower()
        text = str(article.get('text', '')).lower()
        content = f"{title} {text}"

        # Look for explicit ticker mentions first
        import re
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, article.get('title', '') + ' ' + article.get('text', ''))

        # Filter to valid stock tickers (basic validation)
        valid_tickers = [t for t in potential_tickers if len(t) <= 5 and t not in ['NYSE', 'NASDAQ', 'SEC', 'FDA', 'CEO', 'CFO']]
        if valid_tickers:
            return valid_tickers[0]  # Return first valid ticker found

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
            # If no specific category, skip this article rather than forcing to SPY
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
            'original_data': article  # Keep original for debugging
        }

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title and symbol"""
        seen = set()
        unique_articles = []

        for article in articles:
            # SonarLint S3457 fix: Changed f-string to .format()
            key = "{}:{}".format(article['symbol'], article['title'][:100])

            if key not in seen:
                seen.add(key)
                unique_articles.append(article)

        log_debug(f"Deduplication: {len(articles)} -> {len(unique_articles)} articles")
        return unique_articles

    def _filter_recent_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles to only include recent ones"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=Config.DEFAULT_NEWS_AGE_HOURS)
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