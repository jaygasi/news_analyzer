"""
Enhanced news fetcher with earnings transcript analysis capability
Python 3.13.3 compatible
"""
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import pandas as pd
from data_loaders.base_fmp_loader import BaseFMPLoader
from data_loaders.earnings_transcript_fetcher import EarningsTranscriptFetcher  # NEW IMPORT
from utils.simple_logger import log_info, log_error, log_debug
from config import Config


class NewsFetcher(BaseFMPLoader):
    """Enhanced news fetcher with earnings transcript analysis"""

    def __init__(self, api_key: str) -> None:
        """Initialize news fetcher with earnings transcript capability"""
        super().__init__(api_key)
        # Use a rolling window approach instead of complex time tracking
        self.lookback_days = 2  # Get news from last 2 days to ensure we don't miss anything
        
        # NEW: Initialize earnings transcript fetcher
        self.earnings_fetcher = EarningsTranscriptFetcher(api_key)
        log_info("✅ Earnings transcript analysis capability initialized")

    def fetch_all_news(self) -> List[Dict[str, Any]]:
        """Fetch news from all sources including earnings transcripts"""
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
        
        # NEW: Add earnings transcripts if enabled
        if Config.ENABLE_EARNINGS_TRANSCRIPTS:
            news_sources.append(
                ("Earnings Transcripts", lambda: self._fetch_earnings_transcripts())
            )

        for source_name, fetch_method in news_sources:
            try:
                log_debug(f"Fetching from {source_name}...")
                articles = fetch_method()
                
                if articles:
                    log_info(f"✅ {source_name}: {len(articles)} articles")
                    all_news.extend(articles)
                else:
                    log_debug(f"❌ {source_name}: No articles")
                    
            except Exception as e:
                log_error(f"Error fetching {source_name}: {e}")
                continue

        log_info(f"📊 Total articles fetched: {len(all_news)}")
        return all_news

    def _fetch_stock_news(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Fetch general stock news with date filtering"""
        try:
            # Remove limit, add date filtering
            params = {
                "from": from_date,
                "to": to_date
            }
            
            data = self.make_request("stock-news", params, use_v4=True)

            if not data or not isinstance(data, list):
                return []

            articles = []
            for item in data:
                if self._is_valid_article(item) and item.get('symbol'):
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
            
            data = self.make_request("press-releases", params, use_v4=True)

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
        """Fetch earnings-related news with date filtering (earnings calendar)"""
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
    
    def _fetch_earnings_transcripts(self) -> List[Dict[str, Any]]:
        """NEW: Fetch and analyze actual earnings call transcripts"""
        try:
            if not Config.ENABLE_EARNINGS_TRANSCRIPTS:
                return []
            
            log_info("🎙️ Fetching earnings call transcripts...")
            
            # Get list of active tickers from recent news/market activity
            active_tickers = self._get_active_tickers_for_transcripts()
            
            transcript_articles = []
            transcripts_processed = 0
            max_transcripts = Config.MAX_EARNINGS_TRANSCRIPTS_PER_CYCLE
            
            for ticker in active_tickers:
                if transcripts_processed >= max_transcripts:
                    break
                
                try:
                    # Fetch recent transcripts (last 2 quarters)
                    analyses = self.earnings_fetcher.fetch_recent_transcripts(
                        ticker, 
                        lookback_quarters=2
                    )
                    
                    for analysis in analyses:
                        # Convert analysis to articles
                        articles = self.earnings_fetcher.create_earnings_articles(analysis)
                        transcript_articles.extend(articles)
                        transcripts_processed += 1
                        
                        log_debug(f"✅ Processed earnings transcript for {ticker} Q{analysis.quarter} {analysis.year}")
                        
                        # Respect rate limits
                        import time
                        time.sleep(0.5)
                        
                except Exception as e:
                    log_debug(f"Could not fetch transcript for {ticker}: {e}")
                    continue
            
            log_info(f"🎙️ Processed {transcripts_processed} earnings transcripts into {len(transcript_articles)} articles")
            return transcript_articles
            
        except Exception as e:
            log_error(f"Error fetching earnings transcripts: {e}")
            return []
    
    def _get_active_tickers_for_transcripts(self) -> List[str]:
        """Get list of active tickers that might have recent earnings transcripts"""
        
        # Priority tickers - major companies most likely to have transcripts
        priority_tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'CRM', 'ORCL', 'ADBE',
            
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK.B', 'V', 'MA', 'AXP',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
            
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'SBUX', 'MCD', 'DIS', 'NFLX',
            
            # Industrial
            'BA', 'CAT', 'GE', 'LMT', 'RTX', 'UPS', 'FDX', 'DE', 'MMM', 'HON'
        ]
        
        # Limit to reasonable number for API efficiency
        return priority_tickers[:Config.MAX_TICKERS_FOR_TRANSCRIPTS]

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
        title = article.get('title', '').upper()
        text = article.get('text', '').upper()
        content = f"{title} {text}"

        # Common ticker patterns in market news
        ticker_assignments = {
            # Major indices and ETFs
            'S&P 500': 'SPY',
            'S&P500': 'SPY', 
            'NASDAQ': 'QQQ',
            'DOW JONES': 'DIA',
            'RUSSELL': 'IWM',
            
            # Major companies frequently mentioned
            'APPLE': 'AAPL',
            'MICROSOFT': 'MSFT',
            'AMAZON': 'AMZN',
            'TESLA': 'TSLA',
            'GOOGLE': 'GOOGL',
            'META': 'META',
            'NVIDIA': 'NVDA',
            
            # Sectors
            'BANKS': 'XLF',
            'TECHNOLOGY': 'XLK',
            'ENERGY': 'XLE',
            'HEALTHCARE': 'XLV',
            'FINANCE': 'XLF'
        }

        for keyword, ticker in ticker_assignments.items():
            if keyword in content:
                return ticker

        return None

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
        if combined_length < Config.MIN_NEWS_LENGTH:
            return False

        return True

    def _normalize_article(self, article: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Normalize article structure"""
        return {
            'id': article.get('id', ''),
            'symbol': article.get('symbol', ''),
            'title': article.get('title', ''),
            'text': article.get('text', ''),
            'url': article.get('url', ''),
            'publishedDate': article.get('publishedDate', ''),
            'source': source_type
        }
    
    def get_earnings_analysis_for_ticker(self, ticker: str, quarters_back: int = 4) -> List[Dict[str, Any]]:
        """
        NEW: Get comprehensive earnings analysis for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            quarters_back: Number of quarters to analyze
            
        Returns:
            List of earnings analyses
        """
        try:
            if not Config.ENABLE_EARNINGS_TRANSCRIPTS:
                log_warning("Earnings transcripts are disabled in config")
                return []
            
            analyses = self.earnings_fetcher.fetch_recent_transcripts(ticker, quarters_back)
            
            # Convert to serializable format
            serializable_analyses = []
            for analysis in analyses:
                serializable_analyses.append({
                    'ticker': analysis.ticker,
                    'date': analysis.date,
                    'quarter': analysis.quarter,
                    'year': analysis.year,
                    'overall_sentiment': analysis.overall_sentiment,
                    'sentiment_confidence': analysis.sentiment_confidence,
                    'management_tone': analysis.management_tone,
                    'key_highlights': analysis.key_highlights,
                    'analyst_concerns': analysis.analyst_concerns,
                    'guidance_mentions': analysis.guidance_mentions,
                    'financial_metrics': analysis.financial_metrics,
                    'risk_factors': analysis.risk_factors,
                    'transcript_length': analysis.transcript_length
                })
            
            return serializable_analyses
            
        except Exception as e:
            log_error(f"Error getting earnings analysis for {ticker}: {e}")
            return []
