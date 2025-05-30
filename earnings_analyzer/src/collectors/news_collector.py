import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta, date
from config.config import Config
from src.database.connection import SessionLocal
from src.database.models import NewsArticle, Company, EarningsCalendar
from dataclasses import dataclass
from enum import Enum
import time
import re
from sqlalchemy import func
import socket

try:
    import requests
    import feedparser
    import yfinance as yf
    from bs4 import BeautifulSoup
except ImportError as e:
    logging.critical(f"❌ Missing critical dependency: {e}")
    logging.critical("Install with: pip install requests feedparser yfinance beautifulsoup4")
    raise ImportError(f"Missing critical dependency: {e}") from e

logger = logging.getLogger(__name__)

class TradingSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class EarningsEvent:
    ticker: str
    company_name: str
    date: datetime
    time: Optional[str] = None
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None
    source_api: Optional[str] = None

@dataclass
class NewsAnalysis:
    ticker: str
    sentiment_score: float
    trading_signal: TradingSignal
    confidence: float
    key_points: List[str]
    article_count: int

class EarningsCalendarFetcher:
    """Fetches earnings calendar data from multiple APIs"""
    
    # API endpoints
    FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    POLYGON_BASE_URL = "https://api.polygon.io/v3"
    TIINGO_BASE_URL = "https://api.tiingo.com"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': Config.USER_AGENT})
        self._log_available_apis()

    def _log_available_apis(self):
        """Log which APIs are configured"""
        apis = {
            'Finnhub': Config.FINNHUB_API_KEY,
            'FMP': Config.FMP_API_KEY,
            'Polygon': Config.POLYGON_API_KEY,
            'Tiingo': Config.TIINGO_API_KEY,
            'Alpha Vantage': Config.ALPHA_VANTAGE_API_KEY
        }
        
        configured_apis = [name for name, key in apis.items() if key]
        if configured_apis:
            logger.info(f"✅ Earnings APIs configured: {', '.join(configured_apis)}")
        else:
            logger.warning("⚠️ No earnings APIs configured - will use Yahoo Finance fallback")

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def fetch_earnings_calendar(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Fetch earnings calendar from multiple sources"""
        logger.info(f"Fetching earnings calendar for next {days_ahead} days...")
        
        api_config = Config.get_earnings_api_config()
        all_earnings: List[EarningsEvent] = []

        today_date = datetime.now().date()
        from_date_str = today_date.strftime('%Y-%m-%d')
        to_date_str = (today_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        # Try each API in priority order
        for api_name in api_config.get('priority_order', []):
            if not api_config.get('available_apis', {}).get(api_name):
                continue

            try:
                logger.info(f"Trying {api_name.upper()} API...")
                
                if api_name == 'finnhub':
                    results = self._fetch_finnhub_earnings(from_date_str, to_date_str)
                elif api_name == 'fmp':
                    results = self._fetch_fmp_earnings(from_date_str, to_date_str)
                elif api_name == 'polygon':
                    results = self._fetch_polygon_earnings(from_date_str, to_date_str)
                elif api_name == 'tiingo':
                    results = self._fetch_tiingo_earnings(from_date_str, to_date_str)
                elif api_name == 'alpha_vantage':
                    results = self._fetch_alpha_vantage_earnings(from_date_str, to_date_str)
                else:
                    continue
                
                if results:
                    all_earnings.extend(results)
                    logger.info(f"✅ Fetched {len(results)} events from {api_name.upper()}")
                    
                    if len(all_earnings) >= Config.MAX_EARNINGS_EVENTS_FROM_APIS:
                        logger.info(f"Target of {Config.MAX_EARNINGS_EVENTS_FROM_APIS} events reached")
                        break

            except Exception as e:
                logger.error(f"❌ Error with {api_name.upper()} API: {e}")
                continue

        # Use Yahoo Finance fallback if needed
        if len(all_earnings) < Config.MIN_EARNINGS_FOR_YAHOO_FALLBACK:
            logger.info("Using Yahoo Finance fallback...")
            try:
                yahoo_results = self._fetch_yahoo_earnings(days_ahead)
                if yahoo_results:
                    all_earnings.extend(yahoo_results)
                    logger.info(f"✅ Yahoo fallback added {len(yahoo_results)} events")
            except Exception as e:
                logger.error(f"❌ Yahoo fallback failed: {e}")

        # Remove duplicates
        unique_earnings = self._remove_duplicates(all_earnings)
        logger.info(f"📊 Final earnings calendar: {len(unique_earnings)} unique events")
        
        return unique_earnings

    def _remove_duplicates(self, earnings: List[EarningsEvent]) -> List[EarningsEvent]:
        """Remove duplicate earnings events"""
        seen_tickers = {}
        
        for event in earnings:
            if not event or not event.ticker:
                continue
                
            ticker = event.ticker.upper()
            
            if ticker not in seen_tickers:
                seen_tickers[ticker] = event
            else:
                # Keep the event with more information
                existing = seen_tickers[ticker]
                if (event.eps_estimate and not existing.eps_estimate) or \
                   (event.time and not existing.time):
                    seen_tickers[ticker] = event
        
        return list(seen_tickers.values())

    def _fetch_finnhub_earnings(self, from_date: str, to_date: str) -> List[EarningsEvent]:
        """Fetch from Finnhub API"""
        if not Config.FINNHUB_API_KEY:
            return []
            
        url = f"{self.FINNHUB_BASE_URL}/calendar/earnings"
        params = {
            'token': Config.FINNHUB_API_KEY,
            'from': from_date,
            'to': to_date
        }
        
        try:
            response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'earningsCalendar' not in data:
                return []
            
            events = []
            for item in data['earningsCalendar']:
                try:
                    date_str = item.get('date')
                    if not date_str:
                        continue
                        
                    earnings_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    event = EarningsEvent(
                        ticker=item['symbol'],
                        company_name=item.get('companyName', item.get('name', item['symbol'])),
                        date=earnings_date,
                        time=item.get('hour', '').upper() or None,
                        eps_estimate=self._safe_float(item.get('epsEstimate')),
                        revenue_estimate=self._safe_float(item.get('revenueEstimate')),
                        source_api='Finnhub'
                    )
                    events.append(event)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error parsing Finnhub event: {e}")
                    continue
            
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub API request failed: {e}")
            return []

    def _fetch_fmp_earnings(self, from_date: str, to_date: str) -> List[EarningsEvent]:
        """Fetch from Financial Modeling Prep API"""
        if not Config.FMP_API_KEY:
            return []
            
        url = f"{self.FMP_BASE_URL}/earning_calendar"
        params = {
            'apikey': Config.FMP_API_KEY,
            'from': from_date,
            'to': to_date
        }
        
        try:
            response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                return []
            
            events = []
            for item in data:
                try:
                    date_str = item.get('date')
                    if not date_str:
                        continue
                        
                    earnings_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    event = EarningsEvent(
                        ticker=item['symbol'],
                        company_name=item.get('companyName', item['symbol']),
                        date=earnings_date,
                        time=item.get('time', 'TAS').upper() if item.get('time') else None,
                        eps_estimate=self._safe_float(item.get('epsEstimated')),
                        revenue_estimate=self._safe_float(item.get('revenueEstimated')),
                        source_api='FMP'
                    )
                    events.append(event)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error parsing FMP event: {e}")
                    continue
            
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API request failed: {e}")
            return []

    def _fetch_polygon_earnings(self, from_date: str, to_date: str) -> List[EarningsEvent]:
        """Fetch from Polygon API"""
        if not Config.POLYGON_API_KEY:
            return []
            
        url = f"https://api.polygon.io/v2/reference/calendar"
        params = {
            'apiKey': Config.POLYGON_API_KEY,
            'from': from_date,
            'to': to_date,
            'type': 'earnings'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'results' not in data:
                return []
            
            events = []
            for item in data['results']:
                try:
                    date_str = item.get('date') or item.get('reportDate')
                    if not date_str:
                        continue
                        
                    if 'T' in date_str:
                        earnings_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        earnings_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    event = EarningsEvent(
                        ticker=item['ticker'],
                        company_name=item.get('name', item['ticker']),
                        date=earnings_date,
                        time=item.get('timeOfDay', 'TAS').upper() if item.get('timeOfDay') else None,
                        eps_estimate=self._safe_float(item.get('estimate', {}).get('eps')),
                        revenue_estimate=self._safe_float(item.get('estimate', {}).get('revenue')),
                        source_api='Polygon'
                    )
                    events.append(event)
                    
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Error parsing Polygon event: {e}")
                    continue
            
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Polygon API request failed: {e}")
            return []

    def _fetch_tiingo_earnings(self, from_date: str, to_date: str) -> List[EarningsEvent]:
        """Fetch from Tiingo API"""
        if not Config.TIINGO_API_KEY:
            return []
            
        url = f"{self.TIINGO_BASE_URL}/fundamentals/earnings"
        headers = {
            'Authorization': f'Token {Config.TIINGO_API_KEY}',
            'Content-Type': 'application/json'
        }
        params = {
            'startDate': from_date,
            'endDate': to_date
        }
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                return []
            
            events = []
            for item in data:
                try:
                    date_str = item.get('date')
                    if not date_str:
                        continue
                        
                    earnings_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    event = EarningsEvent(
                        ticker=item['ticker'],
                        company_name=item.get('name', item['ticker']),
                        date=earnings_date,
                        time=item.get('time', 'TAS').upper() if item.get('time') else None,
                        eps_estimate=self._safe_float(item.get('epsEstimate')),
                        revenue_estimate=self._safe_float(item.get('revenueEstimate')),
                        source_api='Tiingo'
                    )
                    events.append(event)
                    
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Error parsing Tiingo event: {e}")
                    continue
            
            return events
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Tiingo API request failed: {e}")
            return []

    def _fetch_alpha_vantage_earnings(self, from_date: str, to_date: str) -> List[EarningsEvent]:
        """Fetch from Alpha Vantage API"""
        if not Config.ALPHA_VANTAGE_API_KEY:
            return []
            
        # Alpha Vantage doesn't have a direct earnings calendar endpoint
        # This is a placeholder implementation
        logger.debug("Alpha Vantage earnings calendar not implemented")
        return []

    def _fetch_yahoo_earnings(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Fetch earnings from Yahoo Finance as fallback"""
        logger.info(f"Using Yahoo Finance fallback for next {days_ahead} days")
        
        tickers_to_check = Config.YAHOO_FALLBACK_TICKERS
        if not tickers_to_check:
            # Get tickers from database
            db = SessionLocal()
            try:
                companies = db.query(Company.ticker).limit(Config.MAX_YAHOO_TICKERS_TO_CHECK).all()
                tickers_to_check = [c.ticker for c in companies]
            except Exception as e:
                logger.error(f"Error getting tickers from DB: {e}")
                return []
            finally:
                db.close()

        if not tickers_to_check:
            logger.warning("No tickers available for Yahoo Finance fallback")
            return []

        events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)

        for ticker in tickers_to_check:
            try:
                stock = yf.Ticker(ticker)
                calendar_df = stock.calendar
                
                if calendar_df is not None and not calendar_df.empty:
                    if 'Earnings Date' in calendar_df.columns:
                        earnings_date_str = str(calendar_df.iloc[0]['Earnings Date'])
                        
                        # Parse earnings date
                        try:
                            if ' to ' in earnings_date_str:
                                earnings_date_str = earnings_date_str.split(' to ')[0]
                            
                            earnings_date = datetime.strptime(earnings_date_str.split()[0], '%Y-%m-%d')
                            
                            if today <= earnings_date.date() <= end_date:
                                info = stock.info
                                event = EarningsEvent(
                                    ticker=ticker,
                                    company_name=info.get('longName', ticker),
                                    date=earnings_date,
                                    eps_estimate=self._safe_float(info.get('forwardEps')),
                                    source_api='Yahoo Finance'
                                )
                                events.append(event)
                                
                        except (ValueError, AttributeError):
                            continue
                
                # Respect rate limits
                time.sleep(Config.YAHOO_API_CALL_DELAY_SECONDS)
                
            except Exception as e:
                logger.debug(f"Yahoo Finance error for {ticker}: {e}")
                continue

        logger.info(f"Yahoo Finance fallback found {len(events)} earnings events")
        return events

class EnhancedNewsCollector:
    """Enhanced news collector with multiple sources"""
    
    def __init__(self):
        self.rss_sources = Config.RSS_FEEDS
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': Config.USER_AGENT})
        self.processed_urls: Set[str] = set()
        
        # Keywords and indicators
        self.earnings_keywords = Config.EARNINGS_KEYWORDS
        self.positive_indicators = Config.POSITIVE_SENTIMENT_INDICATORS
        self.negative_indicators = Config.NEGATIVE_SENTIMENT_INDICATORS
        
        # Initialize earnings fetcher
        self._earnings_fetcher = EarningsCalendarFetcher()

    def fetch_comprehensive_earnings_calendar(self, days_ahead: int = 7) -> List[EarningsEvent]:
        """Fetch comprehensive earnings calendar"""
        return self._earnings_fetcher.fetch_earnings_calendar(days_ahead)

    def test_rss_feeds(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Test all RSS feeds for accessibility"""
        logger.info("🧪 Testing RSS feeds...")
        
        working_feeds = []
        failed_feeds = []
        
        for feed_name, feed_config in self.rss_sources.items():
            try:
                feed_url = feed_config['url']
                logger.debug(f"Testing RSS feed: {feed_name}")
                
                response = self.session.get(feed_url, timeout=Config.RSS_FEED_TIMEOUT_SECONDS)
                response.raise_for_status()
                
                parsed_feed = feedparser.parse(response.content)
                
                if parsed_feed.entries:
                    working_feeds.append(feed_name)
                    logger.debug(f"✅ {feed_name}: {len(parsed_feed.entries)} entries")
                else:
                    failed_feeds.append((feed_name, "No entries found"))
                    logger.warning(f"⚠️ {feed_name}: No entries")
                    
            except requests.exceptions.Timeout:
                failed_feeds.append((feed_name, "Timeout"))
                logger.error(f"❌ {feed_name}: Timeout")
            except requests.exceptions.RequestException as e:
                failed_feeds.append((feed_name, f"Request error: {e}"))
                logger.error(f"❌ {feed_name}: {e}")
            except Exception as e:
                failed_feeds.append((feed_name, f"Parse error: {e}"))
                logger.error(f"❌ {feed_name}: {e}")

        logger.info(f"RSS feed test results: {len(working_feeds)} working, {len(failed_feeds)} failed")
        return working_feeds, failed_feeds

    def store_earnings_calendar(self, earnings_events: List[EarningsEvent], db_session):
        """Store earnings calendar events in database"""
        stored_count = 0
        updated_count = 0
        
        try:
            for event in earnings_events:
                # Get or create company
                company = self.get_or_create_company(event.ticker, db_session)
                if not company:
                    logger.warning(f"Could not create company for {event.ticker}")
                    continue
                
                # Check for existing event
                existing_event = db_session.query(EarningsCalendar).filter(
                    EarningsCalendar.company_id == company.id,
                    EarningsCalendar.earnings_date == event.date.date()
                ).first()
                
                if existing_event:
                    # Update existing event
                    updated = False
                    if event.time and event.time != existing_event.earnings_time:
                        existing_event.earnings_time = event.time
                        updated = True
                    if event.eps_estimate and event.eps_estimate != existing_event.eps_estimate:
                        existing_event.eps_estimate = event.eps_estimate
                        updated = True
                    if event.revenue_estimate and event.revenue_estimate != existing_event.revenue_estimate:
                        existing_event.revenue_estimate = event.revenue_estimate
                        updated = True
                    if event.source_api and event.source_api != existing_event.source_api:
                        existing_event.source_api = event.source_api
                        updated = True
                    
                    if updated:
                        updated_count += 1
                        logger.debug(f"Updated earnings for {event.ticker} on {event.date.date()}")
                else:
                    # Create new event
                    new_event = EarningsCalendar(
                        company_id=company.id,
                        earnings_date=event.date,
                        earnings_time=event.time,
                        eps_estimate=event.eps_estimate,
                        revenue_estimate=event.revenue_estimate,
                        source_api=event.source_api
                    )
                    db_session.add(new_event)
                    stored_count += 1
                    logger.debug(f"Added earnings for {event.ticker} on {event.date.date()}")
            
            logger.info(f"Earnings calendar: {stored_count} new, {updated_count} updated")
            
        except Exception as e:
            logger.error(f"Error storing earnings calendar: {e}", exc_info=True)
            raise

    def get_or_create_company(self, ticker: str, db_session) -> Optional[Company]:
        """Get existing company or create new one"""
        # Try to find existing company
        company = db_session.query(Company).filter(Company.ticker == ticker.upper()).first()
        if company:
            return company
        
        logger.info(f"Creating new company record for {ticker}")
        
        try:
            # Get company info from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or not info.get('symbol'):
                # Create with minimal data
                company_name = ticker
                sector = "Unknown"
                industry = "Unknown"
                market_cap = 0
            else:
                company_name = info.get('longName', info.get('shortName', ticker))
                sector = info.get('sector', "Unknown")
                industry = info.get('industry', "Unknown")
                market_cap = info.get('marketCap', 0)
            
            new_company = Company(
                ticker=ticker.upper(),
                company_name=company_name,
                sector=sector,
                industry=industry,
                market_cap=market_cap
            )
            
            db_session.add(new_company)
            db_session.flush()  # Get the ID
            
            logger.info(f"Created company: {ticker} - {company_name}")
            return new_company
            
        except Exception as e:
            logger.warning(f"Error getting company info for {ticker}: {e}")
            
            # Create with minimal data as fallback
            fallback_company = Company(
                ticker=ticker.upper(),
                company_name=ticker,
                sector="Unknown",
                industry="Unknown",
                market_cap=0
            )
            
            db_session.add(fallback_company)
            db_session.flush()
            
            logger.info(f"Created minimal company record for {ticker}")
            return fallback_company

    def extract_ticker_from_text(self, text: str) -> Optional[str]:
        """Extract ticker symbol from text"""
        # Check company name mapping first
        text_lower = text.lower()
        for company_name, ticker in Config.COMPANY_NAME_TO_TICKER_MAP.items():
            if company_name.lower() in text_lower:
                return ticker.upper()
        
        # Use regex patterns to find tickers
        patterns = [
            r'\b(?:NASDAQ|NYSE|AMEX|OTC(?:BB|QX)?)\s*[:\-]\s*([A-Z]{1,5})\b',
            r'\(([A-Z]{1,5})\)',
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{3,5})\b',
            r'\b([A-Z]{2})\b'
        ]
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text)
                for match in matches:
                    ticker = match.upper().strip()
                    if (1 < len(ticker) <= 5 and 
                        ticker.isalpha() and 
                        ticker not in Config.TICKER_EXCLUSION_LIST):
                        return ticker
            except re.error as e:
                logger.error(f"Regex error: {e}")
                continue
        
        return None

    def _calculate_earnings_relevance_score(self, text: str, target_tickers: Set[str] = None) -> Tuple[float, bool]:
        """Calculate how relevant text is to earnings"""
        text_lower = text.lower()
        score = 0.0
        has_priority_keyword = False
        
        # Check earnings keywords
        for keyword, weight in self.earnings_keywords.items():
            if keyword.lower() in text_lower:
                score += weight
                if weight >= Config.EARNINGS_PRIORITY_KEYWORD_THRESHOLD:
                    has_priority_keyword = True
        
        # Bonus for target ticker match
        if target_tickers:
            extracted_ticker = self.extract_ticker_from_text(text)
            if extracted_ticker and extracted_ticker in target_tickers:
                score += Config.EARNINGS_TICKER_MATCH_BONUS
        
        # Calculate confidence
        confidence = min(1.0, score / Config.MAX_EARNINGS_RELEVANCE_SCORE) if Config.MAX_EARNINGS_RELEVANCE_SCORE > 0 else 0.0
        is_relevant = (score >= Config.MIN_EARNINGS_RELEVANCE_SCORE_THRESHOLD) or has_priority_keyword
        
        return confidence, is_relevant

    def analyze_sentiment_quick(self, title: str, content: str) -> Tuple[float, List[str]]:
        """Quick sentiment analysis using keyword matching"""
        text = f"{title}. {content}".lower()
        
        positive_score = sum(1 for indicator in self.positive_indicators if indicator in text)
        negative_score = sum(1 for indicator in self.negative_indicators if indicator in text)
        
        net_score = positive_score - negative_score
        
        # Convert to sentiment score
        if net_score > 0:
            sentiment = min(1.0, net_score * 0.2)
        elif net_score < 0:
            sentiment = max(-1.0, net_score * 0.2)
        else:
            sentiment = 0.0
        
        # Extract key phrases
        key_phrases = []
        try:
            sentences = re.split(r'[.!?]\s+', text)
            for sentence in sentences[:Config.MAX_KEY_PHRASES_QUICK_SENTIMENT]:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    for keyword in self.earnings_keywords:
                        if keyword.lower() in sentence:
                            key_phrases.append(sentence[:150] + "...")
                            break
        except Exception as e:
            logger.debug(f"Error extracting key phrases: {e}")
        
        return sentiment, key_phrases

    def _should_collect_article(self, earnings_confidence: float, is_earnings_relevant: bool,
                              extracted_ticker: Optional[str], target_tickers: Set[str]) -> bool:
        """Determine if article should be collected"""
        # Collect if earnings relevant and meets confidence threshold
        if is_earnings_relevant and earnings_confidence >= Config.MIN_EARNINGS_CONFIDENCE_TO_COLLECT:
            return True
        
        # Collect if ticker matches target
        if extracted_ticker and target_tickers and extracted_ticker in target_tickers:
            return True
        
        return False

    def _process_rss_entry(self, entry: feedparser.FeedParserDict, source_name: str, 
                          source_priority: str, target_tickers: Set[str]) -> Optional[Dict]:
        """Process a single RSS entry"""
        title = entry.get('title', '').strip()
        url = entry.get('link', '').strip()
        
        if not title or not url or url in self.processed_urls:
            return None
        
        logger.debug(f"Processing: {title[:100]}...")
        
        # Parse published date
        published_date = datetime.now()
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                published_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            except (TypeError, ValueError):
                pass
        
        # Skip old articles
        if (datetime.now() - published_date).days > Config.MAX_ARTICLE_AGE_DAYS:
            logger.debug(f"Skipping old article: {title[:50]}...")
            return None
        
        # Get content summary
        content_summary = ""
        if 'summary' in entry:
            content_summary = entry.summary
        elif 'description' in entry:
            content_summary = entry.description
        
        if content_summary:
            soup = BeautifulSoup(content_summary, 'html.parser')
            content_summary = soup.get_text(separator=' ', strip=True)
        
        # Analyze for earnings relevance
        full_text = f"{title} {content_summary}"
        earnings_confidence, is_earnings_relevant = self._calculate_earnings_relevance_score(
            full_text, target_tickers
        )
        
        # Extract ticker
        extracted_ticker = self.extract_ticker_from_text(full_text)
        
        # Decide if we should collect this article
        if not self._should_collect_article(earnings_confidence, is_earnings_relevant, 
                                          extracted_ticker, target_tickers):
            logger.debug(f"Article filtered out: {title[:50]}...")
            return None
        
        # Perform quick sentiment analysis
        sentiment_score, key_phrases = self.analyze_sentiment_quick(title, content_summary)
        
        article_data = {
            'title': title,
            'url': url,
            'content_summary': content_summary,
            'published_date': published_date,
            'source': source_name,
            'priority': source_priority,
            'extracted_ticker': extracted_ticker or "UNKNOWN",
            'initial_sentiment_score': sentiment_score,
            'key_phrases': key_phrases,
            'earnings_confidence': earnings_confidence,
            'is_earnings_relevant': is_earnings_relevant
        }
        
        return article_data

    def collect_from_rss_feed(self, feed_name: str, feed_config: Dict, 
                            target_tickers: Set[str]) -> List[Dict]:
        """Collect articles from a single RSS feed"""
        articles = []
        feed_url = feed_config['url']
        feed_priority = feed_config.get('priority', 'medium')
        
        logger.info(f"Collecting from RSS: {feed_name}")
        
        try:
            response = self.session.get(feed_url, timeout=Config.RSS_FEED_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            feed_data = feedparser.parse(response.content)
            
            if feed_data.bozo:
                logger.warning(f"Feed may be malformed: {feed_name}")
            
            if not feed_data.entries:
                logger.warning(f"No entries in feed: {feed_name}")
                return articles
            
            logger.debug(f"Processing {len(feed_data.entries)} entries from {feed_name}")
            
            for entry in feed_data.entries[:Config.MAX_ARTICLES_PER_RSS_FEED]:
                processed_article = self._process_rss_entry(entry, feed_name, feed_priority, target_tickers)
                if processed_article:
                    articles.append(processed_article)
                    self.processed_urls.add(processed_article['url'])
                    logger.debug(f"Collected: {processed_article['extracted_ticker']} - {processed_article['title'][:60]}...")
            
            logger.info(f"Collected {len(articles)} articles from {feed_name}")
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching RSS feed: {feed_name}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for RSS feed {feed_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_name}: {e}", exc_info=True)
        
        return articles

    def _fetch_finnhub_company_news(self, ticker: str, from_date: str, to_date: str) -> List[Dict]:
        """Fetch company-specific news from Finnhub"""
        if not Config.FINNHUB_API_KEY:
            return []
        
        url = f"{EarningsCalendarFetcher.FINNHUB_BASE_URL}/company-news"
        params = {
            'symbol': ticker,
            'from': from_date,
            'to': to_date,
            'token': Config.FINNHUB_API_KEY
        }
        
        articles = []
        
        try:
            response = self.session.get(url, params=params, timeout=Config.API_TIMEOUT_SECONDS)
            response.raise_for_status()
            news_items = response.json()
            
            if not news_items:
                return []
            
            for item in news_items:
                # Parse date
                published_date = datetime.now()
                if item.get('datetime'):
                    try:
                        published_date = datetime.fromtimestamp(item['datetime'])
                    except (ValueError, TypeError):
                        pass
                
                # Skip old articles
                if (datetime.now() - published_date).days > Config.MAX_ARTICLE_AGE_DAYS:
                    continue
                
                url = item.get('url', '').strip()
                if not url or url in self.processed_urls:
                    continue
                
                title = item.get('headline', 'No Title').strip()
                summary = item.get('summary', '').strip()
                
                # Analyze relevance
                full_text = f"{title} {summary}"
                earnings_confidence, is_earnings_relevant = self._calculate_earnings_relevance_score(
                    full_text, {ticker.upper()}
                )
                
                # Quick sentiment analysis
                sentiment_score, key_phrases = self.analyze_sentiment_quick(title, summary)
                
                article = {
                    'title': title,
                    'url': url,
                    'content_summary': summary,
                    'published_date': published_date,
                    'source': f"Finnhub-{item.get('source', 'Unknown')}",
                    'priority': 'high',
                    'extracted_ticker': ticker.upper(),
                    'initial_sentiment_score': sentiment_score,
                    'key_phrases': key_phrases,
                    'earnings_confidence': earnings_confidence,
                    'is_earnings_relevant': is_earnings_relevant
                }
                
                articles.append(article)
                self.processed_urls.add(url)
                logger.debug(f"Collected Finnhub news for {ticker}: {title[:50]}...")
            
            logger.info(f"Collected {len(articles)} Finnhub articles for {ticker}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub company news error for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error processing Finnhub news for {ticker}: {e}")
        
        return articles

    def _collect_news_from_finnhub_api(self, target_tickers: List[str]) -> Dict[str, List[Dict]]:
        """Collect news from Finnhub API for target tickers"""
        logger.info(f"Collecting Finnhub news for {len(target_tickers)} tickers")
        
        articles_map = {}
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=Config.NEWS_COLLECTION_DAYS_BACK)).strftime('%Y-%m-%d')
        
        for ticker in target_tickers:
            logger.debug(f"Fetching Finnhub news for {ticker}")
            
            ticker_news = self._fetch_finnhub_company_news(ticker, from_date, to_date)
            if ticker_news:
                articles_map.setdefault(ticker.upper(), []).extend(ticker_news)
            
            # Respect rate limits
            time.sleep(Config.API_CALL_DELAY_SECONDS)
        
        total_collected = sum(len(v) for v in articles_map.values())
        logger.info(f"Collected {total_collected} total articles from Finnhub API")
        
        return articles_map

    def collect_targeted_news(self, target_tickers: List[str]) -> Dict[str, List[Dict]]:
        """Collect targeted news from multiple sources"""
        logger.info(f"Collecting targeted news for: {', '.join(target_tickers)}")
        
        target_ticker_set = set(t.upper() for t in target_tickers)
        articles_map = {ticker: [] for ticker in target_ticker_set}
        articles_map["UNKNOWN"] = []
        
        # Step 1: Collect from RSS feeds
        logger.info("--- Starting RSS Feed Collection ---")
        
        sorted_rss_sources = sorted(
            self.rss_sources.items(),
            key=lambda item: Config.RSS_PRIORITY_ORDER.get(item[1].get('priority', 'low'), 99)
        )
        
        rss_collected_urls = set()
        
        for feed_name, feed_config in sorted_rss_sources:
            try:
                articles = self.collect_from_rss_feed(feed_name, feed_config, target_ticker_set)
                
                for article in articles:
                    ticker = article.get('extracted_ticker', 'UNKNOWN').upper()
                    
                    if ticker in articles_map:
                        articles_map[ticker].append(article)
                    elif article.get('is_earnings_relevant'):
                        articles_map.setdefault("UNKNOWN_BUT_RELEVANT", []).append(article)
                    
                    rss_collected_urls.add(article['url'])
                
                # Delay between feeds
                time.sleep(Config.RSS_INTER_FEED_DELAY_SECONDS)
                
            except Exception as e:
                logger.error(f"Error processing RSS source {feed_name}: {e}")
        
        logger.info("--- RSS Feed Collection Complete ---")
        
        # Step 2: Collect from Finnhub API
        if Config.FINNHUB_API_KEY:
            logger.info("--- Starting Finnhub API Collection ---")
            
            finnhub_articles = self._collect_news_from_finnhub_api(list(target_ticker_set))
            
            # Merge Finnhub articles, avoiding duplicates
            for ticker, articles in finnhub_articles.items():
                articles_map.setdefault(ticker.upper(), [])
                
                for article in articles:
                    if article['url'] not in rss_collected_urls:
                        # Check if already in ticker's articles
                        existing_urls = {a['url'] for a in articles_map[ticker.upper()]}
                        if article['url'] not in existing_urls:
                            articles_map[ticker.upper()].append(article)
            
            logger.info("--- Finnhub API Collection Complete ---")
        else:
            logger.info("Finnhub API key not configured, skipping")
        
        # Summary
        total_collected = sum(len(v) for v in articles_map.values())
        logger.info(f"📊 Total articles collected: {total_collected}")
        
        for ticker, articles in articles_map.items():
            if articles:
                logger.info(f"  {ticker}: {len(articles)} articles")
        
        return articles_map

    def enhance_article_content(self, url: str, existing_summary: str = "") -> str:
        """Enhance article content by fetching full text"""
        try:
            logger.debug(f"Enhancing content for: {url}")
            
            response = self.session.get(url, timeout=Config.ARTICLE_FETCH_TIMEOUT_SECONDS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove noise elements
            for selector in Config.HTML_NOISE_SELECTORS:
                for element in soup.select(selector):
                    element.decompose()
            
            # Try to find main content
            content_text = ""
            for selector in Config.HTML_CONTENT_SELECTORS:
                element = soup.select_one(selector)
                if element:
                    content_text = element.get_text(separator=' ', strip=True)
                    if len(content_text) > Config.MIN_FULL_ARTICLE_LENGTH_THRESHOLD:
                        break
            
            # Fallback to body
            if not content_text or len(content_text) < Config.MIN_FULL_ARTICLE_LENGTH_THRESHOLD:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator=' ', strip=True)
            
            # Clean up text
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            # Return enhanced content if significantly better
            if len(content_text) > len(existing_summary) + 100:
                logger.debug(f"Enhanced content length: {len(content_text)}")
                return content_text[:Config.MAX_ARTICLE_CONTENT_LENGTH]
            else:
                logger.debug("Enhanced content not significantly better, using summary")
                return existing_summary[:Config.MAX_ARTICLE_CONTENT_LENGTH]
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch full content for {url}: {e}")
        except Exception as e:
            logger.warning(f"Error enhancing content for {url}: {e}")
        
        return existing_summary[:Config.MAX_ARTICLE_CONTENT_LENGTH]

    def _store_single_article(self, article_data: Dict, db_session) -> bool:
        """Store a single article in the database"""
        try:
            # Check if article already exists
            existing = db_session.query(NewsArticle).filter(NewsArticle.url == article_data['url']).first()
            if existing:
                logger.debug(f"Article already exists: {article_data['url']}")
                return False
            
            # Get or create company
            company_id = None
            ticker = article_data.get('extracted_ticker')
            if ticker and ticker not in ["UNKNOWN", "UNKNOWN_BUT_RELEVANT"]:
                company = self.get_or_create_company(ticker, db_session)
                if company:
                    company_id = company.id
            
            # Enhance article content
            full_content = self.enhance_article_content(
                article_data['url'], 
                article_data['content_summary']
            )
            
            # Create new article
            new_article = NewsArticle(
                title=article_data['title'],
                content=full_content,
                url=article_data['url'],
                published_date=article_data['published_date'],
                source=article_data['source'],
                company_id=company_id,
                initial_sentiment_score=article_data.get('initial_sentiment_score'),
                earnings_confidence_score=article_data.get('earnings_confidence'),
                is_earnings_related=article_data.get('is_earnings_relevant', False),
                processed=False
            )
            
            db_session.add(new_article)
            logger.debug(f"Prepared article for storage: {ticker} - {article_data['title'][:50]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing article {article_data.get('url', 'N/A')}: {e}")
            return False

    def run_earnings_focused_collection(self) -> Dict:
        """Run complete earnings-focused news collection cycle"""
        logger.info("🎯 Starting earnings-focused collection cycle...")
        
        db = SessionLocal()
        stats = {
            'earnings_events_fetched': 0,
            'target_tickers_identified': 0,
            'preliminary_articles_collected': 0,
            'articles_stored_in_db': 0,
            'tickers_with_new_news': set()
        }
        
        try:
            # Step 1: Fetch earnings calendar
            logger.info("📅 Fetching upcoming earnings events...")
            earnings_events = self._earnings_fetcher.fetch_earnings_calendar(
                days_ahead=Config.OPERATIONAL_EARNINGS_DAYS_AHEAD
            )
            stats['earnings_events_fetched'] = len(earnings_events)
            
            if earnings_events:
                # Store earnings calendar
                self.store_earnings_calendar(earnings_events, db)
                db.commit()
                logger.info(f"Stored {len(earnings_events)} earnings events")
            else:
                logger.warning("No upcoming earnings events found")
            
            # Step 2: Identify target tickers
            target_tickers = list(set(event.ticker for event in earnings_events if event.ticker))
            stats['target_tickers_identified'] = len(target_tickers)
            
            if not target_tickers:
                if Config.COLLECT_GENERAL_NEWS_IF_NO_TARGETS:
                    target_tickers = Config.DEFAULT_FALLBACK_TICKERS_IF_NO_EARNINGS
                    logger.info(f"Using fallback tickers: {target_tickers}")
                else:
                    logger.info("No target tickers and general collection disabled")
                    return self._finalize_stats(stats)
            
            logger.info(f"Target tickers for collection: {', '.join(target_tickers[:10])}...")
            
            # Step 3: Collect targeted news
            if target_tickers:
                articles_map = self.collect_targeted_news(target_tickers)
                
                # Store articles
                articles_stored = 0
                for ticker_category, articles in articles_map.items():
                    stats['preliminary_articles_collected'] += len(articles)
                    
                    for article_data in articles:
                        if self._store_single_article(article_data, db):
                            articles_stored += 1
                            ticker = article_data.get('extracted_ticker')
                            if ticker and ticker not in ["UNKNOWN", "UNKNOWN_BUT_RELEVANT"]:
                                stats['tickers_with_new_news'].add(ticker)
                
                stats['articles_stored_in_db'] = articles_stored
                
                if articles_stored > 0:
                    db.commit()
                    logger.info(f"Committed {articles_stored} new articles to database")
                else:
                    logger.info("No new articles to store")
            
            logger.info("✅ Earnings-focused collection cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Error in collection cycle: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()
        
        return self._finalize_stats(stats)

    def _finalize_stats(self, stats: Dict) -> Dict:
        """Finalize collection statistics"""
        return {
            'earnings_events': stats['earnings_events_fetched'],
            'target_tickers': stats['target_tickers_identified'],
            'articles_collected': stats['preliminary_articles_collected'],
            'articles_stored': stats['articles_stored_in_db'],
            'tickers_with_news': len(stats['tickers_with_new_news'])
        }

    def get_collection_stats(self) -> Dict:
        """Get comprehensive collection statistics"""
        db = SessionLocal()
        
        try:
            total_articles = db.query(NewsArticle).count()
            
            today_start = datetime.combine(datetime.utcnow().date(), datetime.min.time())
            today_articles = db.query(NewsArticle).filter(NewsArticle.created_at >= today_start).count()
            
            processed_articles = db.query(NewsArticle).filter(NewsArticle.processed == True).count()
            unprocessed_articles = total_articles - processed_articles
            
            # Upcoming earnings
            seven_days_ahead = (datetime.utcnow() + timedelta(days=7)).date()
            upcoming_earnings = db.query(EarningsCalendar).filter(
                EarningsCalendar.earnings_date >= datetime.utcnow().date(),
                EarningsCalendar.earnings_date <= seven_days_ahead
            ).count()
            
            # Top tickers by news volume
            week_ago = datetime.utcnow() - timedelta(days=7)
            top_tickers = db.query(
                Company.ticker, 
                func.count(NewsArticle.id).label('article_count')
            ).join(
                NewsArticle, 
                NewsArticle.company_id == Company.id
            ).filter(
                NewsArticle.published_date >= week_ago
            ).group_by(
                Company.ticker
            ).order_by(
                func.count(NewsArticle.id).desc()
            ).limit(Config.NUM_TOP_TICKERS_FOR_STATS).all()
            
            top_tickers_list = [
                {'ticker': ticker, 'count': count} 
                for ticker, count in top_tickers
            ]
            
            return {
                'total_articles_in_db': total_articles,
                'articles_added_today_utc': today_articles,
                'fully_processed_articles': processed_articles,
                'unprocessed_articles_pending_sentiment': unprocessed_articles,
                'upcoming_earnings_in_db_next_7d': upcoming_earnings,
                'top_tickers_by_news_volume_last_7d': top_tickers_list,
                'last_stat_update_utc': datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {
                "error": f"Failed to get collection stats: {str(e)}",
                'total_articles_in_db': -1,
                'last_stat_update_utc': datetime.utcnow().isoformat() + "Z"
            }
        finally:
            db.close()

# Create the main NewsCollector class as an alias
NewsCollector = EnhancedNewsCollector
