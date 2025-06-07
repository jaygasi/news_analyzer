"""
Earnings Integration Manager - Coordinates earnings events and transcript analysis
Python 3.13.3 compatible - FIXED VERSION
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from data_loaders.earnings_transcript_fetcher import EarningsTranscriptFetcher
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config


@dataclass
class EarningsEvent:
    """Represents an earnings event for a company"""
    ticker: str
    date: str
    quarter: str
    year: str
    estimated_eps: Optional[float] = None
    estimated_revenue: Optional[float] = None
    time: Optional[str] = None
    event_type: str = 'scheduled'  # 'scheduled', 'recent', 'upcoming'


@dataclass
class EarningsAnalysisResult:
    """Results from earnings analysis for a ticker"""
    ticker: str
    decision: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float
    reasoning: str
    earnings_events: List[EarningsEvent]
    has_transcript_data: bool = False
    transcript_articles: List[Dict[str, Any]] = None


class EarningsIntegrationManager:
    """Manages integration between earnings events and the main analysis pipeline"""
    
    def __init__(self, news_fetcher):
        """Initialize earnings integration manager"""
        self.news_fetcher = news_fetcher
        self.earnings_fetcher = EarningsTranscriptFetcher(Config.FMP_API_KEY)
        self.earnings_cache = {}  # Cache for earnings calendar data
        self.cache_expiry = None
        
        log_info("✅ Earnings integration manager initialized")
    
    def analyze_earnings_for_tickers(self, tickers: List[str]) -> Dict[str, EarningsAnalysisResult]:
        """Analyze earnings events for a list of tickers"""
        if not Config.ENABLE_EARNINGS_EVENTS:
            log_debug("Earnings events disabled, skipping analysis")
            return {}
        
        log_info(f"📅 Analyzing earnings events for {len(tickers)} tickers...")
        
        # Refresh earnings calendar cache if needed
        self._refresh_earnings_cache()
        
        # Find tickers with earnings events
        earnings_results = {}
        tickers_with_events = 0
        
        for ticker in tickers:
            earnings_events = self._find_earnings_events_for_ticker(ticker)
            
            if earnings_events:
                tickers_with_events += 1
                # Analyze the earnings event
                analysis_result = self._analyze_ticker_earnings(ticker, earnings_events)
                earnings_results[ticker] = analysis_result
                
                log_info(f"   📊 {ticker}: {analysis_result.decision} (score: {analysis_result.confidence:+.2f}, conf: {abs(analysis_result.confidence):.2f}) - {self._get_event_description(earnings_events[0])}")
        
        log_info(f"📅 Found {tickers_with_events} tickers with earnings events")
        
        for ticker, result in earnings_results.items():
            log_info(f"   📊 {ticker}: {result.decision} (score: {result.confidence:+.2f}, conf: {abs(result.confidence):.2f}) - {self._get_event_description(result.earnings_events[0])}")
        
        return earnings_results
    
    def _refresh_earnings_cache(self) -> None:
        """Refresh earnings calendar cache if expired"""
        now = datetime.now(timezone.utc)
        
        # Check if cache needs refresh
        if (self.cache_expiry is None or 
            now > self.cache_expiry or 
            not self.earnings_cache):
            
            log_info("🔄 Refreshing earnings calendar cache...")
            
            # Calculate date range for earnings lookup
            start_date = now - timedelta(days=Config.EARNINGS_LOOKBACK_DAYS)
            end_date = now + timedelta(days=Config.EARNINGS_LOOKAHEAD_DAYS)
            
            # Fetch earnings calendar
            try:
                earnings_data = self.news_fetcher.make_request("earning_calendar", {
                    "from": start_date.strftime('%Y-%m-%d'),
                    "to": end_date.strftime('%Y-%m-%d')
                })
                
                if earnings_data and isinstance(earnings_data, list):
                    # Process and cache earnings events
                    self.earnings_cache = {}
                    
                    for event_data in earnings_data:
                        ticker = event_data.get('symbol', '').upper()
                        if not ticker:
                            continue
                        
                        event = EarningsEvent(
                            ticker=ticker,
                            date=event_data.get('date', ''),
                            quarter=str(event_data.get('quarter', '')),
                            year=str(event_data.get('year', '')),
                            estimated_eps=event_data.get('epsEstimated'),
                            estimated_revenue=event_data.get('revenueEstimated'),
                            time=event_data.get('time', ''),
                            event_type=self._determine_event_type(event_data.get('date', ''))
                        )
                        
                        if ticker not in self.earnings_cache:
                            self.earnings_cache[ticker] = []
                        self.earnings_cache[ticker].append(event)
                    
                    # Set cache expiry
                    self.cache_expiry = now + timedelta(hours=Config.EARNINGS_CACHE_HOURS)
                    
                    log_info(f"📅 Loaded {len(earnings_data)} earnings events into cache")
                    
                else:
                    log_warning("No earnings calendar data returned")
                    self.earnings_cache = {}
                    self.cache_expiry = now + timedelta(hours=1)  # Retry in 1 hour
                    
            except Exception as e:
                log_error(f"Error fetching earnings calendar: {e}")
                self.earnings_cache = {}
                self.cache_expiry = now + timedelta(hours=1)  # Retry in 1 hour
    
    def _find_earnings_events_for_ticker(self, ticker: str) -> List[EarningsEvent]:
        """Find earnings events for a specific ticker"""
        return self.earnings_cache.get(ticker.upper(), [])
    
    def _determine_event_type(self, event_date: str) -> str:
        """Determine if earnings event is recent, upcoming, or current"""
        try:
            if not event_date:
                return 'unknown'
            
            event_dt = datetime.strptime(event_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            
            days_diff = (event_dt - now).days
            
            if days_diff < -Config.EARNINGS_LOOKBACK_DAYS:
                return 'past'
            elif days_diff <= 0:
                return 'recent'
            elif days_diff <= Config.EARNINGS_LOOKAHEAD_DAYS:
                return 'upcoming'
            else:
                return 'future'
                
        except Exception:
            return 'unknown'
    
    def _analyze_ticker_earnings(self, ticker: str, earnings_events: List[EarningsEvent]) -> EarningsAnalysisResult:
        """Analyze earnings events for a ticker and generate trading recommendation"""
        
        # Get the most relevant earnings event (closest to current date)
        primary_event = self._get_primary_earnings_event(earnings_events)
        
        # Try to fetch transcript data if available
        transcript_articles = []
        has_transcript_data = False
        
        if primary_event and primary_event.quarter and primary_event.year:
            try:
                # Attempt to fetch earnings transcript
                transcript_analysis = self.earnings_fetcher.fetch_earnings_transcript(
                    ticker=ticker,
                    year=int(primary_event.year),
                    quarter=int(primary_event.quarter)
                )
                
                if transcript_analysis:
                    has_transcript_data = True
                    transcript_articles = self.earnings_fetcher.create_earnings_articles(transcript_analysis)
                    log_debug(f"📄 Found transcript data for {ticker} Q{primary_event.quarter} {primary_event.year}")
                    
            except Exception as e:
                log_debug(f"No transcript data available for {ticker}: {e}")
        
        # Perform earnings-based analysis
        decision, confidence, reasoning = self._generate_earnings_recommendation(
            ticker, primary_event, has_transcript_data, transcript_articles
        )
        
        return EarningsAnalysisResult(
            ticker=ticker,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            earnings_events=earnings_events,
            has_transcript_data=has_transcript_data,
            transcript_articles=transcript_articles
        )
    
    def _get_primary_earnings_event(self, earnings_events: List[EarningsEvent]) -> Optional[EarningsEvent]:
        """Get the most relevant earnings event from a list"""
        if not earnings_events:
            return None
        
        # Sort by date and priority
        now = datetime.now(timezone.utc)
        
        def event_priority(event):
            try:
                event_date = datetime.strptime(event.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                days_diff = abs((event_date - now).days)
                
                # Prioritize recent events, then upcoming events
                if event.event_type == 'recent':
                    return (0, days_diff)  # Highest priority
                elif event.event_type == 'upcoming':
                    return (1, days_diff)  # Second priority
                else:
                    return (2, days_diff)  # Lower priority
                    
            except Exception:
                return (999, 999)  # Lowest priority for invalid dates
        
        sorted_events = sorted(earnings_events, key=event_priority)
        return sorted_events[0]
    
    def _generate_earnings_recommendation(self, ticker: str, earnings_event: EarningsEvent, 
                                        has_transcript: bool, transcript_articles: List[Dict[str, Any]]) -> tuple[str, float, str]:
        """Generate trading recommendation based on earnings analysis"""
        
        # Base confidence and decision
        decision = 'NEUTRAL'
        confidence = 0.6  # Base confidence for earnings events
        reasoning_parts = []
        
        # Analyze earnings event timing
        if earnings_event.event_type == 'recent':
            reasoning_parts.append(f"Recent earnings event ({earnings_event.date})")
            confidence += 0.1  # Boost for recent events
        elif earnings_event.event_type == 'upcoming':
            reasoning_parts.append(f"Upcoming earnings in {self._calculate_days_to_event(earnings_event.date)} days")
            confidence += 0.05  # Small boost for upcoming events
        
        # Analyze transcript data if available
        if has_transcript and transcript_articles:
            reasoning_parts.append("Enhanced with earnings transcript analysis")
            confidence += 0.15  # Significant boost for transcript data
            
            # Simple sentiment analysis of transcript
            transcript_sentiment = self._analyze_transcript_sentiment(transcript_articles)
            if transcript_sentiment > 0.6:
                decision = 'BUY'
                confidence += transcript_sentiment * 0.2
                reasoning_parts.append("Positive management tone and guidance")
            elif transcript_sentiment < 0.4:
                decision = 'SELL'
                confidence += (1 - transcript_sentiment) * 0.2
                reasoning_parts.append("Cautious management tone or concerns")
        
        # Earnings estimates analysis
        if earnings_event.estimated_eps:
            reasoning_parts.append(f"EPS estimate: {earnings_event.estimated_eps}")
        
        # Ensure confidence is within bounds
        confidence = min(confidence, 0.95)
        confidence = max(confidence, Config.MIN_EARNINGS_CONFIDENCE)
        
        # Build reasoning string
        reasoning = f"Earnings analysis for {ticker}: " + "; ".join(reasoning_parts)
        
        return decision, confidence, reasoning
    
    def _analyze_transcript_sentiment(self, transcript_articles: List[Dict[str, Any]]) -> float:
        """Simple sentiment analysis of earnings transcript articles"""
        if not transcript_articles:
            return 0.5  # Neutral
        
        positive_keywords = [
            'strong', 'growth', 'increase', 'beat', 'exceeded', 'positive', 'optimistic',
            'confident', 'solid', 'robust', 'momentum', 'expanding', 'acceleration'
        ]
        
        negative_keywords = [
            'weak', 'decline', 'decrease', 'miss', 'disappointed', 'negative', 'concern',
            'cautious', 'challenging', 'pressure', 'headwinds', 'uncertainty', 'risk'
        ]
        
        total_score = 0
        total_articles = 0
        
        for article in transcript_articles:
            text = (article.get('text', '') + ' ' + article.get('title', '')).lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_count + negative_count > 0:
                article_sentiment = positive_count / (positive_count + negative_count)
                total_score += article_sentiment
                total_articles += 1
        
        if total_articles > 0:
            average_sentiment = total_score / total_articles
            return average_sentiment
        else:
            return 0.5  # Neutral if no sentiment indicators found
    
    def _calculate_days_to_event(self, event_date: str) -> int:
        """Calculate days until earnings event"""
        try:
            event_dt = datetime.strptime(event_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (event_dt - now).days
        except Exception:
            return 0
    
    def _get_event_description(self, event: EarningsEvent) -> str:
        """Get human-readable description of earnings event"""
        if event.event_type == 'recent':
            days_ago = abs(self._calculate_days_to_event(event.date))
            return f"Earnings {days_ago} days ago"
        elif event.event_type == 'upcoming':
            days_ahead = self._calculate_days_to_event(event.date)
            return f"Earnings in {days_ahead} days"
        else:
            return f"Earnings on {event.date}"
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status for debugging"""
        now = datetime.now(timezone.utc)
        
        return {
            'cache_size': len(self.earnings_cache),
            'cache_expiry': self.cache_expiry.isoformat() if self.cache_expiry else None,
            'cache_valid': self.cache_expiry and now < self.cache_expiry,
            'tickers_with_events': len(self.earnings_cache),
            'total_events': sum(len(events) for events in self.earnings_cache.values())
        }
