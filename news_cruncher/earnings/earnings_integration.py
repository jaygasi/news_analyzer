"""
Event-driven earnings analysis system
Integrates earnings events into normal news flow with separate scoring
"""
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import time
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_info, log_error, log_debug, log_warning


@dataclass
class EarningsEvent:
    """Represents an upcoming or recent earnings event"""
    ticker: str
    date: datetime
    quarter: str
    year: str
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    is_upcoming: bool = True
    days_until: int = 0
    
    @property
    def is_earnings_week(self) -> bool:
        """Check if earnings are within the next 7 days"""
        return 0 <= self.days_until <= 7
    
    @property
    def is_post_earnings(self) -> bool:
        """Check if earnings happened in the last 3 days"""
        return -3 <= self.days_until < 0


class EarningsEventDetector:
    """Detects upcoming and recent earnings events for tickers"""
    
    def __init__(self, fmp_loader: BaseFMPLoader):
        self.fmp_loader = fmp_loader
        self.earnings_cache = {}  # Cache earnings data to avoid repeated API calls
        self.cache_duration = timedelta(hours=6)  # Cache for 6 hours
        self.last_cache_time = None
    
    def get_earnings_events(self, tickers: List[str]) -> Dict[str, EarningsEvent]:
        """Get earnings events for the provided tickers"""
        # Refresh cache if needed
        self._refresh_earnings_cache_if_needed()
        
        events = {}
        for ticker in tickers:
            event = self._get_earnings_event_for_ticker(ticker)
            if event and (event.is_earnings_week or event.is_post_earnings):
                events[ticker] = event
        
        if events:
            log_info(f"📅 Found {len(events)} tickers with upcoming/recent earnings: {', '.join(events.keys())}")
        
        return events
    
    def _refresh_earnings_cache_if_needed(self):
        """Refresh earnings cache if it's stale"""
        now = datetime.now(timezone.utc)
        
        if (self.last_cache_time is None or 
            now - self.last_cache_time > self.cache_duration or
            not self.earnings_cache):
            
            log_info("🔄 Refreshing earnings calendar cache...")
            self._fetch_earnings_calendar()
            self.last_cache_time = now
    
    def _fetch_earnings_calendar(self):
        """Fetch earnings calendar for the next 14 days"""
        try:
            # Get earnings for next 14 days
            from_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            to_date = (datetime.now(timezone.utc) + timedelta(days=14)).strftime('%Y-%m-%d')
            
            log_debug(f"Fetching earnings calendar from {from_date} to {to_date}")
            
            earnings_data = self.fmp_loader.make_request("earning_calendar", {
                "from": from_date,
                "to": to_date
            })
            
            if not earnings_data or not isinstance(earnings_data, list):
                log_warning("No earnings calendar data returned")
                return
            
            log_info(f"📅 Loaded {len(earnings_data)} earnings events into cache")
            
            # Process and cache earnings events
            self.earnings_cache = {}
            for item in earnings_data:
                event = self._parse_earnings_item(item)
                if event:
                    self.earnings_cache[event.ticker] = event
            
            log_debug(f"Cached earnings for {len(self.earnings_cache)} unique tickers")
            
        except Exception as e:
            log_error(f"Error fetching earnings calendar: {e}")
    
    def _parse_earnings_item(self, item: Dict[str, Any]) -> Optional[EarningsEvent]:
        """Parse earnings calendar item into EarningsEvent"""
        try:
            ticker = str(item.get('symbol', '')).upper().strip()
            if not ticker or len(ticker) > 5:
                return None
            
            # Parse date
            date_str = item.get('date', '')
            if not date_str:
                return None
            
            try:
                earnings_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                try:
                    earnings_date = datetime.strptime(date_str, '%Y-%m-%d')
                    earnings_date = earnings_date.replace(tzinfo=timezone.utc)
                except:
                    return None
            
            # Calculate days until earnings
            now = datetime.now(timezone.utc)
            days_until = (earnings_date.date() - now.date()).days
            
            # Parse financial data
            eps_estimate = self._safe_float(item.get('epsEstimated'))
            eps_actual = self._safe_float(item.get('eps'))
            revenue_estimate = self._safe_float(item.get('revenueEstimated'))
            revenue_actual = self._safe_float(item.get('revenue'))
            
            return EarningsEvent(
                ticker=ticker,
                date=earnings_date,
                quarter=str(item.get('quarter', '')),
                year=str(item.get('fiscalDateEnding', '')[:4] if item.get('fiscalDateEnding') else ''),
                eps_estimate=eps_estimate,
                eps_actual=eps_actual,
                revenue_estimate=revenue_estimate,
                revenue_actual=revenue_actual,
                is_upcoming=days_until >= 0,
                days_until=days_until
            )
            
        except Exception as e:
            log_debug(f"Error parsing earnings item: {e}")
            return None
    
    def _get_earnings_event_for_ticker(self, ticker: str) -> Optional[EarningsEvent]:
        """Get earnings event for specific ticker from cache"""
        return self.earnings_cache.get(ticker.upper())
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None


class EarningsAnalyzer:
    """Analyzes earnings events and generates earnings-specific scores"""
    
    def __init__(self, fmp_loader: BaseFMPLoader):
        self.fmp_loader = fmp_loader
    
    def analyze_earnings_event(self, ticker: str, event: EarningsEvent, 
                             news_articles: List[Dict[str, Any]]) -> 'EarningsAnalysis':
        """Analyze earnings event and return earnings-specific analysis"""
        
        log_debug(f"Analyzing earnings event for {ticker}: {event.days_until} days")
        
        # Get earnings-specific news and data
        earnings_sentiment = self._analyze_earnings_sentiment(ticker, event, news_articles)
        estimate_analysis = self._analyze_estimates(ticker, event)
        guidance_sentiment = self._analyze_guidance_sentiment(ticker, event, news_articles)
        
        # Calculate overall earnings score
        earnings_score = self._calculate_earnings_score(
            earnings_sentiment, estimate_analysis, guidance_sentiment, event
        )
        
        return EarningsAnalysis(
            ticker=ticker,
            event=event,
            earnings_sentiment=earnings_sentiment,
            estimate_analysis=estimate_analysis,
            guidance_sentiment=guidance_sentiment,
            overall_score=earnings_score,
            confidence=self._calculate_confidence(earnings_sentiment, estimate_analysis, event)
        )
    
    def _analyze_earnings_sentiment(self, ticker: str, event: EarningsEvent, 
                                  articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from earnings-related articles"""
        earnings_articles = [
            a for a in articles 
            if any(keyword in a.get('title', '').lower() + a.get('text', '').lower() 
                   for keyword in ['earnings', 'eps', 'revenue', 'guidance', 'beat', 'miss', 'estimate'])
        ]
        
        if not earnings_articles:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'article_count': 0}
        
        # Simple keyword-based sentiment for earnings
        positive_keywords = [
            'beat', 'beats', 'exceeded', 'strong', 'growth', 'increased', 'higher',
            'positive', 'bullish', 'optimistic', 'upgrade', 'raised', 'guidance'
        ]
        negative_keywords = [
            'miss', 'missed', 'weak', 'declined', 'lower', 'cut', 'reduced',
            'negative', 'bearish', 'pessimistic', 'downgrade', 'lowered'
        ]
        
        positive_score = 0
        negative_score = 0
        
        for article in earnings_articles:
            content = (article.get('title', '') + ' ' + article.get('text', '')).lower()
            positive_score += sum(1 for word in positive_keywords if word in content)
            negative_score += sum(1 for word in negative_keywords if word in content)
        
        total_signals = positive_score + negative_score
        if total_signals == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif positive_score > negative_score:
            sentiment = 'positive'
            confidence = min(0.9, 0.6 + (positive_score / total_signals) * 0.3)
        else:
            sentiment = 'negative'
            confidence = min(0.9, 0.6 + (negative_score / total_signals) * 0.3)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'article_count': len(earnings_articles),
            'positive_signals': positive_score,
            'negative_signals': negative_score
        }
    
    def _analyze_estimates(self, ticker: str, event: EarningsEvent) -> Dict[str, Any]:
        """Analyze earnings estimates vs actuals"""
        analysis = {'has_data': False, 'beat_expectation': None, 'magnitude': 0.0}
        
        # If we have actual results
        if event.eps_actual is not None and event.eps_estimate is not None:
            beat_amount = event.eps_actual - event.eps_estimate
            if event.eps_estimate != 0:
                beat_percentage = (beat_amount / abs(event.eps_estimate)) * 100
            else:
                beat_percentage = 0
            
            analysis.update({
                'has_data': True,
                'beat_expectation': beat_amount > 0,
                'magnitude': abs(beat_percentage),
                'beat_amount': beat_amount,
                'beat_percentage': beat_percentage
            })
        
        return analysis
    
    def _analyze_guidance_sentiment(self, ticker: str, event: EarningsEvent, 
                                  articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze guidance-related sentiment"""
        guidance_articles = [
            a for a in articles 
            if any(keyword in a.get('title', '').lower() + a.get('text', '').lower() 
                   for keyword in ['guidance', 'outlook', 'forecast', 'expects', 'projects'])
        ]
        
        if not guidance_articles:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        # Similar sentiment analysis for guidance
        positive_count = 0
        negative_count = 0
        
        guidance_positive = ['raised', 'increased', 'improved', 'optimistic', 'strong outlook']
        guidance_negative = ['lowered', 'reduced', 'cut', 'pessimistic', 'weak outlook']
        
        for article in guidance_articles:
            content = (article.get('title', '') + ' ' + article.get('text', '')).lower()
            positive_count += sum(1 for phrase in guidance_positive if phrase in content)
            negative_count += sum(1 for phrase in guidance_negative if phrase in content)
        
        if positive_count > negative_count:
            return {'sentiment': 'positive', 'confidence': 0.7}
        elif negative_count > positive_count:
            return {'sentiment': 'negative', 'confidence': 0.7}
        else:
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _calculate_earnings_score(self, earnings_sentiment: Dict, estimate_analysis: Dict, 
                                guidance_sentiment: Dict, event: EarningsEvent) -> float:
        """Calculate overall earnings score (-1.0 to 1.0)"""
        score = 0.0
        
        # Sentiment component (40%)
        sentiment_score = {
            'positive': 0.8,
            'negative': -0.8,
            'neutral': 0.0
        }.get(earnings_sentiment['sentiment'], 0.0)
        
        sentiment_weighted = sentiment_score * earnings_sentiment['confidence'] * 0.4
        
        # Estimate beat/miss component (40%)
        estimate_score = 0.0
        if estimate_analysis['has_data']:
            if estimate_analysis['beat_expectation']:
                # Scale based on magnitude of beat
                magnitude_factor = min(1.0, estimate_analysis['magnitude'] / 20.0)  # 20% = max
                estimate_score = 0.6 + (magnitude_factor * 0.4)  # 0.6 to 1.0
            else:
                # Scale based on magnitude of miss
                magnitude_factor = min(1.0, estimate_analysis['magnitude'] / 20.0)
                estimate_score = -0.6 - (magnitude_factor * 0.4)  # -0.6 to -1.0
        
        estimate_weighted = estimate_score * 0.4
        
        # Guidance component (20%)
        guidance_score = {
            'positive': 0.6,
            'negative': -0.6,
            'neutral': 0.0
        }.get(guidance_sentiment['sentiment'], 0.0)
        
        guidance_weighted = guidance_score * guidance_sentiment['confidence'] * 0.2
        
        # Combine all components
        total_score = sentiment_weighted + estimate_weighted + guidance_weighted
        
        # Apply time decay for upcoming earnings (less certain)
        if event.is_upcoming and event.days_until > 0:
            time_decay = max(0.5, 1.0 - (event.days_until / 14.0))  # Decay over 2 weeks
            total_score *= time_decay
        
        return max(-1.0, min(1.0, total_score))
    
    def _calculate_confidence(self, earnings_sentiment: Dict, estimate_analysis: Dict, 
                            event: EarningsEvent) -> float:
        """Calculate confidence in earnings analysis"""
        base_confidence = 0.6
        
        # Boost confidence with more data
        if earnings_sentiment['article_count'] > 5:
            base_confidence += 0.1
        if estimate_analysis['has_data']:
            base_confidence += 0.2
        
        # Reduce confidence for far future earnings
        if event.is_upcoming and event.days_until > 7:
            base_confidence *= 0.8
        
        return min(0.95, base_confidence)


@dataclass
class EarningsAnalysis:
    """Results of earnings event analysis"""
    ticker: str
    event: EarningsEvent
    earnings_sentiment: Dict[str, Any]
    estimate_analysis: Dict[str, Any]
    guidance_sentiment: Dict[str, Any]
    overall_score: float  # -1.0 to 1.0
    confidence: float     # 0.0 to 1.0
    
    @property
    def direction(self) -> str:
        """Get directional signal from earnings score"""
        if self.overall_score > 0.2:
            return 'BUY'
        elif self.overall_score < -0.2:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def get_reasoning(self) -> str:
        """Get human-readable reasoning for the earnings analysis"""
        parts = []
        
        # Event timing
        if self.event.is_upcoming:
            parts.append(f"Earnings in {self.event.days_until} days")
        else:
            parts.append(f"Post-earnings ({abs(self.event.days_until)} days ago)")
        
        # Sentiment
        if self.earnings_sentiment['article_count'] > 0:
            parts.append(f"News sentiment: {self.earnings_sentiment['sentiment']}")
        
        # Estimates
        if self.estimate_analysis['has_data']:
            if self.estimate_analysis['beat_expectation']:
                parts.append(f"Beat estimates by {self.estimate_analysis['beat_percentage']:.1f}%")
            else:
                parts.append(f"Missed estimates by {abs(self.estimate_analysis['beat_percentage']):.1f}%")
        
        # Guidance
        if self.guidance_sentiment['sentiment'] != 'neutral':
            parts.append(f"Guidance: {self.guidance_sentiment['sentiment']}")
        
        return " | ".join(parts)


class EarningsIntegrationManager:
    """Manages integration of earnings analysis into the main decision flow"""
    
    def __init__(self, fmp_loader: BaseFMPLoader):
        self.detector = EarningsEventDetector(fmp_loader)
        self.analyzer = EarningsAnalyzer(fmp_loader)
    
    def analyze_tickers_for_earnings(self, ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, EarningsAnalysis]:
        """Analyze tickers for earnings events and return earnings analyses"""
        tickers = list(ticker_buckets.keys())
        
        # Detect earnings events
        earnings_events = self.detector.get_earnings_events(tickers)
        
        if not earnings_events:
            return {}
        
        # Analyze each earnings event
        earnings_analyses = {}
        for ticker, event in earnings_events.items():
            articles = ticker_buckets.get(ticker, [])
            analysis = self.analyzer.analyze_earnings_event(ticker, event, articles)
            earnings_analyses[ticker] = analysis
            
            log_debug(f"📊 {ticker} earnings analysis: {analysis.direction} "
                     f"(score: {analysis.overall_score:.3f}, conf: {analysis.confidence:.3f})")
        
        return earnings_analyses