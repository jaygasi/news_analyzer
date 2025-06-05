"""
Emergency data services for when primary LLMs are exhausted
Real implementations of Alpha Vantage, Polygon, and Tiingo APIs
Python 3.13.3 compatible
"""
import requests
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from utils.simple_logger import log_info, log_error, log_debug, log_warning
from config import Config


@dataclass
class DirectionalPrediction:
    """Result of directional analysis"""
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    source: str  # Which service provided the prediction
    raw_score: float = 0.0


class EmergencyDataServices:
    """Real implementations of emergency data services"""
    
    def __init__(self) -> None:
        """Initialize emergency data services"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FinancialNewsAnalyzer/1.0',
            'Accept': 'application/json'
        })
        
        # Rate limiting trackers
        self.last_request_times = {}
        self.request_counts = {}
    
    def _rate_limit(self, service: str, min_delay: float = 1.0) -> None:
        """Apply rate limiting for service"""
        last_time = self.last_request_times.get(service, 0)
        elapsed = time.time() - last_time
        
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        
        self.last_request_times[service] = time.time()
    
    def analyze_with_alpha_vantage(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Alpha Vantage News & Sentiment API"""
        if not Config.ALPHA_VANTAGE_API_KEY:
            return None
        
        try:
            self._rate_limit('alpha_vantage', 12.0)  # 5 calls per minute limit
            
            # Alpha Vantage News & Sentiment API
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': Config.ALPHA_VANTAGE_API_KEY,
                'limit': 20,
                'time_from': '20240101T0000'  # Recent news only
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'feed' not in data or not data['feed']:
                log_debug(f"No Alpha Vantage sentiment data for {ticker}")
                return None
            
            # Parse sentiment data
            total_sentiment = 0.0
            article_count = 0
            reasoning_parts = []
            
            for article in data['feed'][:10]:  # Analyze top 10 articles
                if 'ticker_sentiment' in article:
                    for sentiment_data in article['ticker_sentiment']:
                        if sentiment_data.get('ticker') == ticker:
                            sentiment_score = float(sentiment_data.get('ticker_sentiment_score', 0))
                            sentiment_label = sentiment_data.get('ticker_sentiment_label', 'Neutral')
                            
                            total_sentiment += sentiment_score
                            article_count += 1
                            reasoning_parts.append(f"{sentiment_label}({sentiment_score:.2f})")
            
            if article_count == 0:
                return None
            
            # Calculate average sentiment
            avg_sentiment = total_sentiment / article_count
            
            # Convert to direction and confidence
            if avg_sentiment > 0.15:
                direction = 'BUY'
                confidence = min(0.8, 0.5 + abs(avg_sentiment))
            elif avg_sentiment < -0.15:
                direction = 'SELL'
                confidence = min(0.8, 0.5 + abs(avg_sentiment))
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
            
            reasoning = f"Alpha Vantage sentiment: {avg_sentiment:.3f} from {article_count} articles. " + "; ".join(reasoning_parts[:3])
            
            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning[:200],
                source="alpha_vantage",
                raw_score=avg_sentiment
            )
            
        except Exception as e:
            log_error(f"Alpha Vantage analysis error for {ticker}: {e}")
            return None
    
    def analyze_with_polygon(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Polygon News API"""
        if not Config.POLYGON_API_KEY:
            return None
        
        try:
            self._rate_limit('polygon', 12.0)  # 5 calls per minute for free tier
            
            # Polygon News API
            url = f"https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': ticker,
                'published_utc.gte': '2024-01-01',
                'order': 'desc',
                'limit': 20,
                'apikey': Config.POLYGON_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or not data['results']:
                log_debug(f"No Polygon news data for {ticker}")
                return None
            
            # Analyze news content using enhanced keywords
            combined_content = ""
            article_count = 0
            
            for article in data['results'][:10]:
                title = article.get('title', '')
                description = article.get('description', '')
                combined_content += f"{title} {description} "
                article_count += 1
            
            # Use enhanced keyword analysis on Polygon content
            direction, confidence, reasoning = self._analyze_polygon_content(combined_content)
            
            reasoning = f"Polygon news analysis from {article_count} articles. {reasoning}"
            
            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning[:200],
                source="polygon",
                raw_score=confidence if direction == 'BUY' else -confidence if direction == 'SELL' else 0
            )
            
        except Exception as e:
            log_error(f"Polygon analysis error for {ticker}: {e}")
            return None
    
    def analyze_with_tiingo(self, ticker: str, text: str) -> Optional[DirectionalPrediction]:
        """Analyze using Tiingo News API"""
        if not Config.TIINGO_API_KEY:
            return None
        
        try:
            self._rate_limit('tiingo', 1.0)  # Conservative rate limiting
            
            # Tiingo News API
            url = f"https://api.tiingo.com/tiingo/news"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Token {Config.TIINGO_API_KEY}'
            }
            params = {
                'tickers': ticker,
                'startDate': '2024-01-01',
                'sortBy': 'publishedDate',
                'limit': 20
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                log_debug(f"No Tiingo news data for {ticker}")
                return None
            
            # Analyze news content
            combined_content = ""
            article_count = 0
            source_count = {}
            
            for article in data[:10]:
                title = article.get('title', '')
                description = article.get('description', '')
                source = article.get('source', 'Unknown')
                
                combined_content += f"{title} {description} "
                article_count += 1
                source_count[source] = source_count.get(source, 0) + 1
            
            # Use enhanced keyword analysis on Tiingo content
            direction, confidence, reasoning = self._analyze_tiingo_content(combined_content, source_count)
            
            top_sources = sorted(source_count.items(), key=lambda x: x[1], reverse=True)[:3]
            source_info = ", ".join([f"{src}({cnt})" for src, cnt in top_sources])
            
            reasoning = f"Tiingo analysis from {article_count} articles [{source_info}]. {reasoning}"
            
            return DirectionalPrediction(
                direction=direction,
                confidence=confidence,
                reasoning=reasoning[:200],
                source="tiingo",
                raw_score=confidence if direction == 'BUY' else -confidence if direction == 'SELL' else 0
            )
            
        except Exception as e:
            log_error(f"Tiingo analysis error for {ticker}: {e}")
            return None
    
    def _analyze_polygon_content(self, content: str) -> tuple[str, float, str]:
        """Analyze Polygon news content using enhanced keywords"""
        content_lower = content.lower()
        
        # Financial-specific keywords for Polygon (market data focused)
        strong_positive = ['earnings beat', 'revenue growth', 'guidance raised', 'analyst upgrade', 'price target raised']
        strong_negative = ['earnings miss', 'revenue decline', 'guidance lowered', 'analyst downgrade', 'price target cut']
        
        positive_score = 0.0
        negative_score = 0.0
        signals = []
        
        # Check strong signals first
        for keyword in strong_positive:
            if keyword in content_lower:
                positive_score += 2.0
                signals.append(f"+{keyword}")
        
        for keyword in strong_negative:
            if keyword in content_lower:
                negative_score += 2.0
                signals.append(f"-{keyword}")
        
        # Standard financial terms
        positive_terms = ['beat', 'growth', 'increase', 'strong', 'outperform', 'positive']
        negative_terms = ['miss', 'decline', 'weak', 'underperform', 'negative', 'concern']
        
        for term in positive_terms:
            if term in content_lower:
                positive_score += 0.5
        
        for term in negative_terms:
            if term in content_lower:
                negative_score += 0.5
        
        # Determine direction
        net_score = positive_score - negative_score
        total_signals = positive_score + negative_score
        
        if net_score > 1.0:
            direction = 'BUY'
            confidence = min(0.75, 0.5 + abs(net_score) / max(total_signals, 2))
        elif net_score < -1.0:
            direction = 'SELL'
            confidence = min(0.75, 0.5 + abs(net_score) / max(total_signals, 2))
        else:
            direction = 'NEUTRAL'
            confidence = 0.4
        
        reasoning = f"Score: +{positive_score:.1f}/-{negative_score:.1f}. Signals: {', '.join(signals[:3])}"
        
        return direction, confidence, reasoning
    
    def _analyze_tiingo_content(self, content: str, source_count: Dict[str, int]) -> tuple[str, float, str]:
        """Analyze Tiingo news content with source quality weighting"""
        content_lower = content.lower()
        
        # Source quality multipliers
        high_quality_sources = ['reuters', 'bloomberg', 'wall street journal', 'financial times']
        source_multiplier = 1.0
        
        for source in source_count:
            if any(quality_source in source.lower() for quality_source in high_quality_sources):
                source_multiplier = 1.3
                break
        
        # Tiingo-specific analysis (broader news coverage)
        positive_score = 0.0
        negative_score = 0.0
        signals = []
        
        # Company-specific events
        major_positive = ['acquisition announced', 'merger approved', 'partnership signed', 'contract won']
        major_negative = ['investigation launched', 'lawsuit filed', 'recall announced', 'bankruptcy']
        
        for keyword in major_positive:
            if keyword in content_lower:
                positive_score += 2.5 * source_multiplier
                signals.append(f"+{keyword}")
        
        for keyword in major_negative:
            if keyword in content_lower:
                negative_score += 2.5 * source_multiplier
                signals.append(f"-{keyword}")
        
        # General sentiment terms
        positive_terms = ['success', 'improvement', 'expansion', 'launch', 'approval', 'growth']
        negative_terms = ['failure', 'decline', 'closure', 'delay', 'rejection', 'loss']
        
        for term in positive_terms:
            if term in content_lower:
                positive_score += 0.7 * source_multiplier
        
        for term in negative_terms:
            if term in content_lower:
                negative_score += 0.7 * source_multiplier
        
        # Calculate final score
        net_score = positive_score - negative_score
        total_signals = positive_score + negative_score
        
        if net_score > 1.5:
            direction = 'BUY'
            confidence = min(0.8, 0.45 + abs(net_score) / max(total_signals, 3))
        elif net_score < -1.5:
            direction = 'SELL'
            confidence = min(0.8, 0.45 + abs(net_score) / max(total_signals, 3))
        else:
            direction = 'NEUTRAL'
            confidence = 0.35
        
        quality_note = f"(quality sources)" if source_multiplier > 1.0 else ""
        reasoning = f"Score: +{positive_score:.1f}/-{negative_score:.1f} {quality_note}. Key: {', '.join(signals[:2])}"
        
        return direction, confidence, reasoning