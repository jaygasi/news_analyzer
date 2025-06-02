"""
Enhanced sentiment scoring combining multiple analysis methods
"""
from typing import Optional, List
from dataclasses import dataclass
from .keyword_analyzer import KeywordAnalyzer
from .credibility_scorer import CredibilityScorer
from .multi_source_validator import MultiSourceValidator
from .pattern_matcher import HistoricalPatternMatcher


@dataclass
class SentimentScore:
    """Enhanced sentiment with quality metrics"""
    base_sentiment: float
    magnitude_score: float
    urgency_score: float
    credibility_score: float
    sector_relevance: float
    surprise_factor: float
    confirmation_score: float
    final_sentiment: float
    quality_confidence: float


class EnhancedSentimentScorer:
    """Enhanced sentiment scorer combining multiple analysis methods"""
    
    def __init__(self):
        self.keyword_analyzer = KeywordAnalyzer()
        self.credibility_scorer = CredibilityScorer()
        self.source_validator = MultiSourceValidator()
        self.pattern_matcher = HistoricalPatternMatcher()
        self._initialize_sector_keywords()
    
    def _initialize_sector_keywords(self) -> None:
        """Initialize sector-specific keywords"""
        self.sector_keywords = {
            'biotech': {
                'high_impact': frozenset(['fda', 'approval', 'trial', 'drug', 'therapy', 'treatment']),
                'catalysts': frozenset(['pdufa', 'breakthrough', 'orphan', 'fast track', 'priority']),
                'risks': frozenset(['safety', 'adverse', 'crl', 'rejection', 'delay'])
            },
            'tech': {
                'high_impact': frozenset(['ai', 'cloud', 'subscription', 'platform', 'user growth']),
                'catalysts': frozenset(['breakthrough', 'patent', 'acquisition', 'partnership']),
                'risks': frozenset(['competition', 'regulation', 'data breach', 'antitrust'])
            },
            'energy': {
                'high_impact': frozenset(['oil', 'gas', 'renewable', 'production', 'reserves']),
                'catalysts': frozenset(['discovery', 'drilling', 'capacity', 'contract']),
                'risks': frozenset(['spill', 'accident', 'regulation', 'embargo'])
            }
        }
    
    def calculate_enhanced_sentiment(self, title: str, content: str, symbol: str,
                                   current_price: float, topic: str,
                                   source: str = "", recent_analyses: Optional[List] = None) -> SentimentScore:
        """Calculate enhanced sentiment with comprehensive analysis"""
        
        text = f"{title} {content}".lower()
        
        # Component calculations
        base_sentiment = self.keyword_analyzer.calculate_base_sentiment(text)
        magnitude_score = self.keyword_analyzer.calculate_magnitude_score(text, topic)
        urgency_score = self.keyword_analyzer.calculate_urgency_score(text, title)
        credibility_score = self.credibility_scorer.calculate_credibility_score(text, source, content)
        sector_relevance = self._calculate_sector_relevance(text, topic)
        surprise_factor = self._calculate_surprise_factor(text)
        
        # Multi-source confirmation
        confirmation_score = self._get_confirmation_score(
            symbol, base_sentiment, topic, recent_analyses
        )
        
        # Historical pattern matching
        impact_multiplier, historical_accuracy = self.pattern_matcher.find_historical_impact(
            topic, base_sentiment
        )
        
        # Calculate weighted sentiment
        final_sentiment = self._calculate_weighted_sentiment(
            base_sentiment, magnitude_score, credibility_score, 
            sector_relevance, confirmation_score, urgency_score, impact_multiplier
        )
        
        # Calculate quality confidence
        quality_confidence = self._calculate_quality_confidence(
            credibility_score, confirmation_score, sector_relevance, 
            magnitude_score, historical_accuracy
        )
        
        return SentimentScore(
            base_sentiment=base_sentiment,
            magnitude_score=magnitude_score,
            urgency_score=urgency_score,
            credibility_score=credibility_score,
            sector_relevance=sector_relevance,
            surprise_factor=surprise_factor,
            confirmation_score=confirmation_score,
            final_sentiment=final_sentiment,
            quality_confidence=quality_confidence
        )
    
    def _calculate_sector_relevance(self, text: str, topic: str) -> float:
        """Calculate sector relevance score"""
        relevance = 0.5
        
        if topic in self.sector_keywords:
            sector_data = self.sector_keywords[topic]
            text_words = set(text.split())
            
            high_impact_count = len(text_words & sector_data['high_impact'])
            catalyst_count = len(text_words & sector_data['catalysts'])
            
            if high_impact_count > 0:
                relevance = max(relevance, 0.75)
            if catalyst_count > 0:
                relevance = max(relevance, 0.85)
        
        return relevance
    
    def _calculate_surprise_factor(self, text: str) -> float:
        """Calculate surprise factor from unexpected keywords"""
        surprise_indicators = frozenset([
            'unexpected', 'surprise', 'shocking', 'unprecedented', 'unusual',
            'rare', 'first time', 'never before', 'breaking news'
        ])
        
        text_words = set(text.split())
        if text_words & surprise_indicators:
            return 0.75
        
        return 0.5
    
    def _get_confirmation_score(self, symbol: str, base_sentiment: float, 
                              topic: str, recent_analyses: Optional[List]) -> float:
        """Get multi-source confirmation score"""
        if recent_analyses and len(recent_analyses) > 1:
            analysis_dict = {
                'symbol': symbol,
                'sentiment_score': base_sentiment,
                'timestamp': None,  # Will be set by caller
                'topic': topic
            }
            return self.source_validator.check_multi_source_confirmation(
                analysis_dict, recent_analyses[-10:]
            )
        return 0.5
    
    def _calculate_weighted_sentiment(self, base_sentiment: float, magnitude_score: float,
                                    credibility_score: float, sector_relevance: float,
                                    confirmation_score: float, urgency_score: float,
                                    impact_multiplier: float) -> float:
        """Calculate weighted final sentiment"""
        weights = (0.3, 0.18, 0.18, 0.14, 0.12, 0.08)
        
        weighted_sentiment = (
            base_sentiment * weights[0] +
            base_sentiment * magnitude_score * weights[1] +
            base_sentiment * credibility_score * weights[2] +
            base_sentiment * sector_relevance * weights[3] +
            base_sentiment * confirmation_score * weights[4] +
            base_sentiment * urgency_score * weights[5]
        ) * impact_multiplier
        
        return max(-1.0, min(1.0, weighted_sentiment))
    
    def _calculate_quality_confidence(self, credibility_score: float, confirmation_score: float,
                                    sector_relevance: float, magnitude_score: float,
                                    historical_accuracy: float) -> float:
        """Calculate overall quality confidence"""
        weights = (0.28, 0.22, 0.18, 0.14, 0.18)
        
        return (
            credibility_score * weights[0] +
            confirmation_score * weights[1] +
            sector_relevance * weights[2] +
            magnitude_score * weights[3] +
            historical_accuracy * weights[4]
        )