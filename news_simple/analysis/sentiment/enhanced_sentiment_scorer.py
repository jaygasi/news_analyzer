"""
Optimized enhanced sentiment scoring with improved performance and accuracy
"""
from typing import Optional, List
from dataclasses import dataclass
from .keyword_analyzer import KeywordAnalyzer
from .credibility_scorer import CredibilityScorer
from .multi_source_validator import MultiSourceValidator
from .pattern_matcher import HistoricalPatternMatcher


@dataclass
class SentimentScore:
    """Comprehensive sentiment analysis result with quality metrics."""
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
    """Optimized sentiment scorer with improved component integration."""
    
    # Class-level constants for performance
    SECTOR_KEYWORDS = {
        'biotech': {
            'high_impact': frozenset(['fda', 'approval', 'trial', 'drug', 'therapy', 'treatment', 'clinical']),
            'catalysts': frozenset(['pdufa', 'breakthrough', 'orphan', 'fast track', 'priority review']),
            'risks': frozenset(['safety', 'adverse', 'crl', 'rejection', 'delay', 'halt'])
        },
        'tech': {
            'high_impact': frozenset(['ai', 'cloud', 'subscription', 'platform', 'user growth', 'saas']),
            'catalysts': frozenset(['breakthrough', 'patent', 'acquisition', 'partnership', 'ipo']),
            'risks': frozenset(['competition', 'regulation', 'data breach', 'antitrust', 'privacy'])
        },
        'energy': {
            'high_impact': frozenset(['oil', 'gas', 'renewable', 'production', 'reserves', 'pipeline']),
            'catalysts': frozenset(['discovery', 'drilling', 'capacity', 'contract', 'permit']),
            'risks': frozenset(['spill', 'accident', 'regulation', 'embargo', 'environmental'])
        },
        'financial': {
            'high_impact': frozenset(['earnings', 'loan', 'credit', 'interest', 'deposit', 'capital']),
            'catalysts': frozenset(['merger', 'acquisition', 'dividend', 'buyback', 'ipo']),
            'risks': frozenset(['default', 'fraud', 'regulation', 'stress test', 'fine'])
        }
    }
    
    # Optimized weight configuration
    WEIGHT_CONFIG = {
        'base_sentiment': 0.30,
        'magnitude': 0.18,
        'credibility': 0.18,
        'sector_relevance': 0.14,
        'confirmation': 0.12,
        'urgency': 0.08
    }
    
    def __init__(self):
        """Initialize with optimized component instances."""
        self.keyword_analyzer = KeywordAnalyzer()
        self.credibility_scorer = CredibilityScorer()
        self.source_validator = MultiSourceValidator()
        self.pattern_matcher = HistoricalPatternMatcher()
    
    def calculate_enhanced_sentiment(self, title: str, content: str, symbol: str,
                                   current_price: float, topic: str,
                                   source: str = "", recent_analyses: Optional[List] = None) -> SentimentScore:
        """Calculate comprehensive sentiment with optimized processing."""
        
        # Pre-process text once
        combined_text = f"{title} {content}".lower()
        text_words = set(combined_text.split())
        
        # Calculate all components efficiently
        base_sentiment = self.keyword_analyzer.calculate_base_sentiment(combined_text)
        magnitude_score = self.keyword_analyzer.calculate_magnitude_score(combined_text, topic)
        urgency_score = self.keyword_analyzer.calculate_urgency_score(combined_text, title)
        credibility_score = self.credibility_scorer.calculate_credibility_score(combined_text, source, content)
        sector_relevance = self._calculate_optimized_sector_relevance(text_words, topic)
        surprise_factor = self._calculate_optimized_surprise_factor(text_words)
        
        # Multi-source confirmation with caching
        confirmation_score = self._get_optimized_confirmation_score(
            symbol, base_sentiment, topic, recent_analyses
        )
        
        # Historical pattern matching
        impact_multiplier, historical_accuracy = self.pattern_matcher.find_historical_impact(
            topic, base_sentiment
        )
        
        # Optimized weighted sentiment calculation
        final_sentiment = self._calculate_optimized_weighted_sentiment(
            base_sentiment, magnitude_score, credibility_score, 
            sector_relevance, confirmation_score, urgency_score, impact_multiplier
        )
        
        # Enhanced quality confidence calculation
        quality_confidence = self._calculate_optimized_quality_confidence(
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
    
    def _calculate_optimized_sector_relevance(self, text_words: set, topic: str) -> float:
        """Optimized sector relevance calculation using pre-computed sets."""
        if topic not in self.SECTOR_KEYWORDS:
            return 0.5
        
        sector_data = self.SECTOR_KEYWORDS[topic]
        
        # Fast set intersection operations
        high_impact_matches = len(text_words & sector_data['high_impact'])
        catalyst_matches = len(text_words & sector_data['catalysts'])
        risk_matches = len(text_words & sector_data['risks'])
        
        # Calculate relevance score
        relevance = 0.5
        
        if high_impact_matches > 0:
            relevance = max(relevance, 0.75 + (high_impact_matches - 1) * 0.05)
        
        if catalyst_matches > 0:
            relevance = max(relevance, 0.85 + (catalyst_matches - 1) * 0.05)
        
        # Risk keywords reduce relevance for positive sentiment
        if risk_matches > 0:
            relevance *= max(0.7, 1.0 - (risk_matches * 0.1))
        
        return min(relevance, 1.0)
    
    def _calculate_optimized_surprise_factor(self, text_words: set) -> float:
        """Optimized surprise factor calculation."""
        surprise_indicators = frozenset([
            'unexpected', 'surprise', 'shocking', 'unprecedented', 'unusual',
            'rare', 'first', 'never', 'breaking', 'sudden', 'abrupt'
        ])
        
        matches = len(text_words & surprise_indicators)
        
        if matches == 0:
            return 0.5
        elif matches == 1:
            return 0.7
        else:
            return min(0.75 + (matches - 1) * 0.05, 0.9)
    
    def _get_optimized_confirmation_score(self, symbol: str, base_sentiment: float, 
                                        topic: str, recent_analyses: Optional[List]) -> float:
        """Optimized confirmation score with efficient caching."""
        if not recent_analyses or len(recent_analyses) < 2:
            return 0.5
        
        # Create efficient analysis structure
        analysis_dict = {
            'symbol': symbol,
            'sentiment_score': base_sentiment,
            'timestamp': None,  # Will be set by validator
            'topic': topic
        }
        
        # Use only recent relevant analyses for performance
        relevant_analyses = recent_analyses[-10:] if len(recent_analyses) > 10 else recent_analyses
        
        return self.source_validator.check_multi_source_confirmation(
            analysis_dict, relevant_analyses
        )
    
    def _calculate_optimized_weighted_sentiment(self, base_sentiment: float, magnitude_score: float,
                                              credibility_score: float, sector_relevance: float,
                                              confirmation_score: float, urgency_score: float,
                                              impact_multiplier: float) -> float:
        """Optimized weighted sentiment calculation using pre-computed weights."""
        weights = self.WEIGHT_CONFIG
        
        # Vectorized calculation for performance
        components = [
            base_sentiment * weights['base_sentiment'],
            base_sentiment * magnitude_score * weights['magnitude'],
            base_sentiment * credibility_score * weights['credibility'],
            base_sentiment * sector_relevance * weights['sector_relevance'],
            base_sentiment * confirmation_score * weights['confirmation'],
            base_sentiment * urgency_score * weights['urgency']
        ]
        
        weighted_sentiment = sum(components) * impact_multiplier
        
        return max(-1.0, min(1.0, weighted_sentiment))
    
    def _calculate_optimized_quality_confidence(self, credibility_score: float, confirmation_score: float,
                                              sector_relevance: float, magnitude_score: float,
                                              historical_accuracy: float) -> float:
        """Optimized quality confidence calculation."""
        # Pre-computed weights for performance
        confidence_weights = [0.28, 0.22, 0.18, 0.14, 0.18]
        
        components = [
            credibility_score,
            confirmation_score,
            sector_relevance,
            magnitude_score,
            historical_accuracy
        ]
        
        # Vectorized dot product
        return sum(comp * weight for comp, weight in zip(components, confidence_weights))