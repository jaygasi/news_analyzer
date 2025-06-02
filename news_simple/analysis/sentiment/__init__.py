"""
Sentiment analysis module - modular sentiment components
"""
from .enhanced_sentiment_scorer import EnhancedSentimentScorer, SentimentScore
from .keyword_analyzer import KeywordAnalyzer
from .credibility_scorer import CredibilityScorer
from .multi_source_validator import MultiSourceValidator
from .pattern_matcher import HistoricalPatternMatcher

__all__ = [
    'EnhancedSentimentScorer',
    'SentimentScore',
    'KeywordAnalyzer',
    'CredibilityScorer',
    'MultiSourceValidator',
    'HistoricalPatternMatcher'
]