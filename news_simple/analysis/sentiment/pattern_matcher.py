"""
Historical pattern matching for news impact prediction
"""
from typing import Dict, Tuple


class HistoricalPatternMatcher:
    """Matches current news patterns to historical impact data"""
    
    def __init__(self):
        self._initialize_impact_multipliers()
        self._initialize_accuracy_rates()
    
    def _initialize_impact_multipliers(self) -> None:
        """Initialize impact multipliers for different news types"""
        self.impact_multipliers = {
            'earnings': {
                'strong_beat': 1.25, 'beat': 1.08, 'miss': 0.75, 'strong_miss': 0.55
            },
            'biotech': {
                'fda_approval': 1.8, 'trial_success': 1.4, 'trial_failure': 0.35, 'fda_rejection': 0.25
            },
            'analyst': {
                'upgrade': 1.15, 'downgrade': 0.85, 'initiate_buy': 1.06
            }
        }
    
    def _initialize_accuracy_rates(self) -> None:
        """Initialize historical accuracy rates by news type"""
        self.accuracy_rates = {
            'earnings': 0.72,
            'biotech': 0.82,
            'analyst': 0.62,
            'general': 0.52
        }
    
    def find_historical_impact(self, news_type: str, sentiment_score: float) -> Tuple[float, float]:
        """Find historical impact multiplier and accuracy for news type"""
        category = self._categorize_news(news_type, sentiment_score)
        multiplier = self.impact_multipliers.get(news_type, {}).get(category, 1.0)
        accuracy = self.accuracy_rates.get(news_type, 0.52)
        
        return multiplier, accuracy
    
    def _categorize_news(self, news_type: str, sentiment: float) -> str:
        """Categorize news based on type and sentiment"""
        if news_type == 'earnings':
            if sentiment > 0.55:
                return 'strong_beat'
            elif sentiment > 0.15:
                return 'beat'
            elif sentiment < -0.55:
                return 'strong_miss'
            else:
                return 'miss'
        elif news_type == 'biotech':
            if sentiment > 0.4:
                return 'fda_approval' if sentiment > 0.7 else 'trial_success'
            elif sentiment < -0.4:
                return 'fda_rejection' if sentiment < -0.7 else 'trial_failure'
        elif news_type == 'analyst':
            if sentiment > 0.3:
                return 'upgrade'
            elif sentiment < -0.3:
                return 'downgrade'
            else:
                return 'initiate_buy'
        
        return 'general'
    
    def get_sector_multiplier(self, topic: str) -> float:
        """Get sector-specific multiplier"""
        sector_multipliers = {
            'biotech': 1.2,
            'earnings': 1.1,
            'analyst': 0.95,
            'ma': 1.15,
            'general': 1.0
        }
        
        return sector_multipliers.get(topic, 1.0)