"""
Multi-source validation for news confirmation
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time


class MultiSourceValidator:
    """Validates news across multiple sources for confirmation"""
    
    def __init__(self):
        self.news_cache: Dict[str, float] = {}
        self.confirmation_window_hours = 4
        self.max_cache_size = 200
        self._last_cleanup = time.time()
        self._cleanup_interval = 1800  # 30 minutes
    
    def check_multi_source_confirmation(self, current_analysis: dict, 
                                      recent_analyses: Optional[List[dict]]) -> float:
        """Check for confirmation across multiple sources"""
        if not recent_analyses or len(recent_analyses) < 2:
            return 0.5
        
        current_symbol = current_analysis['symbol']
        current_sentiment = current_analysis['sentiment_score']
        current_timestamp = current_analysis['timestamp']
        
        # Filter related articles
        cutoff_time = current_timestamp - timedelta(hours=self.confirmation_window_hours)
        related_articles = [
            art for art in recent_analyses[-20:]  # Only check last 20 for efficiency
            if (art['symbol'] == current_symbol and 
                art['timestamp'] > cutoff_time)
        ]
        
        if len(related_articles) < 2:
            return 0.5
        
        # Calculate sentiment agreement
        sentiments = [art['sentiment_score'] for art in related_articles]
        sentiment_agreement = self._calculate_sentiment_agreement(current_sentiment, sentiments)
        
        # Topic matching
        current_topic = current_analysis.get('topic', 'general')
        topic_matches = sum(1 for art in related_articles if art.get('topic') == current_topic)
        topic_score = topic_matches / len(related_articles)
        
        return sentiment_agreement * 0.7 + topic_score * 0.3
    
    def _calculate_sentiment_agreement(self, target_sentiment: float, other_sentiments: List[float]) -> float:
        """Calculate agreement between sentiment scores"""
        if not other_sentiments:
            return 0.5
        
        target_direction = 1 if target_sentiment > 0.1 else (-1 if target_sentiment < -0.1 else 0)
        
        agreements = sum(
            1 for sentiment in other_sentiments
            if (1 if sentiment > 0.1 else (-1 if sentiment < -0.1 else 0)) == target_direction
        )
        
        return agreements / len(other_sentiments)
    
    def cleanup_old_cache(self) -> None:
        """Clean up old cache entries"""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self._cleanup_interval:
            cutoff_time = current_time - (24 * 3600)  # 24 hours
            
            old_keys = [
                key for key, timestamp in self.news_cache.items()
                if timestamp < cutoff_time
            ]
            
            for key in old_keys:
                del self.news_cache[key]
            
            self._last_cleanup = current_time