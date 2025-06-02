"""
Recommendation tracking and management
"""
import time
from typing import Dict
from utils.simple_logger import log_debug


class RecommendationTracker:
    """Track and manage trading recommendations"""
    
    def __init__(self):
        self.recent_recommendations: Dict[str, float] = {}  # symbol -> timestamp
        self.recommendation_window_hours = 3
        self.max_tracking = 500
        self._last_cleanup = time.time()
        self._cleanup_interval = 1800  # 30 minutes
    
    def should_recommend(self, symbol: str) -> bool:
        """Check if we should recommend this symbol"""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_recommendations()
        
        if symbol not in self.recent_recommendations:
            return True
        
        last_recommendation = self.recent_recommendations[symbol]
        time_since_last = current_time - last_recommendation
        
        should_recommend = time_since_last >= (self.recommendation_window_hours * 3600)
        
        if not should_recommend:
            log_debug(f"Skipping {symbol} - recommended {time_since_last/3600:.1f}h ago")
        
        return should_recommend
    
    def mark_recommended(self, symbol: str) -> None:
        """Mark symbol as recommended"""
        current_time = time.time()
        self.recent_recommendations[symbol] = current_time
        
        # Immediate cleanup if over limit
        if len(self.recent_recommendations) > self.max_tracking:
            self._cleanup_old_recommendations()
    
    def _cleanup_old_recommendations(self) -> None:
        """Clean up old recommendations"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours
        
        # Remove old entries
        old_symbols = [
            symbol for symbol, timestamp in self.recent_recommendations.items()
            if timestamp < cutoff_time
        ]
        
        for symbol in old_symbols:
            del self.recent_recommendations[symbol]
        
        # If still over limit, remove oldest entries
        if len(self.recent_recommendations) > self.max_tracking:
            sorted_items = sorted(
                self.recent_recommendations.items(), 
                key=lambda x: x[1]
            )
            keep_count = self.max_tracking // 2
            self.recent_recommendations = dict(sorted_items[-keep_count:])
        
        self._last_cleanup = current_time
        
        if old_symbols:
            log_debug(f"Cleaned up {len(old_symbols)} old recommendations")
    
    def get_stats(self) -> dict:
        """Get recommendation tracker statistics"""
        current_time = time.time()
        
        # Count recent recommendations
        recent_count = sum(
            1 for timestamp in self.recent_recommendations.values()
            if current_time - timestamp < (self.recommendation_window_hours * 3600)
        )
        
        return {
            'total_tracked': len(self.recent_recommendations),
            'recent_recommendations': recent_count,
            'window_hours': self.recommendation_window_hours,
            'cleanup_interval_minutes': self._cleanup_interval / 60
        }