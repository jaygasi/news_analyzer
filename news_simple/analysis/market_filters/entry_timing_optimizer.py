"""
Entry timing optimization for trades
"""
import time
from typing import Dict, Tuple
from utils.simple_logger import log_warning, log_debug


class EntryTimingOptimizer:
    """Optimize entry timing for trades"""
    
    def __init__(self) -> None:
        """Initialize timing optimizer"""
        self._pending_entries: Dict[str, float] = {}
        self._max_entries = 80
        self._cooldown_seconds = 240  # 4 minutes
        self._last_cleanup = time.time()
        self._cleanup_interval = 1200  # 20 minutes
    
    def should_enter_now(self, symbol: str, analysis) -> Tuple[bool, str]:
        """Check if entry timing is optimal"""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup_stale_entries()
        
        # Check cooldown period
        if symbol in self._pending_entries:
            last_seen = self._pending_entries[symbol]
            if current_time - last_seen < self._cooldown_seconds:
                return False, f"recent_analysis_cooldown_{self._cooldown_seconds}s"
        
        # Update tracking
        self._pending_entries[symbol] = current_time
        
        # Cleanup if over limit
        if len(self._pending_entries) > self._max_entries:
            self._cleanup_excess_entries()
        
        return True, "timing_optimal"
    
    def cleanup_stale_entries(self) -> None:
        """Clean up old timing entries"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour
            
            old_symbols = [
                symbol for symbol, timestamp in self._pending_entries.items()
                if timestamp < cutoff_time
            ]
            
            for symbol in old_symbols:
                del self._pending_entries[symbol]
            
            self._last_cleanup = current_time
            
            if old_symbols:
                log_debug(f"Cleaned up {len(old_symbols)} stale timing entries")
                
        except Exception as e:
            log_warning(f"Error cleaning up stale entries: {e}")
    
    def _cleanup_excess_entries(self) -> None:
        """Remove excess entries when over limit"""
        if len(self._pending_entries) <= self._max_entries:
            return
        
        # Keep only the most recent entries
        sorted_entries = sorted(
            self._pending_entries.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        keep_count = self._max_entries // 2
        self._pending_entries = dict(sorted_entries[:keep_count])
    
    def get_timing_statistics(self) -> dict:
        """Get timing optimizer statistics"""
        current_time = time.time()
        
        # Count recent entries
        recent_count = sum(
            1 for timestamp in self._pending_entries.values()
            if current_time - timestamp < self._cooldown_seconds
        )
        
        return {
            'total_tracked': len(self._pending_entries),
            'recent_entries': recent_count,
            'cooldown_seconds': self._cooldown_seconds,
            'cleanup_interval': self._cleanup_interval,
            'capacity_utilization': len(self._pending_entries) / self._max_entries
        }