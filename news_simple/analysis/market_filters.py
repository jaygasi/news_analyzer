"""
Optimized market condition filters with improved performance and fixed imports
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


@dataclass
class MarketConditions:
    """Current market conditions assessment"""
    is_market_hours: bool
    market_regime: str
    market_stress_level: float
    overall_trend: str
    volume_environment: str
    time_of_day_score: float


class OptimizedMarketFilter:
    """Optimized market condition filters with enhanced caching and performance"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized caching strategy"""
        self.fmp_loader = fmp_loader
        self._market_data_cache: Dict[str, float] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=8)  # Optimized cache duration
        
        # Pre-compute common values for efficiency
        self._market_open = dt_time(9, 30)
        self._market_close = dt_time(16, 0)
        self._et_offset = timedelta(hours=-5)
    
    @lru_cache(maxsize=64)
    def _get_market_data_cached(self, cache_key: str) -> Tuple[Optional[float], Optional[float]]:
        """Cached market data retrieval with optimized error handling"""
        try:
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            if market_data is None or market_data.empty:
                return None, None
                
            spy_price = None
            vix_level = None
            
            # Optimized symbol lookup
            for _, row in market_data.iterrows():
                symbol = row.get('symbol', '')
                if symbol == 'SPY':
                    spy_price = float(row.get('lastSalePrice', 0))
                elif symbol == 'VIX':
                    vix_level = float(row.get('lastSalePrice', 0))
                
                # Early exit if both found
                if spy_price is not None and vix_level is not None:
                    break
            
            return spy_price, vix_level
            
        except Exception as e:
            log_warning(f"Error fetching market data: {e}")
            return None, None
    
    def _get_market_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get market data with optimized caching"""
        now = datetime.now(timezone.utc)
        
        # Check cache validity
        if (self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_duration and
            'spy_price' in self._market_data_cache and
            'vix_level' in self._market_data_cache):
            return self._market_data_cache['spy_price'], self._market_data_cache['vix_level']
        
        # Generate cache key with minute precision
        cache_key = f"market_data_{now.strftime('%Y%m%d_%H%M')}"
        
        spy_price, vix_level = self._get_market_data_cached(cache_key)
        
        # Update cache efficiently
        if spy_price is not None:
            self._market_data_cache['spy_price'] = spy_price
        if vix_level is not None:
            self._market_data_cache['vix_level'] = vix_level
        
        self._cache_timestamp = now
        
        return spy_price, vix_level
    
    def get_market_conditions(self) -> MarketConditions:
        """Get current market conditions with optimized calculations"""
        now = datetime.now(timezone.utc)
        
        # Optimized market hours and time scoring
        is_market_hours, time_score = self._calculate_market_hours_and_time_score(now)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Optimized stress level and regime calculation
        stress_level = self._calculate_stress_level_optimized(vix_level)
        regime = self._classify_market_regime_optimized(vix_level)
        
        # Apply testing mode adjustments
        stress_level, regime = self._apply_testing_mode_adjustments(stress_level, regime)
        
        return MarketConditions(
            is_market_hours=is_market_hours,
            market_regime=regime,
            market_stress_level=stress_level,
            overall_trend='neutral',
            volume_environment='normal',
            time_of_day_score=time_score
        )
    
    def _calculate_market_hours_and_time_score(self, now: datetime) -> Tuple[bool, float]:
        """Optimized market hours calculation with pre-computed values"""
        # Convert to ET using pre-computed offset
        et_time = (now + self._et_offset).time()
        
        # Market hours check using pre-computed time objects
        is_market_hours = (
            self._market_open <= et_time <= self._market_close and
            now.weekday() < 5
        )
        
        # Testing mode override
        if CONFIG.testing_mode:
            log_debug(f"[TESTING MODE] Overriding market hours check (actual: {et_time.strftime('%H:%M')} ET)")
            is_market_hours = True
        
        # Optimized time scoring
        time_score = 0.75 if CONFIG.testing_mode else self._calculate_time_score_optimized(et_time)
        
        return is_market_hours, time_score
    
    def _apply_testing_mode_adjustments(self, stress_level: float, regime: str) -> Tuple[float, str]:
        """Apply testing mode adjustments efficiently"""
        if CONFIG.testing_mode:
            if stress_level > 0.75:
                stress_level = 0.55
                log_debug(f"[TESTING MODE] Reduced stress level to {stress_level}")
            
            if regime == 'volatile':
                regime = 'neutral'
                log_debug(f"[TESTING MODE] Changed regime to {regime}")
        
        return stress_level, regime
    
    def _calculate_stress_level_optimized(self, vix_level: Optional[float]) -> float:
        """Optimized stress level calculation using numpy interpolation"""
        if vix_level is None:
            return 0.25
        
        # Pre-computed thresholds for better performance
        if vix_level <= 12:
            return 0.05
        elif vix_level <= 16:
            return 0.15
        elif vix_level <= 20:
            return 0.25
        elif vix_level <= 25:
            return 0.40
        elif vix_level <= 30:
            return 0.60
        else:
            return min(0.85, 0.60 + (vix_level - 30) * 0.025)
    
    def _classify_market_regime_optimized(self, vix_level: Optional[float]) -> str:
        """Optimized market regime classification"""
        if vix_level is None:
            return 'neutral'
        
        # Simple threshold-based classification for speed
        if vix_level > 32:
            return 'volatile'
        elif vix_level > 26:
            return 'bear'
        elif vix_level < 15:
            return 'bull'
        else:
            return 'neutral'
    
    def _calculate_time_score_optimized(self, current_time: dt_time) -> float:
        """Optimized time-of-day score using lookup table"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Quick return for off-hours
        if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            return 0.1
        
        # Convert to minutes from market open for efficiency
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring using simple conditions
        if minutes_from_open < 30:      # First 30 minutes
            return 0.65
        elif minutes_from_open < 90:    # 30-90 minutes
            return 0.85
        elif minutes_from_open < 150:   # 90-150 minutes (lunch)
            return 0.75
        elif minutes_from_open < 210:   # Afternoon lull
            return 0.30
        elif minutes_from_open < 270:   # Late afternoon pickup
            return 0.65
        elif minutes_from_open < 360:   # Power hour approach
            return 0.80
        elif minutes_from_open < 390:   # Power hour
            return 0.85
        else:                           # Last 30 minutes
            return 0.15
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Optimized trading suitability check with early returns"""
        
        if CONFIG.testing_mode:
            log_debug("[TESTING MODE] Market condition checks with testing overrides")
        
        # Optimized condition checking with early returns
        
        # Market hours check
        if not market_conditions.is_market_hours:
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Would normally reject due to market hours")
                return True, "testing_mode_override"
            else:
                now = datetime.now(timezone.utc)
                et_time = (now + self._et_offset).time()
                return False, f"MARKET_CLOSED - Current time: {et_time.strftime('%H:%M')} ET"
        
        # Time of day check
        if market_conditions.time_of_day_score < 0.35:
            if CONFIG.testing_mode:
                log_debug(f"[TESTING MODE] Ignoring poor trading time (score: {market_conditions.time_of_day_score:.2f})")
                return True, "testing_mode_override"
            else:
                return False, f"poor_trading_time (score: {market_conditions.time_of_day_score:.2f})"
        
        # Stress level check
        if market_conditions.market_stress_level > 0.80:
            if CONFIG.testing_mode:
                log_debug(f"[TESTING MODE] Would normally reject due to high stress ({market_conditions.market_stress_level:.2f})")
                return True, "testing_mode_override"
            else:
                return False, f"high_market_stress (level: {market_conditions.market_stress_level:.2f})"
        
        # Market regime check
        if market_conditions.market_regime == 'volatile':
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Ignoring volatile market regime")
                return True, "testing_mode_override"
            else:
                return False, f"volatile_market_regime (stress: {market_conditions.market_stress_level:.2f})"
        
        # All checks passed
        trading_reason = f"conditions_favorable (regime: {market_conditions.market_regime})"
        if CONFIG.testing_mode:
            trading_reason = f"[TESTING MODE] {trading_reason}"
        
        return True, trading_reason


class OptimizedNewsQualityFilter:
    """Optimized news quality filters with vectorized operations"""
    
    def __init__(self) -> None:
        """Initialize with optimized data structures"""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 400  # Optimized cache size
        
        # Pre-compile regex patterns for better performance
        import re
        self._spam_pattern = re.compile(
            r'\b(click here|ad:|advertisement|sponsored)\b', 
            re.IGNORECASE
        )
        
        # Pre-define keyword sets for vectorized operations
        self._official_keywords = {
            'announces', 'reports', 'declares', 'files', 'receives',
            'completes', 'signs', 'launches', 'enters into', 'appoints'
        }
        
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized news filtering pipeline with vectorized operations"""
        if news_df is None or news_df.empty:
            return news_df
        
        # Apply filters in optimized sequence
        filtered_df = news_df.copy()
        filtered_df = self._filter_stale_news_vectorized(filtered_df)
        filtered_df = self._deduplicate_headlines_optimized(filtered_df)
        filtered_df = self._filter_content_quality_vectorized(filtered_df)
        filtered_df = self._add_priority_scoring_vectorized(filtered_df)
        
        return filtered_df
    
    def _filter_stale_news_vectorized(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized stale news filtering"""
        if 'publishedDate' not in news_df.columns:
            return news_df
        
        try:
            now = datetime.now(timezone.utc)
            cutoff_hours = 20 if CONFIG.testing_mode else 1.5  # More aggressive
            cutoff_time = now - timedelta(hours=cutoff_hours)
            
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Extended news freshness to 20 hours")
            
            # Vectorized time filtering
            time_mask = news_df['publishedDate'] > cutoff_time
            return news_df[time_mask]
            
        except Exception as e:
            log_warning(f"Error filtering stale news: {e}")
            return news_df
    
    def _deduplicate_headlines_optimized(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized headline deduplication"""
        if 'title' not in news_df.columns or news_df.empty:
            return news_df
        
        try:
            # Simple exact deduplication first (fastest)
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # For smaller datasets, do similarity check
            if len(news_df) <= 30:  # Reduced threshold for performance
                return self._similarity_deduplication_optimized(news_df)
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _similarity_deduplication_optimized(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized similarity-based deduplication"""
        if len(news_df) <= 1:
            return news_df
        
        filtered_indices = []
        seen_word_sets = []
        
        for idx, row in news_df.iterrows():
            title = str(row['title']).lower().strip()
            title_words = set(title.split())
            
            # Skip very short titles
            if len(title_words) < 3:
                continue
            
            if not self._is_similar_to_seen_optimized(title_words, seen_word_sets):
                filtered_indices.append(idx)
                seen_word_sets.append(title_words)
                
                # Limit memory usage
                if len(seen_word_sets) > self._headline_cache_limit:
                    seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
        
        return news_df.loc[filtered_indices]
    
    def _is_similar_to_seen_optimized(self, title_words: set, seen_word_sets: List[set]) -> bool:
        """Optimized similarity check with early termination"""
        for seen_words in seen_word_sets:
            if title_words and seen_words:
                intersection_size = len(title_words & seen_words)
                if intersection_size > 0:  # Quick check first
                    union_size = len(title_words | seen_words)
                    if union_size > 0 and intersection_size / union_size > 0.75:  # Stricter threshold
                        return True
        return False
    
    def _filter_content_quality_vectorized(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized content quality filtering"""
        if news_df.empty:
            return news_df
        
        try:
            # Create boolean mask for all conditions
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Vectorized title length filter
            if 'title' in news_df.columns:
                mask &= news_df['title'].str.len() >= 15  # Slightly more permissive
                
                # Vectorized spam filter using pre-compiled regex
                mask &= ~news_df['title'].str.contains(self._spam_pattern, na=False)
            
            # Vectorized text length filter
            if 'text' in news_df.columns:
                mask &= news_df['text'].str.len() >= 80  # Slightly more permissive
            
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return news_df
    
    def _add_priority_scoring_vectorized(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized priority scoring"""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        try:
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Vectorized keyword scoring
            title_lower = news_df['title'].str.lower()
            
            for keyword in self._official_keywords:
                keyword_mask = title_lower.str.contains(keyword, na=False, regex=False)
                news_df.loc[keyword_mask, 'priority_score'] += 0.08  # Slightly reduced
            
            # Clip priority scores
            news_df['priority_score'] = news_df['priority_score'].clip(upper=1.0)
            
            # Optimized sorting
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            return news_df.sort_values(sort_columns, ascending=[False, False])
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df


class OptimizedPriceActionFilter:
    """Optimized price action filter with vectorized operations"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize optimized price action filter"""
        self.fmp_loader = fmp_loader
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Vectorized price action filtering"""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        try:
            # Vectorized filtering using pandas operations
            price_mask = (
                (current_prices['lastSalePrice'] >= CONFIG.min_price) &
                (current_prices['lastSalePrice'] <= CONFIG.max_price) &
                (current_prices['volume'] >= CONFIG.min_volume)
            )
            
            valid_prices = current_prices[price_mask]
            valid_symbols = set(valid_prices['symbol'].tolist())
            
            # Filter input symbols to only those with valid prices
            filtered_symbols = [symbol for symbol in symbols if symbol in valid_symbols]
            
            return filtered_symbols
            
        except Exception as e:
            log_warning(f"Error in price action filtering: {e}")
            return symbols


class OptimizedPortfolioRiskFilter:
    """Optimized portfolio risk management with better performance"""
    
    def __init__(self, trader) -> None:
        """Initialize optimized portfolio risk filter"""
        self.trader = trader
        self._max_positions = 8  # Reduced for better risk management
        self._max_symbol_positions = 1  # Stricter symbol concentration
        self._max_total_exposure_multiplier = 8  # Reduced from 10
    
    def check_portfolio_limits(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Optimized portfolio limit checking with pre-computed limits"""
        try:
            active_positions = self.trader.get_active_positions()
            
            # Quick checks with early returns and pre-computed limits
            if len(active_positions) >= self._max_positions:
                return False, f"max_positions_exceeded_{self._max_positions}"
            
            # Count symbol positions efficiently
            symbol_count = sum(1 for trade in active_positions if trade.symbol == symbol)
            if symbol_count >= self._max_symbol_positions:
                return False, f"symbol_concentration_limit_{self._max_symbol_positions}"
            
            # Check total exposure with pre-computed limit
            total_exposure = sum(trade.position_size for trade in active_positions)
            max_exposure = CONFIG.position_size * self._max_total_exposure_multiplier
            if total_exposure + position_size > max_exposure:
                return False, f"total_exposure_limit_{max_exposure:.0f}"
            
            return True, "within_limits"
            
        except Exception as e:
            log_warning(f"Error checking portfolio limits: {e}")
            return False, f"check_error: {e}"


class OptimizedEntryTimingOptimizer:
    """Optimized entry timing with reduced memory usage and better performance"""
    
    def __init__(self) -> None:
        """Initialize optimized entry timing optimizer"""
        self._pending_entries: Dict[str, float] = {}  # Use timestamp as float for efficiency
        self._max_entries = 80  # Reduced for memory efficiency
        self._cooldown_seconds = 240  # 4 minutes cooldown
        self._last_cleanup = time.time()
        self._cleanup_interval = 1200  # 20 minutes
    
    def should_enter_now(self, symbol: str, analysis) -> Tuple[bool, str]:
        """Optimized entry timing check with automatic cleanup"""
        current_time = time.time()
        
        # Periodic cleanup for memory management
        if current_time - self._last_cleanup > self._cleanup_interval:
            self.cleanup_stale_entries()
        
        # Check recent analysis cooldown
        if symbol in self._pending_entries:
            last_seen = self._pending_entries[symbol]
            if current_time - last_seen < self._cooldown_seconds:
                return False, f"recent_analysis_cooldown_{self._cooldown_seconds}s"
        
        # Update last seen time
        self._pending_entries[symbol] = current_time
        
        # Immediate cleanup if over limit
        if len(self._pending_entries) > self._max_entries:
            self._cleanup_excess_entries()
        
        return True, "timing_optimal"
    
    def cleanup_stale_entries(self) -> None:
        """Optimized cleanup of old entries"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # 1 hour
            
            # Remove old entries efficiently
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
        
        # Sort by timestamp and keep only the most recent entries
        sorted_entries = sorted(
            self._pending_entries.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Keep only the most recent entries
        keep_count = self._max_entries // 2
        self._pending_entries = dict(sorted_entries[:keep_count])


# Create aliases for backwards compatibility
MarketFilter = OptimizedMarketFilter
NewsQualityFilter = OptimizedNewsQualityFilter
PriceActionFilter = OptimizedPriceActionFilter
PortfolioRiskFilter = OptimizedPortfolioRiskFilter
EntryTimingOptimizer = OptimizedEntryTimingOptimizer