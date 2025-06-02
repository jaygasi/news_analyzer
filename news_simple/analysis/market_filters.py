"""
Optimized market condition filters with improved testing mode and fixed type issues
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
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


class MarketFilter:
    """Optimized market condition filters with enhanced caching"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized caching strategy"""
        self.fmp_loader = fmp_loader
        self._market_data_cache: Dict[str, float] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=10)
    
    @lru_cache(maxsize=128)
    def _get_market_data_cached(self, cache_key: str) -> Tuple[Optional[float], Optional[float]]:
        """Cached market data retrieval"""
        try:
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            if market_data is None or market_data.empty:
                return None, None
                
            spy_price = None
            vix_level = None
            
            spy_row = market_data[market_data['symbol'] == 'SPY']
            vix_row = market_data[market_data['symbol'] == 'VIX']
            
            if not spy_row.empty:
                spy_price = float(spy_row.iloc[0].get('lastSalePrice', 0))
            if not vix_row.empty:
                vix_level = float(vix_row.iloc[0].get('lastSalePrice', 0))
            
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
        
        # Generate cache key
        cache_key = f"market_data_{now.strftime('%Y%m%d_%H%M')}"
        
        spy_price, vix_level = self._get_market_data_cached(cache_key)
        
        # Update cache
        if spy_price is not None:
            self._market_data_cache['spy_price'] = spy_price
        if vix_level is not None:
            self._market_data_cache['vix_level'] = vix_level
        
        self._cache_timestamp = now
        
        return spy_price, vix_level
    
    def get_market_conditions(self) -> MarketConditions:
        """Get current market conditions with optimized logic"""
        now = datetime.now(timezone.utc)
        
        # Market hours and time scoring
        is_market_hours, time_score = self._calculate_market_hours_and_time_score(now)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Market stress and regime
        stress_level = self._calculate_stress_level_fast(vix_level)
        regime = self._classify_market_regime_fast(vix_level)
        
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
        """Calculate market hours and time score - reduced complexity"""
        # Convert to ET for market hours
        et_offset = timedelta(hours=-5)
        et_time = (now + et_offset).time()
        
        # Market hours check
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        is_market_hours = (
            market_open <= et_time <= market_close and
            now.weekday() < 5
        )
        
        # Testing mode override
        if CONFIG.testing_mode:
            log_debug(f"[TESTING MODE] Overriding market hours check (actual: {et_time.strftime('%H:%M')} ET)")
            is_market_hours = True
        
        # Time scoring
        time_score = 0.8 if CONFIG.testing_mode else self._calculate_time_score_fast(et_time)
        
        return is_market_hours, time_score
    
    def _apply_testing_mode_adjustments(self, stress_level: float, regime: str) -> Tuple[float, str]:
        """Apply testing mode adjustments - extracted for clarity"""
        if CONFIG.testing_mode:
            if stress_level > 0.8:
                stress_level = 0.6
                log_debug(f"[TESTING MODE] Reduced stress level to {stress_level}")
            
            if regime == 'volatile':
                regime = 'neutral'
                log_debug(f"[TESTING MODE] Changed regime to {regime}")
        
        return stress_level, regime
    
    def _calculate_stress_level_fast(self, vix_level: Optional[float]) -> float:
        """Optimized stress level calculation"""
        if vix_level is None:
            return 0.3
        
        # Use numpy for faster calculation
        stress_thresholds = np.array([12, 16, 20, 25, 30, 40], dtype=np.float64)
        stress_values = np.array([0.05, 0.15, 0.25, 0.4, 0.6, 0.8], dtype=np.float64)
        
        return float(np.interp(vix_level, stress_thresholds, stress_values))
    
    def _classify_market_regime_fast(self, vix_level: Optional[float]) -> str:
        """Fast market regime classification - removed unused spy_price parameter"""
        if vix_level is None:
            return 'neutral'
        
        if vix_level > 35:
            return 'volatile'
        elif vix_level > 28:
            return 'bear'
        elif vix_level < 14:
            return 'bull'
        else:
            return 'neutral'
    
    def _calculate_time_score_fast(self, current_time: time) -> float:
        """Optimized time-of-day score calculation"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Quick return for off-hours
        if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            return 0.1
        
        # Convert to minutes from market open
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring using pre-computed lookup
        time_scores = {
            0: 0.7, 30: 0.9, 90: 0.8, 150: 0.3, 210: 0.7, 270: 0.85, 360: 0.9, 390: 0.1
        }
        
        # Find closest time point
        closest_time = min(time_scores.keys(), key=lambda x: abs(x - minutes_from_open))
        return time_scores[closest_time]
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Optimized trading suitability check - reduced complexity"""
        
        if CONFIG.testing_mode:
            log_debug("[TESTING MODE] Market condition checks with testing overrides")
        
        # Check each condition separately for clarity
        market_hours_check = self._check_market_hours(market_conditions)
        if market_hours_check[0] is False:
            return market_hours_check
        
        time_check = self._check_time_of_day(market_conditions)
        if time_check[0] is False:
            return time_check
        
        stress_check = self._check_stress_level(market_conditions)
        if stress_check[0] is False:
            return stress_check
        
        regime_check = self._check_market_regime(market_conditions)
        if regime_check[0] is False:
            return regime_check
        
        # All checks passed
        trading_reason = f"conditions_favorable (regime: {market_conditions.market_regime})"
        if CONFIG.testing_mode:
            trading_reason = f"[TESTING MODE] {trading_reason}"
        
        return True, trading_reason
    
    def _check_market_hours(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Check market hours condition"""
        if not market_conditions.is_market_hours:
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Would normally reject due to market hours")
                return True, "testing_mode_override"
            else:
                now = datetime.now(timezone.utc)
                et_offset = timedelta(hours=-5)
                et_time = (now + et_offset).time()
                return False, f"MARKET_CLOSED - Current time: {et_time.strftime('%H:%M')} ET"
        return True, "market_hours_ok"
    
    def _check_time_of_day(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Check time of day condition"""
        if market_conditions.time_of_day_score < 0.4:
            if CONFIG.testing_mode:
                log_debug(f"[TESTING MODE] Ignoring poor trading time (score: {market_conditions.time_of_day_score:.2f})")
                return True, "testing_mode_override"
            else:
                return False, f"poor_trading_time (score: {market_conditions.time_of_day_score:.2f})"
        return True, "time_of_day_ok"
    
    def _check_stress_level(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Check market stress level condition"""
        if market_conditions.market_stress_level > 0.85:
            if CONFIG.testing_mode:
                log_debug(f"[TESTING MODE] Would normally reject due to extreme stress ({market_conditions.market_stress_level:.2f})")
                return True, "testing_mode_override"
            else:
                return False, f"extreme_market_stress (level: {market_conditions.market_stress_level:.2f})"
        return True, "stress_level_ok"
    
    def _check_market_regime(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Check market regime condition"""
        if market_conditions.market_regime == 'volatile':
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Ignoring volatile market regime")
                return True, "testing_mode_override"
            else:
                return False, f"volatile_market_regime (stress: {market_conditions.market_stress_level:.2f})"
        return True, "regime_ok"


class NewsQualityFilter:
    """Optimized news quality filters with improved performance"""
    
    def __init__(self) -> None:
        """Initialize with efficient data structures"""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 500
        
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized news filtering pipeline - reduced complexity"""
        if news_df is None or news_df.empty:
            return news_df
        
        # Apply filters in sequence
        filtered_df = news_df
        filtered_df = self._filter_stale_news(filtered_df)
        filtered_df = self._deduplicate_headlines_fast(filtered_df)
        filtered_df = self._filter_content_quality_fast(filtered_df)
        filtered_df = self._add_priority_scoring_fast(filtered_df)
        
        return filtered_df
    
    def _filter_stale_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter stale news with timezone-aware comparison"""
        if 'publishedDate' not in news_df.columns:
            return news_df
        
        try:
            now = datetime.now(timezone.utc)
            cutoff_hours = 24 if CONFIG.testing_mode else 2
            cutoff_time = now - timedelta(hours=cutoff_hours)
            
            if CONFIG.testing_mode:
                log_debug("[TESTING MODE] Extended news freshness to 24 hours")
            
            return news_df[news_df['publishedDate'] > cutoff_time]
            
        except Exception as e:
            log_warning(f"Error filtering stale news: {e}")
            return news_df
    
    def _deduplicate_headlines_fast(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fast headline deduplication - reduced complexity"""
        if 'title' not in news_df.columns or news_df.empty:
            return news_df
        
        try:
            # Simple exact deduplication first
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # For smaller datasets, do similarity check
            if len(news_df) <= 50:
                return self._similarity_deduplication_fast(news_df)
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _similarity_deduplication_fast(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fast similarity-based deduplication"""
        if len(news_df) <= 1:
            return news_df
        
        filtered_indices = []
        seen_word_sets = []
        
        for idx, row in news_df.iterrows():
            title = str(row['title']).lower().strip()
            title_words = set(title.split())
            
            if not self._is_similar_to_seen(title_words, seen_word_sets):
                filtered_indices.append(idx)
                seen_word_sets.append(title_words)
                
                # Limit memory usage
                if len(seen_word_sets) > self._headline_cache_limit:
                    seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
        
        return news_df.loc[filtered_indices]
    
    def _is_similar_to_seen(self, title_words: set, seen_word_sets: List[set]) -> bool:
        """Check if title is similar to previously seen titles"""
        for seen_words in seen_word_sets:
            if title_words and seen_words:
                intersection_size = len(title_words & seen_words)
                union_size = len(title_words | seen_words)
                if union_size > 0 and intersection_size / union_size > 0.8:
                    return True
        return False
    
    def _filter_content_quality_fast(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized content quality filtering"""
        if news_df.empty:
            return news_df
        
        try:
            # Create boolean mask for all conditions
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Apply filters
            mask = self._apply_title_filters(news_df, mask)
            mask = self._apply_text_filters(news_df, mask)
            
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return news_df
    
    def _apply_title_filters(self, news_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Apply title-based filters"""
        if 'title' in news_df.columns:
            mask &= news_df['title'].str.len() >= 20
            
            # Vectorized suspicious content filter
            suspicious_patterns = ['click here', 'ad:', 'advertisement', 'sponsored']
            for pattern in suspicious_patterns:
                mask &= ~news_df['title'].str.lower().str.contains(pattern, na=False, regex=False)
        
        return mask
    
    def _apply_text_filters(self, news_df: pd.DataFrame, mask: pd.Series) -> pd.Series:
        """Apply text-based filters"""
        if 'text' in news_df.columns:
            mask &= news_df['text'].str.len() >= 100
        
        return mask
    
    def _add_priority_scoring_fast(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Fast priority scoring with vectorized operations"""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        try:
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Vectorized keyword matching
            official_keywords = [
                'announces', 'reports', 'declares', 'files', 'receives',
                'completes', 'signs', 'launches', 'enters into', 'appoints'
            ]
            
            # Single pass through keywords
            for keyword in official_keywords:
                mask = news_df['title'].str.lower().str.contains(keyword, na=False, regex=False)
                news_df.loc[mask, 'priority_score'] += 0.1
            
            # Sort by priority and timestamp
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            return news_df.sort_values(sort_columns, ascending=[False, False])
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df


class PriceActionFilter:
    """Optimized price action filter"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize price action filter"""
        self.fmp_loader = fmp_loader
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Fast price action filtering"""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        try:
            # Convert to dict for faster lookup
            price_data = current_prices.set_index('symbol').to_dict('index')
            
            filtered_symbols = []
            
            for symbol in symbols:
                if symbol not in price_data:
                    continue
                
                if self._meets_price_criteria(price_data[symbol]):
                    filtered_symbols.append(symbol)
            
            return filtered_symbols
            
        except Exception as e:
            log_warning(f"Error in price action filtering: {e}")
            return symbols
    
    def _meets_price_criteria(self, price_row: Dict) -> bool:
        """Check if price data meets criteria"""
        try:
            current_price = float(price_row.get('lastSalePrice', 0))
            volume = float(price_row.get('volume', 0))
            
            return (CONFIG.min_price <= current_price <= CONFIG.max_price and
                    volume >= CONFIG.min_volume)
        except (ValueError, TypeError):
            return False


class PortfolioRiskFilter:
    """Optimized portfolio risk management"""
    
    def __init__(self, trader) -> None:
        """Initialize portfolio risk filter"""
        self.trader = trader
    
    def check_portfolio_limits(self, symbol: str, position_size: float) -> Tuple[bool, str]:
        """Fast portfolio limit checking - removed unused side parameter"""
        try:
            active_positions = self.trader.get_active_positions()
            
            # Quick checks with early returns
            if len(active_positions) >= 10:
                return False, "max_positions_exceeded"
            
            # Count symbol positions
            symbol_count = sum(1 for trade in active_positions if trade.symbol == symbol)
            if symbol_count >= 2:
                return False, "symbol_concentration_limit"
            
            # Check total exposure
            total_exposure = sum(trade.position_size for trade in active_positions)
            if total_exposure + position_size > CONFIG.position_size * 10:
                return False, "total_exposure_limit"
            
            return True, "within_limits"
            
        except Exception as e:
            log_warning(f"Error checking portfolio limits: {e}")
            return False, f"check_error: {e}"


class EntryTimingOptimizer:
    """Optimized entry timing with reduced memory usage"""
    
    def __init__(self) -> None:
        """Initialize entry timing optimizer"""
        self._pending_entries: Dict[str, datetime] = {}
        self._max_entries = 100
    
    def should_enter_now(self, symbol: str, analysis) -> Tuple[bool, str]:
        """Fast entry timing check - removed unused current_price parameter"""
        try:
            # Check recent analysis cooldown
            if symbol in self._pending_entries:
                last_seen = self._pending_entries[symbol]
                if datetime.now() - last_seen < timedelta(minutes=5):
                    return False, "recent_analysis_cooldown"
            
            # Update last seen time
            self._pending_entries[symbol] = datetime.now()
            
            # Cleanup if too many entries
            if len(self._pending_entries) > self._max_entries:
                self.cleanup_stale_entries()
            
            return True, "timing_optimal"
            
        except Exception as e:
            log_warning(f"Error in entry timing check: {e}")
            return True, "timing_check_error"
    
    def cleanup_stale_entries(self) -> None:
        """Cleanup old entries"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            old_symbols = [
                symbol for symbol, timestamp in self._pending_entries.items()
                if timestamp < cutoff_time
            ]
            
            for symbol in old_symbols:
                self._pending_entries.pop(symbol, None)
                
            log_debug(f"Cleaned up {len(old_symbols)} stale timing entries")
            
        except Exception as e:
            log_warning(f"Error cleaning up stale entries: {e}")