"""
Market condition filters with testing mode for after-hours development
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


@dataclass
class MarketConditions:
    """Current market conditions assessment with comprehensive data."""
    is_market_hours: bool
    market_regime: str
    market_stress_level: float
    overall_trend: str
    volume_environment: str
    time_of_day_score: float


class MarketFilter:
    """Market condition filters with testing mode for development"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with improved caching strategy."""
        self.fmp_loader = fmp_loader
        self._market_data_cache: Dict[str, float] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)
    
    def _get_market_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get SPY and VIX data with optimized caching."""
        now = datetime.now(timezone.utc)
        
        # Check cache validity
        if (self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_duration and
            all(key in self._market_data_cache for key in ['spy_price', 'vix_level'])):
            return self._market_data_cache['spy_price'], self._market_data_cache['vix_level']
        
        try:
            # Batch request for both SPY and VIX
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            if market_data is None or market_data.empty:
                return None, None
                
            spy_price = None
            vix_level = None
            
            spy_row = market_data[market_data['symbol'] == 'SPY']
            vix_row = market_data[market_data['symbol'] == 'VIX']
            
            if not spy_row.empty:
                spy_price = spy_row.iloc[0].get('lastSalePrice')
            if not vix_row.empty:
                vix_level = vix_row.iloc[0].get('lastSalePrice')
            
            # Update cache
            if spy_price is not None:
                self._market_data_cache['spy_price'] = spy_price
            if vix_level is not None:
                self._market_data_cache['vix_level'] = vix_level
            
            self._cache_timestamp = now
            
            return spy_price, vix_level
            
        except Exception as e:
            log_warning(f"Error fetching market data: {e}")
            return None, None
    
    def get_market_conditions(self) -> MarketConditions:
        """Assess current market conditions with testing mode support"""
        now = datetime.now(timezone.utc)
        
        # Convert to ET for market hours (assuming EST/EDT)
        et_offset = timedelta(hours=-5)  # EST offset (adjust for DST as needed)
        et_time = (now + et_offset).time()
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        is_market_hours = (
            market_open <= et_time <= market_close and
            now.weekday() < 5  # Monday=0, Friday=4
        )
        
        # Testing mode override
        if CONFIG.testing_mode:
            log_info(f"[TESTING MODE] Overriding market hours check (actual: {et_time.strftime('%H:%M')} ET)")
            is_market_hours = True
        
        # Time of day scoring
        time_score = 0.8 if CONFIG.testing_mode else self._calculate_time_score(et_time)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Market stress assessment
        stress_level = self._calculate_stress_level(vix_level)
        
        # Market regime classification
        regime = self._classify_market_regime(vix_level, spy_price)
        
        # Testing mode adjustments
        if CONFIG.testing_mode:
            if stress_level > 0.8:
                stress_level = 0.6
                log_info(f"[TESTING MODE] Reduced stress level to {stress_level}")
            
            if regime == 'volatile':
                regime = 'neutral'
                log_info(f"[TESTING MODE] Changed regime from volatile to {regime}")
        
        return MarketConditions(
            is_market_hours=is_market_hours,
            market_regime=regime,
            market_stress_level=stress_level,
            overall_trend='neutral',
            volume_environment='normal',
            time_of_day_score=time_score
        )
    
    def _calculate_stress_level(self, vix_level: Optional[float]) -> float:
        """Calculate market stress level with improved thresholds."""
        if vix_level is None:
            return 0.3
        
        # Vectorized stress calculation using numpy for better performance
        stress_thresholds = np.array([12, 16, 20, 25, 30, 40])
        stress_values = np.array([0.05, 0.15, 0.25, 0.4, 0.6, 0.8, 0.95])
        
        # Find the appropriate stress level
        return float(np.interp(vix_level, stress_thresholds, stress_values[:-1]))
    
    def _classify_market_regime(self, vix_level: Optional[float], spy_price: Optional[float]) -> str:
        """Classify market regime with improved logic."""
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
    
    def _calculate_time_score(self, current_time: time) -> float:
        """Calculate optimized time-of-day score."""
        hour = current_time.hour
        minute = current_time.minute
        
        # Convert to minutes from market open (9:30 AM = 0)
        if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            return 0.1
        
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring using interpolation
        time_points = np.array([0, 30, 90, 150, 210, 270, 360, 390])
        score_points = np.array([0.7, 0.9, 0.8, 0.3, 0.7, 0.85, 0.9, 0.1])
        
        return float(np.interp(minutes_from_open, time_points, score_points))
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine trading suitability with testing mode support"""
        
        if CONFIG.testing_mode:
            log_info("[TESTING MODE] Market condition checks with testing overrides")
        
        # Market hours requirement
        if not market_conditions.is_market_hours:
            if CONFIG.testing_mode:
                log_info("[TESTING MODE] Would normally reject due to market hours")
            else:
                now = datetime.now(timezone.utc)
                et_offset = timedelta(hours=-5)
                et_time = (now + et_offset).time()
                return False, f"MARKET_CLOSED - Current time: {et_time.strftime('%H:%M')} ET"
        
        # Time of day filter
        if market_conditions.time_of_day_score < 0.4:
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Ignoring poor trading time (score: {market_conditions.time_of_day_score:.2f})")
            else:
                return False, f"poor_trading_time (score: {market_conditions.time_of_day_score:.2f})"
        
        # Stress level filter
        if market_conditions.market_stress_level > 0.85:
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Would normally reject due to extreme stress ({market_conditions.market_stress_level:.2f})")
            else:
                return False, f"extreme_market_stress (level: {market_conditions.market_stress_level:.2f})"
        
        # Volatile regime filter
        if market_conditions.market_regime == 'volatile':
            if CONFIG.testing_mode:
                log_info("[TESTING MODE] Ignoring volatile market regime")
            else:
                return False, f"volatile_market_regime (stress: {market_conditions.market_stress_level:.2f})"
        
        trading_reason = f"conditions_favorable (regime: {market_conditions.market_regime})"
        if CONFIG.testing_mode:
            trading_reason = f"[TESTING MODE] {trading_reason}"
        
        return True, trading_reason


class NewsQualityFilter:
    """Optimized news quality and freshness filters."""
    
    def __init__(self) -> None:
        """Initialize with efficient data structures."""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 1000
        
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter news with optimized processing pipeline."""
        if news_df is None or news_df.empty:
            return news_df
        
        # Pipeline processing with method chaining
        return (news_df
                .pipe(self._filter_stale_news)
                .pipe(self._deduplicate_headlines)
                .pipe(self._filter_content_quality)
                .pipe(self._add_priority_scoring))
    
    def _filter_stale_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter stale news with timezone-aware comparison."""
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
    
    def _deduplicate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Efficient headline deduplication using vectorized operations."""
        if 'title' not in news_df.columns or news_df.empty:
            return news_df
        
        try:
            # Simple deduplication first
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # Similarity check for smaller datasets only
            if len(news_df) <= 100:
                return self._similarity_deduplication(news_df)
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _similarity_deduplication(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Perform similarity-based deduplication for small datasets."""
        filtered_indices = []
        seen_word_sets = []
        
        for idx, row in news_df.iterrows():
            title = str(row['title']).lower().strip()
            title_words = set(title.split())
            
            # Check similarity with existing headlines
            is_duplicate = any(
                len(title_words & seen_words) / len(title_words | seen_words) > 0.8
                for seen_words in seen_word_sets
                if title_words and seen_words
            )
            
            if not is_duplicate:
                filtered_indices.append(idx)
                seen_word_sets.append(title_words)
                
                # Limit memory usage
                if len(seen_word_sets) > self._headline_cache_limit:
                    seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
        
        return news_df.loc[filtered_indices]
    
    def _filter_content_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized content quality filtering."""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Title filters
            if 'title' in news_df.columns:
                mask &= news_df['title'].str.len() >= 20
                
                # Vectorized suspicious content filter
                suspicious_patterns = ['click here', 'ad:', 'advertisement', 'sponsored']
                for pattern in suspicious_patterns:
                    mask &= ~news_df['title'].str.lower().str.contains(pattern, na=False)
            
            # Text length filter
            if 'text' in news_df.columns:
                mask &= news_df['text'].str.len() >= 100
            
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return news_df
    
    def _add_priority_scoring(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scoring with vectorized operations."""
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
            
            for keyword in official_keywords:
                mask = news_df['title'].str.lower().str.contains(keyword, na=False)
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
    """Filter based on price action and technical conditions."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize price action filter."""
        self.fmp_loader = fmp_loader
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Filter symbols based on price action criteria."""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        try:
            filtered_symbols = []
            
            for symbol in symbols:
                price_row = current_prices[current_prices['symbol'] == symbol]
                if price_row.empty:
                    continue
                
                price_data = price_row.iloc[0]
                current_price = float(price_data.get('lastSalePrice', 0))
                volume = float(price_data.get('volume', 0))
                
                # Basic price action filters
                if (current_price >= CONFIG.min_price and 
                    current_price <= CONFIG.max_price and
                    volume >= CONFIG.min_volume):
                    filtered_symbols.append(symbol)
            
            return filtered_symbols
            
        except Exception as e:
            log_warning(f"Error in price action filtering: {e}")
            return symbols


class PortfolioRiskFilter:
    """Portfolio risk management filters."""
    
    def __init__(self, trader) -> None:
        """Initialize portfolio risk filter."""
        self.trader = trader
    
    def check_portfolio_limits(self, symbol: str, side: str, position_size: float) -> Tuple[bool, str]:
        """Check if trade meets portfolio risk limits."""
        try:
            active_positions = self.trader.get_active_positions()
            
            # Check maximum positions
            if len(active_positions) >= 10:  # Max 10 positions
                return False, "max_positions_exceeded"
            
            # Check symbol concentration
            symbol_count = sum(1 for trade in active_positions if trade.symbol == symbol)
            if symbol_count >= 2:  # Max 2 positions per symbol
                return False, "symbol_concentration_limit"
            
            # Check total exposure
            total_exposure = sum(trade.position_size for trade in active_positions)
            if total_exposure + position_size > CONFIG.position_size * 10:  # Max 10x base position
                return False, "total_exposure_limit"
            
            return True, "within_limits"
            
        except Exception as e:
            log_warning(f"Error checking portfolio limits: {e}")
            return False, f"check_error: {e}"


class EntryTimingOptimizer:
    """Optimize entry timing for trades."""
    
    def __init__(self) -> None:
        """Initialize entry timing optimizer."""
        self._pending_entries: Dict[str, datetime] = {}
    
    def should_enter_now(self, symbol: str, analysis, current_price: float) -> Tuple[bool, str]:
        """Determine if now is a good time to enter."""
        try:
            # Simple timing logic for now
            # Could be enhanced with more sophisticated timing algorithms
            
            # Check if we've seen this symbol recently
            if symbol in self._pending_entries:
                last_seen = self._pending_entries[symbol]
                if datetime.now() - last_seen < timedelta(minutes=5):
                    return False, "recent_analysis_cooldown"
            
            # Update last seen time
            self._pending_entries[symbol] = datetime.now()
            
            return True, "timing_optimal"
            
        except Exception as e:
            log_warning(f"Error in entry timing check: {e}")
            return True, "timing_check_error"
    
    def cleanup_stale_entries(self) -> None:
        """Clean up stale entries from memory."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            self._pending_entries = {
                symbol: timestamp 
                for symbol, timestamp in self._pending_entries.items()
                if timestamp > cutoff_time
            }
        except Exception as e:
            log_warning(f"Error cleaning up stale entries: {e}")