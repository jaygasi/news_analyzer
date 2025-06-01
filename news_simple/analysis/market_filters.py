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


class TestingMarketFilter:
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
            'spy_price' in self._market_data_cache and 
            'vix_level' in self._market_data_cache):
            return self._market_data_cache['spy_price'], self._market_data_cache['vix_level']
        
        try:
            # Batch request for both SPY and VIX
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            spy_price = None
            vix_level = None
            
            if market_data is not None and not market_data.empty:
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
        # Use timezone-aware datetime
        now = datetime.now(timezone.utc)
        
        # Convert to ET for market hours (assuming ET timezone)
        et_offset = timedelta(hours=-5)  # EST offset (adjust for DST as needed)
        et_time = (now + et_offset).time()
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        is_market_hours = (
            market_open <= et_time <= market_close and
            now.weekday() < 5  # Monday=0, Friday=4
        )
        
        # TESTING MODE OVERRIDE
        if CONFIG.testing_mode:
            log_info(f"[TESTING MODE] Overriding market hours check (actual: {et_time.strftime('%H:%M')} ET)")
            is_market_hours = True  # Always allow trading in testing mode
        
        # Time of day scoring (simulate normal hours in testing mode)
        if CONFIG.testing_mode:
            # Simulate good trading hours for testing
            time_score = 0.8
        else:
            time_score = self._calculate_time_score(et_time)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Market stress assessment with improved logic
        stress_level = self._calculate_stress_level(vix_level)
        
        # Market regime classification
        regime = self._classify_market_regime(vix_level, spy_price)
        
        # Testing mode adjustments
        if CONFIG.testing_mode:
            # Make conditions more favorable for testing
            if stress_level > 0.8:
                stress_level = 0.6  # Reduce extreme stress for testing
                log_info(f"[TESTING MODE] Reduced stress level to {stress_level}")
            
            if regime == 'volatile':
                regime = 'neutral'  # Change volatile to neutral for testing
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
            return 0.3  # Default moderate stress
        
        # Improved stress calculation with gradual transitions
        if vix_level < 12:
            return 0.05  # Very low stress
        elif vix_level < 16:
            return 0.15  # Low stress
        elif vix_level < 20:
            return 0.25  # Moderate-low stress
        elif vix_level < 25:
            return 0.4   # Moderate stress
        elif vix_level < 30:
            return 0.6   # High stress
        elif vix_level < 40:
            return 0.8   # Very high stress
        else:
            return 0.95  # Extreme stress
    
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
        if hour < 9 or (hour == 9 and minute < 30):
            return 0.1  # Before market open
        elif hour >= 16:
            return 0.1  # After market close
        
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring based on market microstructure
        if 0 <= minutes_from_open < 30:      # 9:30-10:00: Opening volatility
            return 0.7
        elif 30 <= minutes_from_open < 90:   # 10:00-11:00: Strong activity
            return 0.9
        elif 90 <= minutes_from_open < 150:  # 11:00-12:00: Good activity
            return 0.8
        elif 150 <= minutes_from_open < 210: # 12:00-1:30: Lunch lull
            return 0.3
        elif 210 <= minutes_from_open < 270: # 1:30-2:30: Afternoon pickup
            return 0.7
        elif 270 <= minutes_from_open < 360: # 2:30-3:30: Strong activity
            return 0.85
        elif 360 <= minutes_from_open < 390: # 3:30-4:00: End of day
            return 0.9
        else:
            return 0.1
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine trading suitability with testing mode support"""
        
        # Testing mode logging
        if CONFIG.testing_mode:
            log_info("[TESTING MODE] Market condition checks with testing overrides")
        
        # Market hours requirement with detailed message
        if not market_conditions.is_market_hours:
            now = datetime.now(timezone.utc)
            et_offset = timedelta(hours=-5)  # EST offset
            et_time = (now + et_offset).time()
            
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Would normally reject due to market hours: {et_time.strftime('%H:%M')} ET")
                # Continue to other checks in testing mode
            else:
                return False, f"MARKET_CLOSED - Current time: {et_time.strftime('%H:%M')} ET (Market: 9:30-16:00 ET)"
        
        # Time of day filter with lower threshold
        if market_conditions.time_of_day_score < 0.4:
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Ignoring poor trading time (score: {market_conditions.time_of_day_score:.2f})")
            else:
                return False, f"poor_trading_time (score: {market_conditions.time_of_day_score:.2f})"
        
        # Stress level filter with graduated response
        if market_conditions.market_stress_level > 0.85:
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Would normally reject due to extreme stress ({market_conditions.market_stress_level:.2f})")
            else:
                return False, f"extreme_market_stress (level: {market_conditions.market_stress_level:.2f})"
        
        # Volatile regime filter
        if market_conditions.market_regime == 'volatile':
            if CONFIG.testing_mode:
                log_info(f"[TESTING MODE] Ignoring volatile market regime")
            else:
                return False, f"volatile_market_regime (stress: {market_conditions.market_stress_level:.2f})"
        
        trading_reason = f"conditions_favorable (regime: {market_conditions.market_regime})"
        if CONFIG.testing_mode:
            trading_reason = f"[TESTING MODE] {trading_reason}"
        
        return True, trading_reason


# Update the existing NewsQualityFilter and other classes remain the same...
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
        
        # Create a copy to avoid modifying original
        filtered_df = news_df.copy()
        
        # 1. Timezone-aware freshness filter
        filtered_df = self._filter_stale_news(filtered_df)
        
        # 2. Deduplicate headlines efficiently
        filtered_df = self._deduplicate_headlines(filtered_df)
        
        # 3. Content quality filters
        filtered_df = self._filter_content_quality(filtered_df)
        
        # 4. Priority scoring
        filtered_df = self._add_priority_scoring(filtered_df)
        
        return filtered_df
    
    def _filter_stale_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter stale news with timezone-aware comparison."""
        if 'publishedDate' not in news_df.columns:
            return news_df
        
        try:
            # Ensure timezone awareness
            now = datetime.now(timezone.utc)
            
            # In testing mode, allow older news
            if CONFIG.testing_mode:
                cutoff_time = now - timedelta(hours=24)  # 24 hours in testing
                log_debug("[TESTING MODE] Extended news freshness to 24 hours")
            else:
                cutoff_time = now - timedelta(hours=2)   # 2 hours in production
            
            # Convert published dates to UTC if not already
            mask = news_df['publishedDate'] > cutoff_time
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error filtering stale news: {e}")
            return news_df
    
    def _deduplicate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Efficient headline deduplication using vectorized operations."""
        if 'title' not in news_df.columns:
            return news_df
        
        try:
            # Simple deduplication using pandas
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # More sophisticated similarity check for remaining articles
            if len(news_df) > 100:  # Only for larger datasets to avoid performance issues
                return news_df
            
            # For smaller datasets, do similarity checking
            filtered_indices = []
            seen_word_sets = []
            
            for idx, row in news_df.iterrows():
                title = str(row['title']).lower().strip()
                title_words = set(title.split())
                
                is_duplicate = False
                for seen_words in seen_word_sets:
                    if title_words and seen_words:
                        overlap = len(title_words & seen_words) / len(title_words | seen_words)
                        if overlap > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_indices.append(idx)
                    seen_word_sets.append(title_words)
                    
                    # Limit memory usage
                    if len(seen_word_sets) > self._headline_cache_limit:
                        seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
            
            return news_df.loc[filtered_indices]
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _filter_content_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized content quality filtering."""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Title length filter
            if 'title' in news_df.columns:
                mask &= news_df['title'].str.len() >= 20
                
                # Suspicious content filter
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
            # Initialize priority score
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Official announcement keywords
            official_keywords = [
                'announces', 'reports', 'declares', 'files', 'receives',
                'completes', 'signs', 'launches', 'enters into', 'appoints'
            ]
            
            # Vectorized keyword matching
            for keyword in official_keywords:
                mask = news_df['title'].str.lower().str.contains(keyword, na=False)
                news_df.loc[mask, 'priority_score'] += 0.1
            
            # Sort by priority and timestamp
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            news_df = news_df.sort_values(sort_columns, ascending=[False, False])
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df