"""
Market condition analysis and filtering
"""
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Tuple, Optional
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


class MarketConditionFilter:
    """Market condition analysis and filtering"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with market data loader"""
        self.fmp_loader = fmp_loader
        self._market_data_cache = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=8)
        
        # Pre-compute time constants
        self._market_open = dt_time(9, 30)
        self._market_close = dt_time(16, 0)
        self._et_offset = timedelta(hours=-5)
    
    @lru_cache(maxsize=64)
    def _get_market_data_cached(self, cache_key: str) -> Tuple[Optional[float], Optional[float]]:
        """Cached market data retrieval"""
        try:
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            if market_data is None or market_data.empty:
                return None, None
                
            spy_price = None
            vix_level = None
            
            for _, row in market_data.iterrows():
                symbol = row.get('symbol', '')
                if symbol == 'SPY':
                    spy_price = float(row.get('lastSalePrice', 0))
                elif symbol == 'VIX':
                    vix_level = float(row.get('lastSalePrice', 0))
                
                if spy_price is not None and vix_level is not None:
                    break
            
            return spy_price, vix_level
            
        except Exception as e:
            log_warning(f"Error fetching market data: {e}")
            return None, None
    
    def _get_market_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get market data with caching"""
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
        """Get current market conditions"""
        now = datetime.now(timezone.utc)
        
        # Market hours and time scoring
        is_market_hours, time_score = self._calculate_market_hours_and_time_score(now)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Calculate stress level and regime
        stress_level = self._calculate_stress_level(vix_level)
        regime = self._classify_market_regime(vix_level)
        
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
        """Calculate market hours and time score"""
        et_time = (now + self._et_offset).time()
        
        # Market hours check
        is_market_hours = (
            self._market_open <= et_time <= self._market_close and
            now.weekday() < 5
        )
        
        # Testing mode override
        if CONFIG.testing_mode:
            log_debug(f"[TESTING MODE] Overriding market hours check (actual: {et_time.strftime('%H:%M')} ET)")
            is_market_hours = True
        
        # Time scoring
        time_score = 0.75 if CONFIG.testing_mode else self._calculate_time_score(et_time)
        
        return is_market_hours, time_score
    
    def _calculate_time_score(self, current_time: dt_time) -> float:
        """Calculate time-of-day score"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Quick return for off-hours
        if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            return 0.1
        
        # Convert to minutes from market open
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring
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
    
    def _calculate_stress_level(self, vix_level: Optional[float]) -> float:
        """Calculate market stress level from VIX"""
        if vix_level is None:
            return 0.25
        
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
    
    def _classify_market_regime(self, vix_level: Optional[float]) -> str:
        """Classify market regime based on VIX"""
        if vix_level is None:
            return 'neutral'
        
        if vix_level > 32:
            return 'volatile'
        elif vix_level > 26:
            return 'bear'
        elif vix_level < 15:
            return 'bull'
        else:
            return 'neutral'
    
    def _apply_testing_mode_adjustments(self, stress_level: float, regime: str) -> Tuple[float, str]:
        """Apply testing mode adjustments"""
        if CONFIG.testing_mode:
            if stress_level > 0.75:
                stress_level = 0.55
                log_debug(f"[TESTING MODE] Reduced stress level to {stress_level}")
            
            if regime == 'volatile':
                regime = 'neutral'
                log_debug(f"[TESTING MODE] Changed regime to {regime}")
        
        return stress_level, regime
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine if trading conditions are favorable"""
        
        if CONFIG.testing_mode:
            log_debug("[TESTING MODE] Market condition checks with testing overrides")
        
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