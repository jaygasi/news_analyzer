"""
Momentum calculation utilities
"""
import pandas as pd
from typing import Dict
from utils.simple_logger import log_debug


class MomentumCalculator:
    """Calculate various momentum indicators and scores"""
    
    @staticmethod
    def calculate_momentum_score(rsi: float, trend: str, mas: Dict[str, float], current_price: float) -> float:
        """Calculate comprehensive momentum score"""
        try:
            # RSI component (-1 to 1 scale)
            rsi_normalized = (rsi - 50) / 50
            rsi_score = max(-1, min(1, rsi_normalized))
            
            # Trend component
            trend_scores = {'uptrend': 0.5, 'sideways': 0, 'downtrend': -0.5}
            trend_score = trend_scores.get(trend, 0)
            
            # Moving average alignment score
            ma_score = MomentumCalculator._calculate_ma_alignment_score(mas, current_price)
            
            # Combine all components
            momentum = (rsi_score * 0.4 + trend_score * 0.3 + ma_score * 0.3)
            return max(-1, min(1, momentum))
            
        except Exception as e:
            log_debug(f"Error calculating momentum score: {e}")
            return 0.0
    
    @staticmethod
    def _calculate_ma_alignment_score(mas: Dict[str, float], current_price: float) -> float:
        """Calculate moving average alignment score"""
        try:
            ma_score = 0
            sma_20 = mas.get('sma_20', current_price)
            sma_5 = mas.get('sma_5', current_price)
            ema_12 = mas.get('ema_12', current_price)
            ema_26 = mas.get('ema_26', current_price)
            
            # Price vs MA
            if current_price > sma_20:
                ma_score += 0.3
            elif current_price < sma_20:
                ma_score -= 0.3
            
            # Short vs long MA
            if sma_5 > sma_20:
                ma_score += 0.2
            elif sma_5 < sma_20:
                ma_score -= 0.2
            
            # EMA alignment
            if ema_12 > ema_26:
                ma_score += 0.2
            elif ema_12 < ema_26:
                ma_score -= 0.2
            
            return ma_score
            
        except Exception as e:
            log_debug(f"Error calculating MA alignment score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_price_momentum(prices: pd.Series, periods: list = [5, 10, 20]) -> Dict[str, float]:
        """Calculate price momentum over different periods"""
        try:
            momentum = {}
            current_price = prices.iloc[-1]
            
            for period in periods:
                if len(prices) > period:
                    past_price = prices.iloc[-period-1]
                    pct_change = (current_price - past_price) / past_price
                    momentum[f'momentum_{period}d'] = pct_change
                else:
                    momentum[f'momentum_{period}d'] = 0.0
            
            return momentum
            
        except Exception as e:
            log_debug(f"Error calculating price momentum: {e}")
            return {f'momentum_{p}d': 0.0 for p in periods}
    
    @staticmethod
    def calculate_volume_momentum(volumes: pd.Series, period: int = 20) -> float:
        """Calculate volume momentum indicator"""
        try:
            if len(volumes) < period + 1:
                return 1.0
            
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.rolling(period).mean().iloc[-1]
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                return min(volume_ratio / 2.0, 2.0)  # Cap at 2x
            
            return 1.0
            
        except Exception as e:
            log_debug(f"Error calculating volume momentum: {e}")
            return 1.0
    
    @staticmethod
    def detect_momentum_divergence(prices: pd.Series, rsi_values: pd.Series) -> str:
        """Detect momentum divergence patterns"""
        try:
            if len(prices) < 10 or len(rsi_values) < 10:
                return 'none'
            
            # Check last 10 periods for divergence
            recent_prices = prices.tail(10)
            recent_rsi = rsi_values.tail(10)
            
            # Price trend
            price_trend = 'up' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'down'
            
            # RSI trend
            rsi_trend = 'up' if recent_rsi.iloc[-1] > recent_rsi.iloc[0] else 'down'
            
            # Detect divergence
            if price_trend == 'up' and rsi_trend == 'down':
                return 'bearish_divergence'
            elif price_trend == 'down' and rsi_trend == 'up':
                return 'bullish_divergence'
            else:
                return 'none'
                
        except Exception as e:
            log_debug(f"Error detecting momentum divergence: {e}")
            return 'none'
    
    @staticmethod
    def calculate_momentum_quality(momentum_score: float, volume_ratio: float, trend_strength: float) -> float:
        """Calculate overall momentum quality score"""
        try:
            # Absolute momentum strength
            momentum_strength = abs(momentum_score)
            
            # Volume confirmation (higher volume = better quality)
            volume_factor = min(volume_ratio / 1.5, 1.0)  # Normalize at 1.5x average
            
            # Trend consistency (stronger trends = better quality)
            trend_factor = min(trend_strength, 1.0)
            
            # Weighted combination
            quality = (
                momentum_strength * 0.5 +
                volume_factor * 0.3 +
                trend_factor * 0.2
            )
            
            return min(quality, 1.0)
            
        except Exception as e:
            log_debug(f"Error calculating momentum quality: {e}")
            return 0.5