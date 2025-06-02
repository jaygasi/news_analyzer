"""
Trend detection and analysis utilities
"""
import pandas as pd
from typing import Dict, Optional
from utils.simple_logger import log_debug


class TrendAnalyzer:
    """Analyze price trends and patterns"""
    
    @staticmethod
    def detect_trend(prices: pd.Series, mas: Dict[str, float]) -> str:
        """Detect price trend using multiple criteria"""
        try:
            if len(prices) < 5:
                return 'sideways'
            
            current_price = float(prices.iloc[-1])
            
            # Get moving averages with fallbacks
            sma_5 = mas.get('sma_5', current_price)
            sma_10 = mas.get('sma_10', current_price)
            sma_20 = mas.get('sma_20', current_price)
            
            # Ensure all values are valid floats
            for ma_name, ma_value in [('sma_5', sma_5), ('sma_10', sma_10), ('sma_20', sma_20)]:
                if ma_value is None or pd.isna(ma_value):
                    mas[ma_name] = current_price
            
            sma_5 = float(mas.get('sma_5', current_price))
            sma_10 = float(mas.get('sma_10', current_price))
            sma_20 = float(mas.get('sma_20', current_price))
            
            # Multiple criteria for trend detection
            ma_alignment = sma_5 > sma_10 > sma_20  # Bullish alignment
            ma_alignment_bear = sma_5 < sma_10 < sma_20  # Bearish alignment
            
            price_above_ma = current_price > sma_20
            price_below_ma = current_price < sma_20
            
            # Recent price momentum
            if len(prices) >= 5:
                recent_change = (current_price - prices.iloc[-5]) / prices.iloc[-5]
                momentum_up = recent_change > 0.02  # 2% up
                momentum_down = recent_change < -0.02  # 2% down
            else:
                momentum_up = momentum_down = False
            
            # Combine signals
            bullish_signals = sum([ma_alignment, price_above_ma, momentum_up])
            bearish_signals = sum([ma_alignment_bear, price_below_ma, momentum_down])
            
            if bullish_signals >= 2:
                return 'uptrend'
            elif bearish_signals >= 2:
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            log_debug(f"Error detecting trend: {e}")
            return 'sideways'
    
    @staticmethod
    def calculate_trend_strength(prices: pd.Series, trend: str) -> float:
        """Calculate the strength of the current trend"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Calculate slope of price movement
            recent_prices = prices.tail(10)
            x = range(len(recent_prices))
            y = recent_prices.values
            
            # Simple linear regression slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 == 0:
                return 0.5
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Normalize slope to strength score
            avg_price = sum_y / n
            normalized_slope = abs(slope) / avg_price if avg_price > 0 else 0
            
            strength = min(normalized_slope * 100, 1.0)  # Cap at 1.0
            
            return strength
            
        except Exception as e:
            log_debug(f"Error calculating trend strength: {e}")
            return 0.5
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, current_price: float, lookback: int = 30) -> Dict[str, float]:
        """Find support and resistance levels"""
        try:
            if len(df) < lookback or 'high' not in df.columns or 'low' not in df.columns:
                return {
                    'support': current_price * 0.98,
                    'resistance': current_price * 1.02
                }
            
            recent_data = df.tail(lookback)
            
            # Find support from recent lows
            support = TrendAnalyzer._find_support_level(recent_data['low'], current_price)
            
            # Find resistance from recent highs
            resistance = TrendAnalyzer._find_resistance_level(recent_data['high'], current_price)
            
            return {
                'support': support or current_price * 0.98,
                'resistance': resistance or current_price * 1.02
            }
            
        except Exception as e:
            log_debug(f"Error finding support/resistance: {e}")
            return {
                'support': current_price * 0.98,
                'resistance': current_price * 1.02
            }
    
    @staticmethod
    def _find_support_level(lows: pd.Series, current_price: float) -> Optional[float]:
        """Find nearest significant support level"""
        try:
            support_candidates = []
            for low in lows:
                touches = sum(abs(lows - low) / low < 0.02)  # Within 2%
                if touches >= 2:  # At least 2 touches
                    support_candidates.append(low)
            
            if support_candidates:
                valid_supports = [s for s in support_candidates if s < current_price * 0.98]
                return max(valid_supports) if valid_supports else None
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def _find_resistance_level(highs: pd.Series, current_price: float) -> Optional[float]:
        """Find nearest significant resistance level"""
        try:
            resistance_candidates = []
            for high in highs:
                touches = sum(abs(highs - high) / high < 0.02)  # Within 2%
                if touches >= 2:  # At least 2 touches
                    resistance_candidates.append(high)
            
            if resistance_candidates:
                valid_resistances = [r for r in resistance_candidates if r > current_price * 1.02]
                return min(valid_resistances) if valid_resistances else None
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def calculate_support_resistance_score(prices: pd.Series, current_price: float) -> float:
        """Calculate support/resistance strength score"""
        try:
            if len(prices) < 10:
                return 0.5
            
            recent_prices = prices.tail(20)
            
            # Count nearby price levels
            tolerance = 0.02
            nearby_count = sum(
                abs(price - current_price) / current_price <= tolerance
                for price in recent_prices
            )
            
            # Normalize the score
            level_strength = nearby_count / len(recent_prices)
            return min(level_strength * 2, 1.0)
            
        except Exception as e:
            log_debug(f"Error calculating support/resistance score: {e}")
            return 0.5