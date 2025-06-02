"""
Optimized trend analysis with improved pattern recognition and performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from utils.simple_logger import log_debug


class TrendAnalyzer:
    """High-performance trend analysis with enhanced pattern recognition."""
    
    # Pre-computed trend thresholds for performance
    TREND_THRESHOLDS = {
        'strong_bullish': 0.75,
        'bullish': 0.60,
        'neutral': 0.40,
        'bearish': 0.25,
        'strong_bearish': 0.10
    }
    
    @staticmethod
    def detect_trend(prices: pd.Series, mas: Dict[str, float]) -> str:
        """Optimized trend detection using multiple criteria with vectorized operations."""
        try:
            if len(prices) < 5:
                return 'sideways'
            
            current_price = float(prices.iloc[-1])
            
            # Safely extract moving averages with fallbacks
            sma_5 = float(mas.get('sma_5', current_price))
            sma_10 = float(mas.get('sma_10', current_price))
            sma_20 = float(mas.get('sma_20', current_price))
            
            # Validate moving averages
            valid_mas = all(not pd.isna(ma) and ma > 0 for ma in [sma_5, sma_10, sma_20])
            if not valid_mas:
                sma_5 = sma_10 = sma_20 = current_price
            
            # Calculate trend components efficiently
            trend_score = TrendAnalyzer._calculate_trend_score(
                current_price, sma_5, sma_10, sma_20, prices
            )
            
            # Determine trend based on score
            if trend_score >= TrendAnalyzer.TREND_THRESHOLDS['strong_bullish']:
                return 'strong_uptrend'
            elif trend_score >= TrendAnalyzer.TREND_THRESHOLDS['bullish']:
                return 'uptrend'
            elif trend_score <= TrendAnalyzer.TREND_THRESHOLDS['strong_bearish']:
                return 'strong_downtrend'
            elif trend_score <= TrendAnalyzer.TREND_THRESHOLDS['bearish']:
                return 'downtrend'
            else:
                return 'sideways'
                
        except Exception as e:
            log_debug(f"Error detecting trend: {e}")
            return 'sideways'
    
    @staticmethod
    def _calculate_trend_score(current_price: float, sma_5: float, sma_10: float, 
                             sma_20: float, prices: pd.Series) -> float:
        """Calculate comprehensive trend score (0-1 scale)."""
        score_components = []
        
        # Moving average alignment (40% weight)
        ma_alignment = TrendAnalyzer._calculate_ma_alignment_score(
            current_price, sma_5, sma_10, sma_20
        )
        score_components.append((ma_alignment, 0.4))
        
        # Price momentum (30% weight)
        momentum_score = TrendAnalyzer._calculate_momentum_score(prices)
        score_components.append((momentum_score, 0.3))
        
        # Price position relative to MA (30% weight)
        position_score = TrendAnalyzer._calculate_position_score(current_price, sma_20)
        score_components.append((position_score, 0.3))
        
        # Weighted average
        weighted_score = sum(score * weight for score, weight in score_components)
        return max(0.0, min(1.0, weighted_score))
    
    @staticmethod
    def _calculate_ma_alignment_score(current_price: float, sma_5: float, 
                                    sma_10: float, sma_20: float) -> float:
        """Calculate moving average alignment score."""
        # Perfect bullish alignment: price > sma_5 > sma_10 > sma_20
        conditions = [
            current_price > sma_5,
            sma_5 > sma_10,
            sma_10 > sma_20,
            current_price > sma_20
        ]
        
        bullish_score = sum(conditions) / len(conditions)
        
        # Perfect bearish alignment: price < sma_5 < sma_10 < sma_20
        bearish_conditions = [
            current_price < sma_5,
            sma_5 < sma_10,
            sma_10 < sma_20,
            current_price < sma_20
        ]
        
        bearish_score = sum(bearish_conditions) / len(bearish_conditions)
        
        # Return score (1.0 = perfect bullish, 0.0 = perfect bearish, 0.5 = neutral)
        if bullish_score > bearish_score:
            return 0.5 + (bullish_score * 0.5)
        else:
            return 0.5 - (bearish_score * 0.5)
    
    @staticmethod
    def _calculate_momentum_score(prices: pd.Series) -> float:
        """Calculate price momentum score using multiple timeframes."""
        try:
            if len(prices) < 10:
                return 0.5
            
            current_price = prices.iloc[-1]
            momentum_scores = []
            
            # Multiple timeframe momentum
            timeframes = [3, 5, 10]
            for period in timeframes:
                if len(prices) > period:
                    past_price = prices.iloc[-period]
                    if past_price > 0:
                        momentum = (current_price - past_price) / past_price
                        # Normalize momentum to 0-1 scale (assuming ±10% is significant)
                        normalized = 0.5 + (momentum / 0.2)  # ±20% maps to 0-1
                        momentum_scores.append(max(0.0, min(1.0, normalized)))
            
            return np.mean(momentum_scores) if momentum_scores else 0.5
            
        except Exception:
            return 0.5
    
    @staticmethod
    def _calculate_position_score(current_price: float, sma_20: float) -> float:
        """Calculate price position relative to key moving average."""
        if sma_20 <= 0:
            return 0.5
        
        # Calculate relative position
        relative_position = (current_price - sma_20) / sma_20
        
        # Normalize to 0-1 scale (±10% from MA maps to 0-1)
        normalized = 0.5 + (relative_position / 0.2)
        return max(0.0, min(1.0, normalized))
    
    @staticmethod
    def calculate_trend_strength(prices: pd.Series, trend: str) -> float:
        """Enhanced trend strength calculation with linear regression."""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Use recent prices for trend strength
            recent_prices = prices.tail(10).values
            x_values = np.arange(len(recent_prices))
            
            # Linear regression for trend slope
            try:
                slope, intercept = np.polyfit(x_values, recent_prices, 1)
                
                # Normalize slope relative to average price
                avg_price = np.mean(recent_prices)
                if avg_price > 0:
                    normalized_slope = abs(slope) / avg_price
                    strength = min(normalized_slope * 10, 1.0)  # Scale factor
                else:
                    strength = 0.5
                
            except np.linalg.LinAlgError:
                # Fallback to simple range-based calculation
                price_range = np.max(recent_prices) - np.min(recent_prices)
                avg_price = np.mean(recent_prices)
                strength = (price_range / avg_price) if avg_price > 0 else 0.5
            
            return max(0.1, min(strength, 1.0))
            
        except Exception as e:
            log_debug(f"Error calculating trend strength: {e}")
            return 0.5
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, current_price: float, 
                              lookback: int = 30) -> Dict[str, Optional[float]]:
        """Enhanced support/resistance detection with clustering."""
        try:
            if len(df) < lookback or not all(col in df.columns for col in ['high', 'low']):
                return {
                    'support': current_price * 0.98,
                    'resistance': current_price * 1.02
                }
            
            recent_data = df.tail(lookback)
            
            # Find support and resistance using improved clustering
            support = TrendAnalyzer._find_support_level_enhanced(recent_data['low'], current_price)
            resistance = TrendAnalyzer._find_resistance_level_enhanced(recent_data['high'], current_price)
            
            return {
                'support': support or current_price * 0.98,
                'resistance': resistance or current_price * 1.02,
                'support_strength': TrendAnalyzer._calculate_level_strength(recent_data['low'], support) if support else 0.5,
                'resistance_strength': TrendAnalyzer._calculate_level_strength(recent_data['high'], resistance) if resistance else 0.5
            }
            
        except Exception as e:
            log_debug(f"Error finding support/resistance: {e}")
            return {
                'support': current_price * 0.98,
                'resistance': current_price * 1.02
            }
    
    @staticmethod
    def _find_support_level_enhanced(lows: pd.Series, current_price: float) -> Optional[float]:
        """Enhanced support level detection with price clustering."""
        try:
            # Convert to numpy for faster processing
            low_values = lows.values
            
            # Filter lows below current price
            candidate_supports = low_values[low_values < current_price * 0.99]
            
            if len(candidate_supports) < 2:
                return None
            
            # Cluster nearby support levels
            support_clusters = TrendAnalyzer._cluster_price_levels(candidate_supports, tolerance=0.02)
            
            # Find the strongest cluster (most touches) closest to current price
            best_support = None
            best_score = 0
            
            for level, count in support_clusters.items():
                # Score based on touch count and proximity to current price
                proximity_score = 1.0 - (abs(current_price - level) / current_price)
                total_score = count * proximity_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_support = level
            
            return best_support
            
        except Exception:
            return None
    
    @staticmethod
    def _find_resistance_level_enhanced(highs: pd.Series, current_price: float) -> Optional[float]:
        """Enhanced resistance level detection with price clustering."""
        try:
            high_values = highs.values
            
            # Filter highs above current price
            candidate_resistances = high_values[high_values > current_price * 1.01]
            
            if len(candidate_resistances) < 2:
                return None
            
            # Cluster nearby resistance levels
            resistance_clusters = TrendAnalyzer._cluster_price_levels(candidate_resistances, tolerance=0.02)
            
            # Find the strongest cluster closest to current price
            best_resistance = None
            best_score = 0
            
            for level, count in resistance_clusters.items():
                proximity_score = 1.0 - (abs(level - current_price) / current_price)
                total_score = count * proximity_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_resistance = level
            
            return best_resistance
            
        except Exception:
            return None
    
    @staticmethod
    def _cluster_price_levels(prices: np.ndarray, tolerance: float = 0.02) -> Dict[float, int]:
        """Cluster similar price levels together."""
        clusters = {}
        
        for price in prices:
            # Find if price belongs to existing cluster
            cluster_found = False
            for cluster_center in list(clusters.keys()):
                if abs(price - cluster_center) / cluster_center <= tolerance:
                    clusters[cluster_center] += 1
                    cluster_found = True
                    break
            
            # Create new cluster if no match found
            if not cluster_found:
                clusters[price] = 1
        
        return clusters
    
    @staticmethod
    def _calculate_level_strength(prices: pd.Series, level: Optional[float]) -> float:
        """Calculate the strength of a support/resistance level."""
        if level is None:
            return 0.0
        
        try:
            # Count how many times price touched this level
            tolerance = 0.02  # 2% tolerance
            touches = sum(1 for price in prices if abs(price - level) / level <= tolerance)
            
            # Normalize touch count to 0-1 scale
            return min(touches / 5.0, 1.0)  # 5+ touches = maximum strength
            
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_support_resistance_score(prices: pd.Series, current_price: float) -> float:
        """Calculate overall support/resistance environment score."""
        try:
            if len(prices) < 10:
                return 0.5
            
            recent_prices = prices.tail(20)
            
            # Calculate price level density around current price
            tolerance = 0.02
            nearby_prices = sum(
                1 for price in recent_prices 
                if abs(price - current_price) / current_price <= tolerance
            )
            
            # Normalize density score
            density_score = min(nearby_prices / len(recent_prices), 1.0)
            
            # Calculate volatility score (lower volatility = stronger levels)
            volatility = recent_prices.pct_change().std()
            volatility_score = max(0.0, 1.0 - (volatility * 10))  # Scale volatility
            
            # Combine scores
            final_score = (density_score * 0.6) + (volatility_score * 0.4)
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            log_debug(f"Error calculating support/resistance score: {e}")
            return 0.5