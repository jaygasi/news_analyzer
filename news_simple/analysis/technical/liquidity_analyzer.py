"""
Liquidity analysis utilities
"""
import pandas as pd
from config import CONFIG
from utils.simple_logger import log_debug


class LiquidityAnalyzer:
    """Analyze market liquidity and trading conditions"""
    
    @staticmethod
    def calculate_liquidity_score(bid_ask_spread: float, current_volume: float, avg_volume: float) -> float:
        """Calculate comprehensive liquidity score"""
        try:
            # Spread component (tighter spreads = higher liquidity)
            spread_score = max(0, 1 - (bid_ask_spread / 0.05))  # 5% spread = 0 score
            
            # Volume component (relative to average)
            vol_ratio = current_volume / max(avg_volume, 1) if avg_volume > 0 else 1.0
            vol_score = min(vol_ratio / 2.0, 1.0)  # Cap at 2x average
            
            # Minimum volume threshold
            min_vol_score = 1.0 if current_volume >= CONFIG.min_volume else 0.5
            
            # Weighted combination
            liquidity = (spread_score * 0.4 + vol_score * 0.4 + min_vol_score * 0.2)
            return max(0.0, min(1.0, liquidity))
            
        except Exception as e:
            log_debug(f"Error calculating liquidity score: {e}")
            return 0.5
    
    @staticmethod
    def calculate_bid_ask_spread(bid_price: float, ask_price: float) -> float:
        """Calculate bid-ask spread percentage"""
        try:
            if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
                return 0.05  # Default 5% assumption
            
            spread = (ask_price - bid_price) / bid_price
            return max(0.001, min(0.20, spread))  # Cap between 0.1% and 20%
            
        except Exception as e:
            log_debug(f"Error calculating bid-ask spread: {e}")
            return 0.05
    
    @staticmethod
    def analyze_volume_profile(volumes: pd.Series, period: int = 20) -> dict:
        """Analyze volume profile and characteristics"""
        try:
            if len(volumes) < period:
                current_vol = volumes.iloc[-1] if len(volumes) > 0 else 0
                return {
                    'current_volume': current_vol,
                    'average_volume': current_vol,
                    'volume_ratio': 1.0,
                    'volume_trend': 'stable',
                    'volume_score': 0.5
                }
            
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.rolling(period).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend
            recent_avg = volumes.tail(5).mean()
            older_avg = volumes.tail(period).head(5).mean()
            
            if recent_avg > older_avg * 1.2:
                volume_trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'stable'
            
            # Volume score
            volume_score = min(volume_ratio / 2.0, 1.0)
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_score': volume_score
            }
            
        except Exception as e:
            log_debug(f"Error analyzing volume profile: {e}")
            return {
                'current_volume': 0,
                'average_volume': 0,
                'volume_ratio': 1.0,
                'volume_trend': 'stable',
                'volume_score': 0.5
            }
    
    @staticmethod
    def calculate_market_impact_estimate(position_size: float, avg_volume: float, volatility: float) -> float:
        """Estimate market impact of a trade"""
        try:
            if avg_volume <= 0:
                return 0.05  # 5% impact assumption
            
            # Simple market impact model
            volume_fraction = position_size / (avg_volume * 1.0)  # Assuming price = avg_volume for simplicity
            
            # Impact increases with square root of volume fraction
            base_impact = (volume_fraction ** 0.5) * 0.01  # 1% per sqrt of volume fraction
            
            # Adjust for volatility
            volatility_adjustment = 1.0 + volatility
            
            market_impact = base_impact * volatility_adjustment
            
            return min(market_impact, 0.20)  # Cap at 20%
            
        except Exception as e:
            log_debug(f"Error calculating market impact: {e}")
            return 0.05
    
    @staticmethod
    def assess_execution_quality(current_volume: float, avg_volume: float, 
                               bid_ask_spread: float, volatility: float) -> dict:
        """Assess expected execution quality"""
        try:
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Execution speed score
            if volume_ratio > 1.5:
                speed_score = 0.9  # High volume = fast execution
            elif volume_ratio > 1.0:
                speed_score = 0.7
            else:
                speed_score = 0.5
            
            # Cost score (lower spread = better)
            if bid_ask_spread < 0.01:
                cost_score = 0.9
            elif bid_ask_spread < 0.03:
                cost_score = 0.7
            else:
                cost_score = 0.5
            
            # Stability score (lower volatility = more stable)
            if volatility < 0.3:
                stability_score = 0.9
            elif volatility < 0.6:
                stability_score = 0.7
            else:
                stability_score = 0.5
            
            overall_quality = (speed_score + cost_score + stability_score) / 3.0
            
            return {
                'speed_score': speed_score,
                'cost_score': cost_score,
                'stability_score': stability_score,
                'overall_quality': overall_quality,
                'recommendation': LiquidityAnalyzer._get_execution_recommendation(overall_quality)
            }
            
        except Exception as e:
            log_debug(f"Error assessing execution quality: {e}")
            return {
                'speed_score': 0.5,
                'cost_score': 0.5,
                'stability_score': 0.5,
                'overall_quality': 0.5,
                'recommendation': 'caution'
            }
    
    @staticmethod
    def _get_execution_recommendation(quality_score: float) -> str:
        """Get execution recommendation based on quality score"""
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.7:
            return 'good'
        elif quality_score >= 0.6:
            return 'fair'
        else:
            return 'caution'