"""
Simplified technical analysis for stock direction prediction
Python 3.13.3 compatible
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from data_loaders.base_fmp_loader import BaseFMPLoader
from utils.simple_logger import log_debug, log_error
from config import Config


@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    indicators: Dict[str, float]
    reasoning: str


class TechnicalAnalyzer:
    """Simplified technical analysis using key indicators"""
    
    def __init__(self, fmp_loader: BaseFMPLoader) -> None:
        """Initialize technical analyzer"""
        self.fmp_loader = fmp_loader
        self.lookback_days = Config.TECHNICAL_LOOKBACK_DAYS  # Days of historical data to analyze
    
    def analyze_ticker(self, ticker: str) -> Optional[TechnicalSignal]:
        """Perform technical analysis on ticker"""
        try:
            # Get historical data
            historical_data = self._get_historical_data(ticker)
            if historical_data is None or len(historical_data) < 20:
                log_debug(f"Insufficient historical data for {ticker}")
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(historical_data)
            
            # Generate signal
            signal = self._generate_signal(indicators)
            
            return signal
            
        except Exception as e:
            log_error(f"Technical analysis failed for {ticker}: {e}")
            return None
    
    def _get_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            data = self.fmp_loader.make_request(f"historical-price-full/{ticker}")
            
            if not data or 'historical' not in data:
                return None
            
            df = pd.DataFrame(data['historical'])
            
            if df.empty:
                return None
            
            # Convert date and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Convert price columns to float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Take recent data
            return df.tail(self.lookback_days)
            
        except Exception as e:
            log_error(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key technical indicators"""
        indicators = {}
        
        try:
            close_prices = df['close']
            high_prices = df['high']
            low_prices = df['low']
            volumes = df['volume']
            
            # Moving Averages
            indicators['sma_10'] = close_prices.rolling(10).mean().iloc[-1]
            indicators['sma_20'] = close_prices.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = close_prices.rolling(50).mean().iloc[-1] if len(df) >= 50 else indicators['sma_20']
            
            # Current price
            current_price = close_prices.iloc[-1]
            indicators['current_price'] = current_price
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(close_prices)
            
            # MACD
            macd_line, signal_line = self._calculate_macd(close_prices)
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Volume analysis
            avg_volume = volumes.rolling(20).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price momentum
            indicators['price_change_1d'] = (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2] if len(close_prices) >= 2 else 0
            indicators['price_change_5d'] = (close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6] if len(close_prices) >= 6 else 0
            indicators['price_change_20d'] = (close_prices.iloc[-1] - close_prices.iloc[-21]) / close_prices.iloc[-21] if len(close_prices) >= 21 else 0
            
            # Support and Resistance
            recent_high = high_prices.tail(20).max()
            recent_low = low_prices.tail(20).min()
            indicators['distance_to_high'] = (recent_high - current_price) / current_price
            indicators['distance_to_low'] = (current_price - recent_low) / current_price
            
            return indicators
            
        except Exception as e:
            log_error(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            if period is None:
                period = Config.TECHNICAL_RSI_PERIOD

            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = None, slow: int = None, signal: int = None) -> Tuple[float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if fast is None:
                fast = Config.TECHNICAL_MACD_FAST
            if slow is None:
                slow = Config.TECHNICAL_MACD_SLOW
            if signal is None:
                signal = Config.TECHNICAL_MACD_SIGNAL
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])
            
        except Exception:
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = None, std_dev: float = None) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        try:
            if period is None:
                period = Config.TECHNICAL_BOLLINGER_PERIOD
            if std_dev is None:
                std_dev = Config.TECHNICAL_BOLLINGER_STD_DEV
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return float(upper_band.iloc[-1]), float(sma.iloc[-1]), float(lower_band.iloc[-1])
            
        except Exception:
            current_price = float(prices.iloc[-1])
            return current_price * 1.02, current_price, current_price * 0.98
    
    def _generate_signal(self, indicators: Dict[str, float]) -> TechnicalSignal:
        """Generate trading signal from indicators"""
        if not indicators:
            return TechnicalSignal(
                direction='NEUTRAL',
                strength=0.0,
                indicators={},
                reasoning="No indicators available"
            )
        
        signals = []
        reasoning_parts = []
        
        # Moving Average Signal
        ma_signal = self._evaluate_moving_averages(indicators)
        signals.append(ma_signal)
        reasoning_parts.append(f"MA trend: {ma_signal['direction']}")
        
        # RSI Signal
        rsi_signal = self._evaluate_rsi(indicators)
        signals.append(rsi_signal)
        reasoning_parts.append(f"RSI: {indicators.get('rsi', 50):.1f}")
        
        # MACD Signal
        macd_signal = self._evaluate_macd(indicators)
        signals.append(macd_signal)
        reasoning_parts.append(f"MACD: {'bullish' if macd_signal['score'] > 0 else 'bearish'}")
        
        # Bollinger Bands Signal
        bb_signal = self._evaluate_bollinger_bands(indicators)
        signals.append(bb_signal)
        reasoning_parts.append(f"BB position: {indicators.get('bb_position', 0.5):.2f}")
        
        # Volume Confirmation
        volume_signal = self._evaluate_volume(indicators)
        signals.append(volume_signal)
        
        # Combine signals
        total_score = sum(signal['score'] * signal['weight'] for signal in signals)
        total_weight = sum(signal['weight'] for signal in signals)
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        # Determine direction and strength
        if final_score > 0.3:
            direction = 'BUY'
            strength = min(final_score, 1.0)
        elif final_score < -0.3:
            direction = 'SELL'
            strength = min(abs(final_score), 1.0)
        else:
            direction = 'NEUTRAL'
            strength = 1.0 - abs(final_score)
        
        reasoning = "; ".join(reasoning_parts)
        
        return TechnicalSignal(
            direction=direction,
            strength=strength,
            indicators=indicators,
            reasoning=reasoning
        )
    
    def _evaluate_moving_averages(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate moving average trend"""
        current = indicators.get('current_price', 0)
        sma_10 = indicators.get('sma_10', current)
        sma_20 = indicators.get('sma_20', current)
        sma_50 = indicators.get('sma_50', current)
        
        score = 0.0
        
        # Price vs MAs
        if current > sma_10:
            score += 0.3
        if current > sma_20:
            score += 0.4
        if current > sma_50:
            score += 0.3
        
        if current < sma_10:
            score -= 0.3
        if current < sma_20:
            score -= 0.4
        if current < sma_50:
            score -= 0.3
        
        # MA alignment
        if sma_10 > sma_20:
            score += 0.2
        if sma_20 > sma_50:
            score += 0.2
        
        if sma_10 < sma_20:
            score -= 0.2
        if sma_20 < sma_50:
            score -= 0.2
        
        direction = 'BUY' if score > 0.3 else 'SELL' if score < -0.3 else 'NEUTRAL'
        
        return {
            'score': score,
            'weight': 0.3,
            'direction': direction
        }
    
    def _evaluate_rsi(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate RSI signal"""
        rsi = indicators.get('rsi', 50)
        
        if rsi < 30:
            score = 0.8  # Oversold - potential buy
            direction = 'BUY'
        elif rsi < 40:
            score = 0.4
            direction = 'BUY'
        elif rsi > 70:
            score = -0.8  # Overbought - potential sell
            direction = 'SELL'
        elif rsi > 60:
            score = -0.4
            direction = 'SELL'
        else:
            score = 0.0
            direction = 'NEUTRAL'
        
        return {
            'score': score,
            'weight': 0.25,
            'direction': direction
        }
    
    def _evaluate_macd(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate MACD signal"""
        macd = indicators.get('macd', 0)
        signal = indicators.get('macd_signal', 0)
        histogram = indicators.get('macd_histogram', 0)
        
        score = 0.0
        
        # MACD above signal line
        if macd > signal:
            score += 0.5
        else:
            score -= 0.5
        
        # MACD histogram trend
        if histogram > 0:
            score += 0.3
        else:
            score -= 0.3
        
        direction = 'BUY' if score > 0.2 else 'SELL' if score < -0.2 else 'NEUTRAL'
        
        return {
            'score': score,
            'weight': 0.2,
            'direction': direction
        }
    
    def _evaluate_bollinger_bands(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate Bollinger Bands signal"""
        bb_position = indicators.get('bb_position', 0.5)
        
        if bb_position < 0.2:
            score = 0.6  # Near lower band - potential buy
            direction = 'BUY'
        elif bb_position > 0.8:
            score = -0.6  # Near upper band - potential sell
            direction = 'SELL'
        else:
            score = 0.0
            direction = 'NEUTRAL'
        
        return {
            'score': score,
            'weight': 0.15,
            'direction': direction
        }
    
    def _evaluate_volume(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate volume confirmation"""
        volume_ratio = indicators.get('volume_ratio', 1.0)
        price_change_1d = indicators.get('price_change_1d', 0)
        
        score = 0.0
        
        # High volume with price movement is significant
        if volume_ratio > 1.5:  # Above average volume
            if price_change_1d > 0.02:  # Price up > 2%
                score = 0.5
            elif price_change_1d < -0.02:  # Price down > 2%
                score = -0.5
        
        direction = 'BUY' if score > 0.2 else 'SELL' if score < -0.2 else 'NEUTRAL'
        
        return {
            'score': score,
            'weight': 0.1,
            'direction': direction
        }