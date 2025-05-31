"""
Professional Technical Analysis Strategies for News-Driven Trading
Implements proven technical setups and entry/exit strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug


class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1


class TrendDirection(Enum):
    """Trend direction classifications"""
    STRONG_BULLISH = 3
    BULLISH = 2
    NEUTRAL = 1
    BEARISH = 0
    STRONG_BEARISH = -1


@dataclass
class StrategySignal:
    """Technical strategy signal with detailed reasoning"""
    strategy_name: str
    signal_type: str           # "long", "short", "avoid"
    strength: SignalStrength   # Signal strength 1-5
    confidence: float          # 0-1 confidence in signal
    entry_price: float         # Suggested entry price
    stop_loss: float           # Suggested stop loss
    target_1: float            # First profit target
    target_2: Optional[float]  # Second profit target
    reasoning: str             # Why this signal was generated
    risk_reward_ratio: float   # Risk/reward ratio
    holding_period: str        # Expected holding period
    
    # Technical levels
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    key_moving_average: Optional[float] = None


class TechnicalStrategies:
    """Professional technical analysis strategies for news-driven trading"""
    
    def __init__(self, fmp_loader):
        """Initialize with data loader for historical analysis"""
        self.fmp_loader = fmp_loader
        self.strategy_cache = {}
        
    def analyze_with_strategies(self, symbol: str, current_price: float, 
                              technical_analysis, news_sentiment: float) -> List[StrategySignal]:
        """Analyze symbol using multiple proven technical strategies"""
        if not technical_analysis:
            return []
        
        try:
            # Get additional data for strategy analysis
            historical_data = self._get_strategy_data(symbol)
            if historical_data is None:
                return []
            
            signals = []
            
            # Strategy 1: RSI Mean Reversion + News Catalyst
            rsi_signal = self._rsi_mean_reversion_strategy(
                symbol, current_price, technical_analysis, news_sentiment, historical_data
            )
            if rsi_signal:
                signals.append(rsi_signal)
            
            # Strategy 2: Moving Average Breakout + News Confirmation
            ma_signal = self._moving_average_breakout_strategy(
                symbol, current_price, technical_analysis, news_sentiment, historical_data
            )
            if ma_signal:
                signals.append(ma_signal)
            
            # Strategy 3: Support/Resistance + News Catalyst
            sr_signal = self._support_resistance_strategy(
                symbol, current_price, technical_analysis, news_sentiment, historical_data
            )
            if sr_signal:
                signals.append(sr_signal)
            
            # Strategy 4: Volume Breakout + News Momentum
            volume_signal = self._volume_breakout_strategy(
                symbol, current_price, technical_analysis, news_sentiment, historical_data
            )
            if volume_signal:
                signals.append(volume_signal)
            
            # Strategy 5: Trend Following + News Alignment
            trend_signal = self._trend_following_strategy(
                symbol, current_price, technical_analysis, news_sentiment, historical_data
            )
            if trend_signal:
                signals.append(trend_signal)
            
            # Sort by strength and confidence
            signals.sort(key=lambda x: (x.strength.value, x.confidence), reverse=True)
            
            return signals
            
        except Exception as e:
            log_error(f"Error in strategy analysis for {symbol}: {e}")
            return []
    
    def _get_strategy_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data needed for strategy analysis"""
        try:
            # Get 60 days of data for strategy calculations
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            data = self.fmp_loader._make_request(f"historical-price-full/{symbol}", {
                'from': start_date,
                'to': end_date
            })
            
            if data and 'historical' in data and data['historical']:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Ensure numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate additional technical indicators
                df = self._add_technical_indicators(df)
                return df
                
        except Exception as e:
            log_debug(f"Could not get strategy data for {symbol}: {e}")
        
        return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to dataframe"""
        try:
            # Moving averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_8'] = df['close'].ewm(span=8).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Average True Range (ATR) for volatility
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(14).mean()
            
            return df
            
        except Exception as e:
            log_error(f"Error adding technical indicators: {e}")
            return df
    
    def _rsi_mean_reversion_strategy(self, symbol: str, current_price: float,
                                   tech_analysis, news_sentiment: float, 
                                   df: pd.DataFrame) -> Optional[StrategySignal]:
        """RSI Mean Reversion Strategy + News Catalyst"""
        try:
            if len(df) < 20:
                return None
            
            current_rsi = tech_analysis.rsi
            prev_rsi = df['rsi'].iloc[-2] if len(df) >= 2 else current_rsi
            
            # Strategy conditions
            oversold_threshold = 30
            overbought_threshold = 70
            extreme_oversold = 20
            extreme_overbought = 80
            
            # Long setup: Oversold + Positive news
            if (current_rsi < oversold_threshold and news_sentiment > 0.3):
                
                # Determine strength based on RSI level and news sentiment
                if current_rsi < extreme_oversold and news_sentiment > 0.6:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.85
                elif current_rsi < 25 and news_sentiment > 0.5:
                    strength = SignalStrength.STRONG
                    confidence = 0.75
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.65
                
                # Calculate targets using ATR
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                
                entry_price = current_price
                stop_loss = current_price - (atr * 1.5)  # 1.5 ATR stop
                target_1 = current_price + (atr * 2.0)   # 2:1 risk/reward
                target_2 = current_price + (atr * 3.0)   # 3:1 risk/reward
                
                # Find resistance level
                resistance = self._find_resistance_level(df, current_price)
                if resistance:
                    target_1 = min(target_1, resistance * 0.98)  # Just below resistance
                
                risk_reward = (target_1 - entry_price) / (entry_price - stop_loss)
                
                return StrategySignal(
                    strategy_name="RSI Mean Reversion Long",
                    signal_type="long",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"RSI oversold ({current_rsi:.1f}) + positive news catalyst. "
                             f"Mean reversion expected with {risk_reward:.1f}:1 R/R.",
                    risk_reward_ratio=risk_reward,
                    holding_period="1-3 days",
                    resistance_level=resistance
                )
            
            # Short setup: Overbought + Negative news
            elif (current_rsi > overbought_threshold and news_sentiment < -0.3):
                
                if current_rsi > extreme_overbought and news_sentiment < -0.6:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.85
                elif current_rsi > 75 and news_sentiment < -0.5:
                    strength = SignalStrength.STRONG
                    confidence = 0.75
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.65
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                
                entry_price = current_price
                stop_loss = current_price + (atr * 1.5)
                target_1 = current_price - (atr * 2.0)
                target_2 = current_price - (atr * 3.0)
                
                # Find support level
                support = self._find_support_level(df, current_price)
                if support:
                    target_1 = max(target_1, support * 1.02)  # Just above support
                
                risk_reward = (entry_price - target_1) / (stop_loss - entry_price)
                
                return StrategySignal(
                    strategy_name="RSI Mean Reversion Short",
                    signal_type="short",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"RSI overbought ({current_rsi:.1f}) + negative news catalyst. "
                             f"Mean reversion expected with {risk_reward:.1f}:1 R/R.",
                    risk_reward_ratio=risk_reward,
                    holding_period="1-3 days",
                    support_level=support
                )
            
            return None
            
        except Exception as e:
            log_error(f"RSI strategy error for {symbol}: {e}")
            return None
    
    def _moving_average_breakout_strategy(self, symbol: str, current_price: float,
                                        tech_analysis, news_sentiment: float,
                                        df: pd.DataFrame) -> Optional[StrategySignal]:
        """Moving Average Breakout Strategy + News Confirmation"""
        try:
            if len(df) < 50:
                return None
            
            # Get recent MA values
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            ema_8 = df['ema_8'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            
            if pd.isna(sma_20) or pd.isna(sma_50):
                return None
            
            # Check for Golden Cross setup (bullish)
            if (ema_8 > ema_21 and sma_20 > sma_50 and 
                current_price > sma_20 and news_sentiment > 0.4):
                
                # Determine strength based on MA alignment and news
                ma_separation = (sma_20 - sma_50) / sma_50
                
                if ma_separation > 0.05 and news_sentiment > 0.7:  # 5% separation + strong news
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.80
                elif ma_separation > 0.02 and news_sentiment > 0.5:
                    strength = SignalStrength.STRONG
                    confidence = 0.70
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.60
                
                # Use 20 SMA as support for stop
                entry_price = current_price
                stop_loss = sma_20 * 0.98  # 2% below 20 SMA
                
                # Target based on recent volatility
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                target_1 = current_price + (atr * 2.5)
                target_2 = current_price + (atr * 4.0)
                
                risk_reward = (target_1 - entry_price) / (entry_price - stop_loss)
                
                return StrategySignal(
                    strategy_name="MA Breakout Long",
                    signal_type="long",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Bullish MA alignment (20SMA>{sma_20:.2f}, 50SMA>{sma_50:.2f}) "
                             f"+ positive news. Trend continuation expected.",
                    risk_reward_ratio=risk_reward,
                    holding_period="3-7 days",
                    key_moving_average=sma_20
                )
            
            # Check for Death Cross setup (bearish)
            elif (ema_8 < ema_21 and sma_20 < sma_50 and 
                  current_price < sma_20 and news_sentiment < -0.4):
                
                ma_separation = (sma_50 - sma_20) / sma_50
                
                if ma_separation > 0.05 and news_sentiment < -0.7:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.80
                elif ma_separation > 0.02 and news_sentiment < -0.5:
                    strength = SignalStrength.STRONG
                    confidence = 0.70
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.60
                
                entry_price = current_price
                stop_loss = sma_20 * 1.02  # 2% above 20 SMA
                
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
                target_1 = current_price - (atr * 2.5)
                target_2 = current_price - (atr * 4.0)
                
                risk_reward = (entry_price - target_1) / (stop_loss - entry_price)
                
                return StrategySignal(
                    strategy_name="MA Breakout Short",
                    signal_type="short",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Bearish MA alignment (20SMA<{sma_20:.2f}, 50SMA<{sma_50:.2f}) "
                             f"+ negative news. Trend continuation expected.",
                    risk_reward_ratio=risk_reward,
                    holding_period="3-7 days",
                    key_moving_average=sma_20
                )
            
            return None
            
        except Exception as e:
            log_error(f"MA breakout strategy error for {symbol}: {e}")
            return None
    
    def _support_resistance_strategy(self, symbol: str, current_price: float,
                                   tech_analysis, news_sentiment: float,
                                   df: pd.DataFrame) -> Optional[StrategySignal]:
        """Support/Resistance Breakout Strategy"""
        try:
            if len(df) < 30:
                return None
            
            # Find key support and resistance levels
            support = self._find_support_level(df, current_price)
            resistance = self._find_resistance_level(df, current_price)
            
            if not support or not resistance:
                return None
            
            # Breakout above resistance with positive news
            if (current_price > resistance * 1.005 and  # 0.5% above resistance
                news_sentiment > 0.5 and
                tech_analysis.volume_score > 0.7):  # High volume confirmation
                
                # Distance from resistance determines strength
                breakout_strength = (current_price - resistance) / resistance
                
                if breakout_strength > 0.03 and news_sentiment > 0.7:  # 3% breakout
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.85
                elif breakout_strength > 0.015:  # 1.5% breakout
                    strength = SignalStrength.STRONG
                    confidence = 0.75
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.65
                
                entry_price = current_price
                stop_loss = resistance * 0.995  # Back below resistance
                
                # Target based on support-resistance range
                sr_range = resistance - support
                target_1 = resistance + (sr_range * 0.618)  # 61.8% extension
                target_2 = resistance + (sr_range * 1.0)    # 100% extension
                
                risk_reward = (target_1 - entry_price) / (entry_price - stop_loss)
                
                return StrategySignal(
                    strategy_name="Resistance Breakout Long",
                    signal_type="long",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Breakout above resistance ${resistance:.2f} with high volume "
                             f"and positive news. Target S/R range extension.",
                    risk_reward_ratio=risk_reward,
                    holding_period="2-5 days",
                    support_level=support,
                    resistance_level=resistance
                )
            
            # Breakdown below support with negative news
            elif (current_price < support * 0.995 and  # 0.5% below support
                  news_sentiment < -0.5 and
                  tech_analysis.volume_score > 0.7):
                
                breakdown_strength = (support - current_price) / support
                
                if breakdown_strength > 0.03 and news_sentiment < -0.7:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.85
                elif breakdown_strength > 0.015:
                    strength = SignalStrength.STRONG
                    confidence = 0.75
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.65
                
                entry_price = current_price
                stop_loss = support * 1.005  # Back above support
                
                sr_range = resistance - support
                target_1 = support - (sr_range * 0.618)
                target_2 = support - (sr_range * 1.0)
                
                risk_reward = (entry_price - target_1) / (stop_loss - entry_price)
                
                return StrategySignal(
                    strategy_name="Support Breakdown Short",
                    signal_type="short",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Breakdown below support ${support:.2f} with high volume "
                             f"and negative news. Target S/R range extension.",
                    risk_reward_ratio=risk_reward,
                    holding_period="2-5 days",
                    support_level=support,
                    resistance_level=resistance
                )
            
            return None
            
        except Exception as e:
            log_error(f"S/R strategy error for {symbol}: {e}")
            return None
    
    def _volume_breakout_strategy(self, symbol: str, current_price: float,
                                tech_analysis, news_sentiment: float,
                                df: pd.DataFrame) -> Optional[StrategySignal]:
        """Volume Breakout Strategy with News Catalyst"""
        try:
            if len(df) < 20:
                return None
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume_sma'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # High volume (2x+ average) + strong news sentiment
            if (volume_ratio >= 2.0 and abs(news_sentiment) > 0.6):
                
                # Price action confirmation
                recent_high = df['high'].tail(5).max()
                recent_low = df['low'].tail(5).min()
                price_range = recent_high - recent_low
                
                # Bullish volume breakout
                if (news_sentiment > 0.6 and 
                    current_price > recent_high * 0.999):  # At/near recent high
                    
                    if volume_ratio >= 4.0:  # Exceptional volume
                        strength = SignalStrength.VERY_STRONG
                        confidence = 0.80
                    elif volume_ratio >= 3.0:
                        strength = SignalStrength.STRONG
                        confidence = 0.70
                    else:
                        strength = SignalStrength.MODERATE
                        confidence = 0.60
                    
                    entry_price = current_price
                    stop_loss = recent_low
                    target_1 = current_price + (price_range * 1.0)
                    target_2 = current_price + (price_range * 1.5)
                    
                    risk_reward = (target_1 - entry_price) / (entry_price - stop_loss)
                    
                    return StrategySignal(
                        strategy_name="Volume Breakout Long",
                        signal_type="long",
                        strength=strength,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        reasoning=f"High volume breakout ({volume_ratio:.1f}x avg) with strong "
                                 f"positive news. Institutional interest evident.",
                        risk_reward_ratio=risk_reward,
                        holding_period="1-4 days"
                    )
                
                # Bearish volume breakdown
                elif (news_sentiment < -0.6 and 
                      current_price < recent_low * 1.001):  # At/near recent low
                    
                    if volume_ratio >= 4.0:
                        strength = SignalStrength.VERY_STRONG
                        confidence = 0.80
                    elif volume_ratio >= 3.0:
                        strength = SignalStrength.STRONG
                        confidence = 0.70
                    else:
                        strength = SignalStrength.MODERATE
                        confidence = 0.60
                    
                    entry_price = current_price
                    stop_loss = recent_high
                    target_1 = current_price - (price_range * 1.0)
                    target_2 = current_price - (price_range * 1.5)
                    
                    risk_reward = (entry_price - target_1) / (stop_loss - entry_price)
                    
                    return StrategySignal(
                        strategy_name="Volume Breakdown Short",
                        signal_type="short",
                        strength=strength,
                        confidence=confidence,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_1=target_1,
                        target_2=target_2,
                        reasoning=f"High volume breakdown ({volume_ratio:.1f}x avg) with strong "
                                 f"negative news. Distribution evident.",
                        risk_reward_ratio=risk_reward,
                        holding_period="1-4 days"
                    )
            
            return None
            
        except Exception as e:
            log_error(f"Volume strategy error for {symbol}: {e}")
            return None
    
    def _trend_following_strategy(self, symbol: str, current_price: float,
                                tech_analysis, news_sentiment: float,
                                df: pd.DataFrame) -> Optional[StrategySignal]:
        """Trend Following Strategy with News Alignment"""
        try:
            if len(df) < 50:
                return None
            
            # Determine trend strength
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if pd.isna(sma_20) or pd.isna(sma_50):
                return None
            
            # Strong uptrend conditions
            uptrend_conditions = [
                current_price > sma_20,
                sma_20 > sma_50,
                df['close'].iloc[-1] > df['close'].iloc[-5],  # 5-day momentum
                df['close'].iloc[-5] > df['close'].iloc[-10]  # 10-day momentum
            ]
            
            # Strong downtrend conditions  
            downtrend_conditions = [
                current_price < sma_20,
                sma_20 < sma_50,
                df['close'].iloc[-1] < df['close'].iloc[-5],
                df['close'].iloc[-5] < df['close'].iloc[-10]
            ]
            
            uptrend_strength = sum(uptrend_conditions)
            downtrend_strength = sum(downtrend_conditions)
            
            # Strong uptrend + positive news
            if (uptrend_strength >= 3 and news_sentiment > 0.4):
                
                if uptrend_strength == 4 and news_sentiment > 0.7:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.75
                elif uptrend_strength == 4 or news_sentiment > 0.6:
                    strength = SignalStrength.STRONG
                    confidence = 0.65
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.55
                
                entry_price = current_price
                stop_loss = sma_20 * 0.97  # 3% below 20 SMA
                
                # Trend-based targets
                trend_momentum = (current_price - sma_50) / sma_50
                target_1 = current_price * (1 + (trend_momentum * 0.5))
                target_2 = current_price * (1 + trend_momentum)
                
                risk_reward = (target_1 - entry_price) / (entry_price - stop_loss)
                
                return StrategySignal(
                    strategy_name="Trend Following Long",
                    signal_type="long",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Strong uptrend ({uptrend_strength}/4 conditions) with positive "
                             f"news alignment. Trend continuation expected.",
                    risk_reward_ratio=risk_reward,
                    holding_period="5-10 days",
                    key_moving_average=sma_20
                )
            
            # Strong downtrend + negative news
            elif (downtrend_strength >= 3 and news_sentiment < -0.4):
                
                if downtrend_strength == 4 and news_sentiment < -0.7:
                    strength = SignalStrength.VERY_STRONG
                    confidence = 0.75
                elif downtrend_strength == 4 or news_sentiment < -0.6:
                    strength = SignalStrength.STRONG
                    confidence = 0.65
                else:
                    strength = SignalStrength.MODERATE
                    confidence = 0.55
                
                entry_price = current_price
                stop_loss = sma_20 * 1.03  # 3% above 20 SMA
                
                trend_momentum = (sma_50 - current_price) / sma_50
                target_1 = current_price * (1 - (trend_momentum * 0.5))
                target_2 = current_price * (1 - trend_momentum)
                
                risk_reward = (entry_price - target_1) / (stop_loss - entry_price)
                
                return StrategySignal(
                    strategy_name="Trend Following Short",
                    signal_type="short",
                    strength=strength,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    reasoning=f"Strong downtrend ({downtrend_strength}/4 conditions) with negative "
                             f"news alignment. Trend continuation expected.",
                    risk_reward_ratio=risk_reward,
                    holding_period="5-10 days",
                    key_moving_average=sma_20
                )
            
            return None
            
        except Exception as e:
            log_error(f"Trend following strategy error for {symbol}: {e}")
            return None
    
    def _find_support_level(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find nearest significant support level"""
        try:
            # Look at recent lows for support
            lows = df['low'].tail(30)
            
            # Find price levels that held multiple times
            support_candidates = []
            for low in lows:
                touches = sum(abs(lows - low) / low < 0.02)  # Within 2%
                if touches >= 2:  # At least 2 touches
                    support_candidates.append(low)
            
            if support_candidates:
                # Return the highest support below current price
                valid_supports = [s for s in support_candidates if s < current_price * 0.98]
                return max(valid_supports) if valid_supports else None
            
            return None
            
        except Exception:
            return None
    
    def _find_resistance_level(self, df: pd.DataFrame, current_price: float) -> Optional[float]:
        """Find nearest significant resistance level"""
        try:
            # Look at recent highs for resistance
            highs = df['high'].tail(30)
            
            resistance_candidates = []
            for high in highs:
                touches = sum(abs(highs - high) / high < 0.02)  # Within 2%
                if touches >= 2:  # At least 2 touches
                    resistance_candidates.append(high)
            
            if resistance_candidates:
                # Return the lowest resistance above current price
                valid_resistances = [r for r in resistance_candidates if r > current_price * 1.02]
                return min(valid_resistances) if valid_resistances else None
            
            return None
            
        except Exception:
            return None
    
    def get_best_strategy(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Get the best strategy signal from a list"""
        if not signals:
            return None
        
        # Filter for good risk/reward ratios
        good_signals = [s for s in signals if s.risk_reward_ratio >= 1.5]
        
        if not good_signals:
            good_signals = signals  # Fallback to all signals
        
        # Sort by strength and confidence
        good_signals.sort(key=lambda x: (x.strength.value, x.confidence), reverse=True)
        
        return good_signals[0]
    
    def format_strategy_summary(self, signal: StrategySignal) -> str:
        """Format strategy signal for logging"""
        return (f"{signal.strategy_name}: {signal.signal_type.upper()} "
               f"${signal.entry_price:.2f} -> ${signal.target_1:.2f} "
               f"(SL: ${signal.stop_loss:.2f}, R/R: {signal.risk_reward_ratio:.1f}) "
               f"[{signal.strength.name} - {signal.confidence:.0%}]")