"""
Technical indicators calculation utilities
"""
import pandas as pd
import numpy as np
from typing import Dict
from utils.simple_logger import log_debug, log_error


class TechnicalIndicators:
    """Calculate various technical indicators efficiently"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI with optimized vectorized operations"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            clean_prices = pd.to_numeric(prices, errors='coerce').dropna()
            if len(clean_prices) < period + 1:
                return 50.0

            delta = clean_prices.diff().dropna()
            delta = pd.to_numeric(delta, errors='coerce').fillna(0.0)

            gain = delta.where(delta > 0.0, 0.0)
            loss = (-delta).where(delta < 0.0, 0.0)
            
            alpha = 2.0 / (period + 1)
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            log_debug(f"Error calculating RSI: {e}")
            return 50.0
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages"""
        try:
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            mas = {}
            
            # Simple moving averages
            for period in [5, 10, 20]:
                if len(prices) >= period:
                    mas[f'sma_{period}'] = prices.rolling(period).mean().iloc[-1]
                else:
                    mas[f'sma_{period}'] = current_price
            
            # Exponential moving averages
            for span in [12, 26]:
                if len(prices) >= span:
                    mas[f'ema_{span}'] = prices.ewm(span=span).mean().iloc[-1]
                else:
                    mas[f'ema_{span}'] = current_price
            
            return mas
            
        except Exception as e:
            log_debug(f"Error calculating moving averages: {e}")
            return {
                'sma_5': 0, 'sma_10': 0, 'sma_20': 0,
                'ema_12': 0, 'ema_26': 0
            }
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, period: int = 20) -> float:
        """Calculate annualized volatility"""
        try:
            if len(prices) < 2:
                return 0.5
            
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return 0.5
            
            if len(returns) >= period:
                vol = returns.rolling(period).std().iloc[-1]
            else:
                vol = returns.std()
            
            if not pd.isna(vol):
                annualized_vol = vol * np.sqrt(252)
                normalized_vol = min(annualized_vol / 0.5, 1.0)
                return max(0.0, normalized_vol)
            
            return 0.5
            
        except Exception as e:
            log_debug(f"Error calculating volatility: {e}")
            return 0.5
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices.iloc[-1] if len(prices) > 0 else 0
                return {
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98
                }
            
            bb_middle = prices.rolling(period).mean().iloc[-1]
            bb_std = prices.rolling(period).std().iloc[-1]
            
            return {
                'bb_upper': bb_middle + (bb_std * std_dev),
                'bb_middle': bb_middle,
                'bb_lower': bb_middle - (bb_std * std_dev)
            }
            
        except Exception as e:
            log_debug(f"Error calculating Bollinger Bands: {e}")
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98
            }
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD indicators"""
        try:
            if len(prices) < slow:
                return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
            
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            macd_histogram = macd - macd_signal
            
            return {
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_histogram': macd_histogram.iloc[-1]
            }
            
        except Exception as e:
            log_debug(f"Error calculating MACD: {e}")
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period or not all(col in df.columns for col in ['high', 'low', 'close']):
                return 0.02  # Default 2% ATR
            
            df = df.copy()
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            atr = df['true_range'].rolling(period).mean().iloc[-1]
            return atr if not pd.isna(atr) else 0.02
            
        except Exception as e:
            log_debug(f"Error calculating ATR: {e}")
            return 0.02
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        try:
            if df.empty or 'close' not in df.columns:
                return df
            
            df = df.copy()
            
            # Moving averages
            mas = TechnicalIndicators.calculate_moving_averages(df['close'])
            for key, value in mas.items():
                df[key] = value
            
            # RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            
            # Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            for key, value in bb.items():
                df[key] = value
            
            # MACD
            macd = TechnicalIndicators.calculate_macd(df['close'])
            for key, value in macd.items():
                df[key] = value
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # ATR
            df['atr'] = TechnicalIndicators.calculate_atr(df)
            
            return df
            
        except Exception as e:
            log_error(f"Error adding technical indicators: {e}")
            return df