"""
Optimized technical indicators with vectorized operations and improved performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from numba import jit
from utils.simple_logger import log_debug, log_error


# Numba-optimized functions for performance-critical calculations
@jit(nopython=True)
def _calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> float:
    """Numba-optimized RSI calculation."""
    if len(prices) < period + 1:
        return 50.0
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial average
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Exponential smoothing
    alpha = 1.0 / period
    for i in range(period, len(gains)):
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit(nopython=True)
def _calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Numba-optimized ATR calculation."""
    if len(high) < period + 1:
        return 0.02
    
    # Calculate true range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Calculate ATR using exponential moving average
    atr = np.mean(tr[1:period+1])
    alpha = 1.0 / period
    
    for i in range(period + 1, len(tr)):
        atr = (1 - alpha) * atr + alpha * tr[i]
    
    return atr


class TechnicalIndicators:
    """High-performance technical indicators with vectorized operations and caching."""
    
    # Class-level constants for optimization
    DEFAULT_PERIODS = {
        'rsi': 14,
        'sma': [5, 10, 20, 50],
        'ema': [12, 26],
        'bollinger': 20,
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'stochastic': {'k': 14, 'd': 3},
        'atr': 14
    }
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Optimized RSI calculation with fallback handling."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            # Convert to numpy array for numba optimization
            price_array = prices.values.astype(np.float64)
            
            # Remove any NaN values
            valid_prices = price_array[~np.isnan(price_array)]
            if len(valid_prices) < period + 1:
                return 50.0
            
            rsi = _calculate_rsi_numba(valid_prices, period)
            return float(rsi) if not np.isnan(rsi) else 50.0
            
        except Exception as e:
            log_debug(f"Error in RSI calculation: {e}")
            return 50.0
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
        """Vectorized moving average calculations."""
        try:
            if prices.empty:
                return {f'{ma_type}_{period}': 0.0 
                       for ma_type in ['sma', 'ema'] 
                       for period in TechnicalIndicators.DEFAULT_PERIODS['sma'] + TechnicalIndicators.DEFAULT_PERIODS['ema']}
            
            current_price = float(prices.iloc[-1])
            mas = {}
            
            # Vectorized SMA calculations
            for period in TechnicalIndicators.DEFAULT_PERIODS['sma']:
                if len(prices) >= period:
                    ma_value = prices.rolling(window=period, min_periods=1).mean().iloc[-1]
                    mas[f'sma_{period}'] = float(ma_value) if not pd.isna(ma_value) else current_price
                else:
                    mas[f'sma_{period}'] = current_price
            
            # Vectorized EMA calculations
            for span in TechnicalIndicators.DEFAULT_PERIODS['ema']:
                if len(prices) >= span:
                    ema_value = prices.ewm(span=span, adjust=False).mean().iloc[-1]
                    mas[f'ema_{span}'] = float(ema_value) if not pd.isna(ema_value) else current_price
                else:
                    mas[f'ema_{span}'] = current_price
            
            return mas
            
        except Exception as e:
            log_debug(f"Error calculating moving averages: {e}")
            return {f'sma_{p}': 0.0 for p in TechnicalIndicators.DEFAULT_PERIODS['sma']} | \
                   {f'ema_{p}': 0.0 for p in TechnicalIndicators.DEFAULT_PERIODS['ema']}
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, period: int = 20) -> float:
        """Optimized volatility calculation with proper annualization."""
        try:
            if len(prices) < 2:
                return 0.5
            
            # Vectorized returns calculation
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return 0.5
            
            # Use rolling standard deviation if sufficient data
            if len(returns) >= period:
                volatility = returns.rolling(window=period, min_periods=2).std().iloc[-1]
            else:
                volatility = returns.std()
            
            if pd.isna(volatility) or volatility <= 0:
                return 0.5
            
            # Annualize volatility (252 trading days)
            annualized_vol = volatility * np.sqrt(252)
            
            # Normalize to 0-1 scale (assuming 50% is high volatility)
            normalized_vol = min(annualized_vol / 0.5, 1.0)
            return max(0.0, normalized_vol)
            
        except Exception as e:
            log_debug(f"Error calculating volatility: {e}")
            return 0.5
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Vectorized Bollinger Bands calculation."""
        try:
            if len(prices) < period:
                current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
                default_spread = current_price * 0.02
                return {
                    'bb_upper': current_price + default_spread,
                    'bb_middle': current_price,
                    'bb_lower': current_price - default_spread
                }
            
            # Vectorized calculations using pandas rolling operations
            rolling_stats = prices.rolling(window=period, min_periods=1).agg(['mean', 'std'])
            
            bb_middle = float(rolling_stats['mean'].iloc[-1])
            bb_std = float(rolling_stats['std'].iloc[-1])
            
            # Handle edge cases
            if pd.isna(bb_std) or bb_std == 0:
                bb_std = bb_middle * 0.02  # 2% default standard deviation
            
            return {
                'bb_upper': bb_middle + (bb_std * std_dev),
                'bb_middle': bb_middle,
                'bb_lower': bb_middle - (bb_std * std_dev)
            }
            
        except Exception as e:
            log_debug(f"Error calculating Bollinger Bands: {e}")
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
            return {
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98
            }
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Optimized MACD calculation with vectorized operations."""
        try:
            if len(prices) < slow:
                return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
            
            # Vectorized EMA calculations
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            
            macd_line = ema_fast - ema_slow
            macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            macd_histogram = macd_line - macd_signal_line
            
            return {
                'macd': float(macd_line.iloc[-1]),
                'macd_signal': float(macd_signal_line.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1])
            }
            
        except Exception as e:
            log_debug(f"Error calculating MACD: {e}")
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Optimized ATR calculation using numba for performance."""
        try:
            required_cols = ['high', 'low', 'close']
            if len(df) < period or not all(col in df.columns for col in required_cols):
                return 0.02  # Default 2% ATR
            
            # Extract arrays for numba processing
            high_array = df['high'].values.astype(np.float64)
            low_array = df['low'].values.astype(np.float64)
            close_array = df['close'].values.astype(np.float64)
            
            # Use numba-optimized calculation
            atr = _calculate_atr_numba(high_array, low_array, close_array, period)
            
            return float(atr) if not np.isnan(atr) and atr > 0 else 0.02
            
        except Exception as e:
            log_debug(f"Error calculating ATR: {e}")
            return 0.02
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Vectorized Stochastic oscillator calculation."""
        try:
            required_cols = ['high', 'low', 'close']
            if len(df) < k_period or not all(col in df.columns for col in required_cols):
                return {'stoch_k': 50.0, 'stoch_d': 50.0}
            
            # Vectorized rolling min/max calculations
            rolling_low = df['low'].rolling(window=k_period, min_periods=1).min()
            rolling_high = df['high'].rolling(window=k_period, min_periods=1).max()
            
            # Avoid division by zero using numpy operations
            range_values = rolling_high - rolling_low
            range_values = np.where(range_values == 0, 1e-10, range_values)
            
            # Calculate %K
            stoch_k = 100 * ((df['close'] - rolling_low) / range_values)
            
            # Calculate %D (moving average of %K)
            stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
            
            return {
                'stoch_k': float(stoch_k.iloc[-1]),
                'stoch_d': float(stoch_d.iloc[-1])
            }
            
        except Exception as e:
            log_debug(f"Error calculating Stochastic: {e}")
            return {'stoch_k': 50.0, 'stoch_d': 50.0}
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Optimized batch addition of all technical indicators."""
        try:
            if df.empty or 'close' not in df.columns:
                return df
            
            # Work on a copy to avoid modifying original
            result_df = df.copy()
            
            # Batch calculate moving averages
            mas = TechnicalIndicators.calculate_moving_averages(df['close'])
            for key, value in mas.items():
                result_df[key] = value
            
            # Calculate other indicators
            result_df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            
            # Bollinger Bands
            bb_data = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            for key, value in bb_data.items():
                result_df[key] = value
            
            # MACD
            macd_data = TechnicalIndicators.calculate_macd(df['close'])
            for key, value in macd_data.items():
                result_df[key] = value
            
            # Stochastic
            stoch_data = TechnicalIndicators.calculate_stochastic(df)
            for key, value in stoch_data.items():
                result_df[key] = value
            
            # Volume indicators (vectorized)
            if 'volume' in df.columns:
                result_df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
                # Safe division avoiding zeros
                volume_sma_safe = np.where(result_df['volume_sma'] == 0, 1, result_df['volume_sma'])
                result_df['volume_ratio'] = df['volume'] / volume_sma_safe
            
            # ATR
            if all(col in df.columns for col in ['high', 'low', 'close']):
                result_df['atr'] = TechnicalIndicators.calculate_atr(df)
            else:
                result_df['atr'] = 0.02
            
            return result_df
            
        except Exception as e:
            log_error(f"Error adding technical indicators: {e}")
            return df
    
    @staticmethod
    def calculate_momentum_indicators(prices: pd.Series, volume: Optional[pd.Series] = None) -> Dict[str, float]:
        """Optimized momentum indicators with vectorized calculations."""
        try:
            if len(prices) < 2:
                return {'roc_1': 0.0, 'roc_5': 0.0, 'roc_10': 0.0, 'momentum': 0.0, 'volume_momentum': 1.0}
            
            indicators = {}
            
            # Vectorized Rate of Change calculations
            for period in [1, 5, 10]:
                if len(prices) > period:
                    roc = ((prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]) * 100
                    indicators[f'roc_{period}'] = float(roc) if not pd.isna(roc) else 0.0
                else:
                    indicators[f'roc_{period}'] = 0.0
            
            # Price momentum
            if len(prices) >= 10:
                momentum = prices.iloc[-1] - prices.iloc[-10]
                indicators['momentum'] = float(momentum) if not pd.isna(momentum) else 0.0
            else:
                indicators['momentum'] = 0.0
            
            # Volume momentum
            if volume is not None and len(volume) >= 5:
                vol_momentum = volume.iloc[-1] / volume.iloc[-5] if volume.iloc[-5] > 0 else 1.0
                indicators['volume_momentum'] = float(vol_momentum) if not pd.isna(vol_momentum) else 1.0
            else:
                indicators['volume_momentum'] = 1.0
            
            return indicators
            
        except Exception as e:
            log_debug(f"Error calculating momentum indicators: {e}")
            return {'roc_1': 0.0, 'roc_5': 0.0, 'roc_10': 0.0, 'momentum': 0.0, 'volume_momentum': 1.0}