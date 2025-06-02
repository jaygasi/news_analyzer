"""
Technical indicators calculation utilities with performance optimizations
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils.simple_logger import log_debug, log_error


class TechnicalIndicators:
    """Calculate various technical indicators efficiently with vectorized operations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI with optimized vectorized operations and error handling."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            # Fast path for small datasets
            clean_prices = pd.to_numeric(prices, errors='coerce').dropna()
            if len(clean_prices) < period + 1:
                return 50.0

            # Vectorized calculation
            delta_series = clean_prices.diff()
            delta_series = delta_series.fillna(0.0)
            
            # Convert to NumPy array for np.where to help Pylance's type inference
            delta_array = delta_series.to_numpy()

            # Use numpy for faster calculations
            gain = np.where(delta_array > 0, delta_array, 0.0)
            loss = np.where(delta_array < 0, -delta_array, 0.0)
            
            # Exponential moving average using pandas (optimized)
            alpha = 2.0 / (period + 1)
            avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean()
            avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean()
            
            # Avoid division by zero
            avg_loss_safe = np.where(avg_loss == 0, 1e-10, avg_loss)
            rs = avg_gain / avg_loss_safe
            rsi = 100 - (100 / (1 + rs))
            
            final_rsi = float(rsi.iloc[-1])
            return final_rsi if not pd.isna(final_rsi) else 50.0
            
        except Exception as e:
            log_debug(f"Error calculating RSI: {e}")
            return 50.0
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
        """Calculate various moving averages with optimized computations."""
        try:
            if prices.empty:
                return {f'{ma_type}_{period}': 0.0 
                       for ma_type in ['sma', 'ema'] 
                       for period in [5, 10, 20, 12, 26]}
            
            current_price = float(prices.iloc[-1])
            mas = {}
            
            # Simple moving averages - vectorized calculation
            sma_periods = [5, 10, 20, 50]
            for period in sma_periods:
                if len(prices) >= period:
                    mas[f'sma_{period}'] = float(prices.rolling(period, min_periods=1).mean().iloc[-1])
                else:
                    mas[f'sma_{period}'] = current_price
            
            # Exponential moving averages - optimized calculation
            ema_spans = [12, 26]
            for span in ema_spans:
                if len(prices) >= span:
                    mas[f'ema_{span}'] = float(prices.ewm(span=span, adjust=False).mean().iloc[-1])
                else:
                    mas[f'ema_{span}'] = current_price
            
            return mas
            
        except Exception as e:
            log_debug(f"Error calculating moving averages: {e}")
            return {
                'sma_5': 0, 'sma_10': 0, 'sma_20': 0, 'sma_50': 0,
                'ema_12': 0, 'ema_26': 0
            }
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, period: int = 20) -> float:
        """Calculate annualized volatility with optimized computation."""
        try:
            if len(prices) < 2:
                return 0.5
            
            # Vectorized returns calculation
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return 0.5
            
            # Use rolling window if enough data
            if len(returns) >= period:
                vol = returns.rolling(period, min_periods=2).std().iloc[-1]
            else:
                vol = returns.std()
            
            if pd.isna(vol) or vol <= 0:
                return 0.5
            
            # Annualize and normalize
            annualized_vol = vol * np.sqrt(252)
            normalized_vol = min(annualized_vol / 0.5, 1.0)
            return max(0.0, normalized_vol)
            
        except Exception as e:
            log_debug(f"Error calculating volatility: {e}")
            return 0.5
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands with optimized vectorized operations."""
        try:
            if len(prices) < period:
                current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
                return {
                    'bb_upper': current_price * 1.02,
                    'bb_middle': current_price,
                    'bb_lower': current_price * 0.98
                }
            
            # Vectorized calculations
            rolling_mean = prices.rolling(period, min_periods=1).mean()
            rolling_std = prices.rolling(period, min_periods=1).std()
            
            bb_middle = float(rolling_mean.iloc[-1])
            bb_std = float(rolling_std.iloc[-1])
            
            if pd.isna(bb_std) or bb_std == 0:
                bb_std = bb_middle * 0.02  # 2% default
            
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
        """Calculate MACD indicators with optimized computation."""
        try:
            if len(prices) < slow:
                return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
            
            # Vectorized EMA calculations
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
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
        """Calculate Average True Range with optimized vectorized operations."""
        try:
            required_cols = ['high', 'low', 'close']
            if len(df) < period or not all(col in df.columns for col in required_cols):
                return 0.02  # Default 2% ATR
            
            # Vectorized true range calculation
            high_low = df['high'] - df['low']
            high_close_prev = np.abs(df['high'] - df['close'].shift(1))
            low_close_prev = np.abs(df['low'] - df['close'].shift(1))
            
            # Use numpy for efficient maximum calculation
            true_range = np.maximum(
                high_low,
                np.maximum(high_close_prev, low_close_prev)
            )
            
            # Calculate ATR using pandas rolling mean
            atr_series = pd.Series(true_range).rolling(period, min_periods=1).mean()
            atr = float(atr_series.iloc[-1])
            
            return atr if not pd.isna(atr) and atr > 0 else 0.02
            
        except Exception as e:
            log_debug(f"Error calculating ATR: {e}")
            return 0.02
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic oscillator with optimized computation."""
        try:
            if len(df) < k_period or not all(col in df.columns for col in ['high', 'low', 'close']):
                return {'stoch_k': 50.0, 'stoch_d': 50.0}
            
            # Vectorized calculations
            lowest_low = df['low'].rolling(k_period, min_periods=1).min()
            highest_high = df['high'].rolling(k_period, min_periods=1).max()
            
            # Avoid division by zero
            range_val = highest_high - lowest_low
            range_val = np.where(range_val == 0, 1e-10, range_val)
            
            stoch_k = 100 * ((df['close'] - lowest_low) / range_val)
            stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
            
            return {
                'stoch_k': float(stoch_k.iloc[-1]),
                'stoch_d': float(stoch_d.iloc[-1])
            }
            
        except Exception as e:
            log_debug(f"Error calculating Stochastic: {e}")
            return {'stoch_k': 50.0, 'stoch_d': 50.0}
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe with optimized batch processing."""
        try:
            if df.empty or 'close' not in df.columns:
                return df
            
            df = df.copy()
            
            # Batch calculate moving averages
            mas = TechnicalIndicators.calculate_moving_averages(df['close'])
            for key, value in mas.items():
                df[key] = value
            
            # Calculate RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
            
            # Batch calculate Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
            for key, value in bb.items():
                df[key] = value
            
            # Batch calculate MACD
            macd = TechnicalIndicators.calculate_macd(df['close'])
            for key, value in macd.items():
                df[key] = value
            
            # Calculate Stochastic
            stoch = TechnicalIndicators.calculate_stochastic(df)
            for key, value in stoch.items():
                df[key] = value
            
            # Volume indicators (if available)
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
                # Avoid division by zero
                volume_sma_safe = np.where(df['volume_sma'] == 0, 1, df['volume_sma'])
                df['volume_ratio'] = df['volume'] / volume_sma_safe
            
            # ATR calculation
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['atr'] = TechnicalIndicators.calculate_atr(df)
            else:
                df['atr'] = 0.02
            
            return df
            
        except Exception as e:
            log_error(f"Error adding technical indicators: {e}")
            return df
    
    @staticmethod
    def calculate_momentum_indicators(prices: pd.Series, volume: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate momentum indicators with optimized batch processing."""
        try:
            if len(prices) < 2:
                return {'roc_1': 0.0, 'roc_5': 0.0, 'roc_10': 0.0, 'momentum': 0.0}
            
            indicators = {}
            
            # Rate of Change (ROC) for different periods
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
            
            # Volume momentum (if available)
            if volume is not None and len(volume) >= 5:
                vol_momentum = volume.iloc[-1] / volume.iloc[-5] if volume.iloc[-5] > 0 else 1.0
                indicators['volume_momentum'] = float(vol_momentum) if not pd.isna(vol_momentum) else 1.0
            else:
                indicators['volume_momentum'] = 1.0
            
            return indicators
            
        except Exception as e:
            log_debug(f"Error calculating momentum indicators: {e}")
            return {'roc_1': 0.0, 'roc_5': 0.0, 'roc_10': 0.0, 'momentum': 0.0, 'volume_momentum': 1.0}