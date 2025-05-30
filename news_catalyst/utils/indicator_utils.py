"""
Optimized indicator utilities with improved numerical stability and performance
"""
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from scipy.stats import linregress
import warnings

try:
    from statsmodels.nonparametric.kernel_regression import KernelReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    KernelReg = None

# Suppress numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_trend_linear_regression(values: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Calculate trend using linear regression with improved error handling.
    
    Args:
        values: List or array of values
        
    Returns:
        Array of trend values
    """
    if not values or len(values) < 2:
        return np.array([])
    
    values_array = np.asarray(values, dtype=np.float64)
    
    # Remove NaN values
    valid_mask = ~np.isnan(values_array)
    if np.sum(valid_mask) < 2:
        return np.full_like(values_array, np.nan)
    
    x = np.arange(len(values_array))
    
    try:
        # Use only valid values for regression
        valid_x = x[valid_mask]
        valid_y = values_array[valid_mask]
        
        slope, intercept, _, _, _ = linregress(valid_x, valid_y)
        trend = intercept + slope * x
        
        return trend.astype(np.float64)
    except (ValueError, LinAlgError):
        return np.full_like(values_array, np.nan)


def calculate_trend(df: pd.DataFrame,
                   target_column: str = 'close',
                   out_column: str = 'close_smoothed_slope',
                   bandwidth: int = 9,
                   slope_window: int = 3) -> Union[pd.Series, float]:
    """
    Calculate trend with smoothing and slope computation.
    
    Args:
        df: Input DataFrame
        target_column: Column to analyze
        out_column: Output column name
        bandwidth: Smoothing bandwidth
        slope_window: Window for slope calculation
        
    Returns:
        Trend values as Series or single value
    """
    if df.empty or target_column not in df.columns:
        return pd.Series([], name=out_column)
    
    try:
        # Add smoothed line
        df_with_smooth = add_kernel_reg_smoothed_line(
            df.copy(),
            column_list=[target_column],
            output_cols=[f"{target_column}_smoothed"],
            bandwidth=bandwidth,
            var_type='c'
        )
        
        # Calculate slope
        df_with_slope = compute_slope(
            df_with_smooth,
            target_col=f"{target_column}_smoothed",
            slope_col=out_column,
            window_size=slope_window
        )
        
        return df_with_slope[out_column]
    except Exception:
        # Fallback to simple linear trend
        values = df[target_column].values
        trend_values = calculate_trend_linear_regression(values)
        return pd.Series(trend_values, index=df.index, name=out_column)


def compute_slope_internal(y_values: np.ndarray) -> float:
    """
    Internal function to compute slope with improved numerical stability.
    
    Args:
        y_values: Array of y values
        
    Returns:
        Slope value
    """
    # Remove non-finite values
    finite_mask = np.isfinite(y_values)
    y_clean = y_values[finite_mask]
    
    if len(y_clean) < 2:
        return 0.0
    
    # Check for constant values
    if np.all(y_clean == y_clean[0]):
        return 0.0
    
    x_clean = np.arange(len(y_clean), dtype=np.float64)
    
    try:
        # Use numpy polyfit for better numerical stability
        slope, _ = np.polyfit(x_clean, y_clean, 1)
        return float(slope) if np.isfinite(slope) else 0.0
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_slope(df: pd.DataFrame, 
                 target_col: str, 
                 slope_col: str, 
                 window_size: int) -> pd.DataFrame:
    """
    Compute slope of time series with rolling window and optimized calculation.
    
    Args:
        df: Input DataFrame
        target_col: Column containing target price data
        slope_col: Column name for computed slopes
        window_size: Rolling window size
        
    Returns:
        DataFrame with computed slopes
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist in DataFrame")
    
    if window_size < 2:
        raise ValueError("Window size must be at least 2")
    
    # Create copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Vectorized rolling slope calculation
    rolling_slopes = (
        df_copy[target_col]
        .rolling(window=window_size, min_periods=2)
        .apply(compute_slope_internal, raw=True)
    )
    
    df_copy[slope_col] = rolling_slopes
    
    return df_copy


def calculate_trend_numpy(values: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """
    Calculate trend using numpy with improved error handling.
    
    Args:
        values: Input values
        
    Returns:
        Tuple of (slope, intercept)
    """
    if not values:
        return 0.0, 0.0
    
    values_array = np.asarray(values, dtype=np.float64)
    
    # Remove NaN values
    valid_mask = ~np.isnan(values_array)
    if np.sum(valid_mask) < 2:
        return 0.0, 0.0
    
    valid_values = values_array[valid_mask]
    x = np.arange(1, len(valid_values) + 1, dtype=np.float64)
    
    try:
        slope, intercept = np.polyfit(x, valid_values, 1)
        return float(slope), float(intercept)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0


def add_kernel_reg_smoothed_line(df: pd.DataFrame, 
                                column_list: List[str] = None,
                                output_cols: Optional[List[str]] = None,
                                bandwidth: Union[float, List[float]] = 2,
                                var_type: str = 'c') -> pd.DataFrame:
    """
    Add smoothed lines using kernel regression with fallback to simpler methods.
    
    Args:
        df: Input DataFrame
        column_list: List of columns to smooth
        output_cols: List of output column names
        bandwidth: Bandwidth parameter(s)
        var_type: Variable type for kernel regression
        
    Returns:
        DataFrame with smoothed columns added
    """
    if column_list is None:
        column_list = ['close']
    
    if output_cols is None:
        output_cols = [f"{col}_smoothed" for col in column_list]
    
    if len(column_list) != len(output_cols):
        raise ValueError("Number of input columns must equal number of output columns")
    
    df_copy = df.copy()
    
    # Ensure bandwidth is a list
    if not isinstance(bandwidth, list):
        bandwidth = [bandwidth] * len(column_list)
    
    for input_col, output_col, bw in zip(column_list, output_cols, bandwidth):
        if input_col not in df_copy.columns:
            continue
        
        data = df_copy[input_col].values
        
        # Check for sufficient data
        if len(data) < 3:
            df_copy[output_col] = data
            continue
        
        # Remove NaN values for processing
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) < 3:
            df_copy[output_col] = data
            continue
        
        try:
            if STATSMODELS_AVAILABLE and len(data) >= 10:
                # Use kernel regression for larger datasets
                index_array = np.arange(len(data), dtype=np.float64)
                
                kernel_reg = KernelReg(
                    endog=data, 
                    exog=index_array, 
                    var_type=var_type, 
                    bw=[bw]
                )
                smoothed_values, _ = kernel_reg.fit(index_array)
                df_copy[output_col] = smoothed_values
            else:
                # Fallback to moving average smoothing
                window_size = max(3, min(int(bw * 2), len(data) // 2))
                smoothed = pd.Series(data).rolling(
                    window=window_size, 
                    center=True, 
                    min_periods=1
                ).mean()
                df_copy[output_col] = smoothed.values
                
        except Exception:
            # Ultimate fallback: simple moving average
            window_size = max(3, min(5, len(data) // 2))
            smoothed = pd.Series(data).rolling(
                window=window_size, 
                center=True, 
                min_periods=1
            ).mean()
            df_copy[output_col] = smoothed.values
    
    return df_copy


def calculate_rsi(prices: Union[pd.Series, np.ndarray], 
                 period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) with optimized computation.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values as numpy array
    """
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)  # Neutral RSI
    
    prices_array = np.asarray(prices, dtype=np.float64)
    
    # Calculate price changes
    price_changes = np.diff(prices_array)
    
    # Separate gains and losses
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi_values = np.full(len(prices), np.nan)
    
    # Calculate RSI for each period
    for i in range(period, len(prices)):
        if i == period:
            # First RSI calculation
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
        else:
            # Subsequent RSI calculations (smoothed)
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            
            if avg_loss == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values


def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray], 
                            period: int = 20, 
                            std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands with optimized computation.
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    prices_series = pd.Series(prices)
    
    # Calculate moving average (middle band)
    middle_band = prices_series.rolling(window=period, min_periods=1).mean()
    
    # Calculate standard deviation
    rolling_std = prices_series.rolling(window=period, min_periods=1).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band.values, middle_band.values, lower_band.values


def calculate_macd(prices: Union[pd.Series, np.ndarray], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence) with optimized computation.
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    prices_series = pd.Series(prices)
    
    # Calculate exponential moving averages
    ema_fast = prices_series.ewm(span=fast_period, min_periods=1).mean()
    ema_slow = prices_series.ewm(span=slow_period, min_periods=1).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line.values, signal_line.values, histogram.values