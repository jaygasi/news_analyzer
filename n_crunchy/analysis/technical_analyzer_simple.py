import pandas as pd
import numpy as np # Import numpy for robust numerical operations
from datetime import datetime, timedelta, timezone # Import timezone
from data_loaders.base_fmp_loader import BaseFMPLoader
from config import DEFAULT_TECHNICAL_INDICATORS_SETTINGS
from utils.utils import get_logger, safe_float
from utils.exceptions import DataProcessingError

logger = get_logger(__name__)

class TechnicalAnalyzer:
    """
    Performs simplified technical analysis on stock price data using
    well-known indicators.
    """
    def __init__(self):
        self.fmp_loader = BaseFMPLoader()
        self.default_settings = DEFAULT_TECHNICAL_INDICATORS_SETTINGS

    def _get_historical_data(self, ticker: str, days: int) -> pd.DataFrame | None:
        """
        Fetches historical daily price data for a given ticker for the last 'days'.
        Returns a pandas DataFrame with 'date' and 'close' columns.
        """
        today = datetime.now(timezone.utc) # Fix for SonarLint S6903
        from_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')

        data = self.fmp_loader.get_historical_price(ticker, from_date=from_date, to_date=to_date)
        if not data:
            logger.warning(f"No historical data found for {ticker} from FMP for the last {days} days.")
            return None
        
        # FMP historical data is typically in reverse chronological order, sort to ensure correct order
        df = pd.DataFrame(data)
        if 'date' not in df.columns or 'close' not in df.columns:
            logger.error(f"Missing 'date' or 'close' column in FMP historical data for {ticker}.")
            raise DataProcessingError(f"Missing required columns in historical data for {ticker}.")
        
        df['date'] = pd.to_datetime(df['date'])
        # Ensure 'close' column is strictly numeric (float)
        df['close'] = df['close'].apply(safe_float)
        df['close'] = pd.to_numeric(df['close'], errors='coerce') # Coerce non-numeric to NaN
        df = df.sort_values(by='date').reset_index(drop=True)
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> float | None:
        """Calculates the Relative Strength Index (RSI)."""
        if len(df) < period + 1:
            logger.debug(f"Not enough data for RSI calculation (need {period+1}, got {len(df)}).")
            return None

        # Calculate delta and ensure it's numeric before comparisons
        delta = df['close'].diff()
        delta = pd.to_numeric(delta, errors='coerce').fillna(0.0) # Ensure numeric type and fill NaNs

        # These lines are now safe as delta is guaranteed numeric
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Using .ewm for Wilder's smoothing approximation
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        # Fill any remaining NaNs in avg_gain/avg_loss after EWM before division
        avg_gain = avg_gain.fillna(0.0)
        avg_loss = avg_loss.fillna(0.0)

        # Calculate RS: avg_gain / avg_loss. Handle division by zero.
        # If avg_loss is zero (no downward movement), RS is infinite, meaning RSI is 100.
        # Using np.divide and np.where for explicit handling of division by zero.
        rs_values = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf, dtype=float), where=avg_loss != 0)
        
        # Convert to Pandas Series to use .replace and .fillna
        rs = pd.Series(rs_values, index=df.index)

        # Replace infinite values with 100 (if only gains) and any remaining NaNs with 50 (neutral)
        rs = rs.replace([np.inf, -np.inf], 100.0).fillna(50.0)
        
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Return the last valid RSI value, dropping leading NaNs
        return rsi.dropna().iloc[-1] if not rsi.dropna().empty else None

    def _calculate_macd(self, df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> tuple[float | None, float | None, float | None]:
        """Calculates MACD, Signal Line, and MACD Histogram."""
        if len(df) < slow_period + signal_period:
            logger.debug(f"Not enough data for MACD calculation (need {slow_period+signal_period}, got {len(df)}).")
            return None, None, None

        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd - signal_line
        
        # Ensure latest values are not NaN before returning
        if macd.empty or signal_line.empty or macd_histogram.empty:
             return None, None, None

        # Drop NaNs and get the last valid value for each Series
        return (
            macd.dropna().iloc[-1] if not macd.dropna().empty else None,
            signal_line.dropna().iloc[-1] if not signal_line.dropna().empty else None,
            macd_histogram.dropna().iloc[-1] if not macd_histogram.dropna().empty else None
        )

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int, dev: int) -> tuple[float | None, float | None, float | None]:
        """Calculates Bollinger Bands (Middle, Upper, Lower)."""
        if len(df) < period:
            logger.debug(f"Not enough data for Bollinger Bands calculation (need {period}, got {len(df)}).")
            return None, None, None

        middle_band = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std_dev * dev)
        lower_band = middle_band - (std_dev * dev)
        
        # Ensure latest values are not NaN before returning
        if middle_band.empty or upper_band.empty or lower_band.empty:
            return None, None, None

        # Drop NaNs and get the last valid value for each Series
        return (
            middle_band.dropna().iloc[-1] if not middle_band.dropna().empty else None,
            upper_band.dropna().iloc[-1] if not upper_band.dropna().empty else None,
            lower_band.dropna().iloc[-1] if not lower_band.dropna().empty else None
        )

    def _calculate_moving_averages(self, df: pd.DataFrame, short_period: int, long_period: int) -> tuple[float | None, float | None]:
        """Calculates Short and Long Simple Moving Averages."""
        if len(df) < long_period:
            logger.debug(f"Not enough data for Moving Averages calculation (need {long_period}, got {len(df)}).")
            return None, None
        
        ma_short = df['close'].rolling(window=short_period).mean()
        ma_long = df['close'].rolling(window=long_period).mean()
        
        # Ensure latest values are not NaN before returning
        if ma_short.empty or ma_long.empty:
            return None, None

        # Drop NaNs and get the last valid value for each Series
        return (
            ma_short.dropna().iloc[-1] if not ma_short.dropna().empty else None,
            ma_long.dropna().iloc[-1] if not ma_long.dropna().empty else None
        )

    def _interpret_rsi(self, rsi: float | None, analysis_results: dict):
        """Interprets RSI value and updates analysis_results."""
        if rsi is not None:
            analysis_results["indicators"]["RSI"] = rsi
            if rsi > 70:
                analysis_results["technical_score"] -= 0.2 # Overbought
                analysis_results["reason"] += " RSI overbought."
            elif rsi < 30:
                analysis_results["technical_score"] += 0.2 # Oversold
                analysis_results["reason"] += " RSI oversold."

    def _interpret_macd(self, macd: float | None, signal_line: float | None, macd_hist: float | None, analysis_results: dict):
        """Interprets MACD values and updates analysis_results."""
        if macd is not None and signal_line is not None and macd_hist is not None:
            analysis_results["indicators"]["MACD"] = macd
            analysis_results["indicators"]["MACD_Signal"] = signal_line
            analysis_results["indicators"]["MACD_Hist"] = macd_hist
            if macd > signal_line and macd_hist > 0:
                analysis_results["technical_score"] += 0.2 # Bullish crossover
                analysis_results["reason"] += " MACD bullish crossover."
            elif macd < signal_line and macd_hist < 0:
                analysis_results["technical_score"] -= 0.2 # Bearish crossover
                analysis_results["reason"] += " MACD bearish crossover."

    def _interpret_bollinger_bands(self, latest_close: float, middle_band: float | None, upper_band: float | None, lower_band: float | None, analysis_results: dict):
        """Interprets Bollinger Bands and updates analysis_results."""
        if middle_band is not None and upper_band is not None and lower_band is not None:
            analysis_results["indicators"]["BB_Middle"] = middle_band
            analysis_results["indicators"]["BB_Upper"] = upper_band
            analysis_results["indicators"]["BB_Lower"] = lower_band
            if latest_close > upper_band:
                analysis_results["technical_score"] -= 0.1 # Price above upper band, overbought signal
                analysis_results["reason"] += " Price above BB upper band."
            elif latest_close < lower_band:
                analysis_results["technical_score"] += 0.1 # Price below lower band, oversold signal
                analysis_results["reason"] += " Price below BB lower band."

    def _interpret_moving_averages(self, latest_close: float, ma_short: float | None, ma_long: float | None, analysis_results: dict):
        """Interprets Moving Averages and updates analysis_results."""
        if ma_short is not None and ma_long is not None:
            analysis_results["indicators"]["MA_Short"] = ma_short
            analysis_results["indicators"]["MA_Long"] = ma_long
            if ma_short > ma_long and latest_close > ma_short:
                analysis_results["technical_score"] += 0.3 # Golden cross/strong bullish trend
                analysis_results["reason"] += " Short MA above Long MA (bullish)."
            elif ma_short < ma_long and latest_close < ma_short:
                analysis_results["technical_score"] -= 0.3 # Death cross/strong bearish trend
                analysis_results["reason"] += " Short MA below Long MA (bearish)."

    def analyze(self, ticker: str, settings: dict | None = None) -> dict:
        """
        Performs comprehensive technical analysis for a given ticker.
        Allows dynamic settings override.
        Returns a dictionary of indicator values and a combined technical score/trend.
        """
        current_settings = self.default_settings.copy()
        if settings:
            current_settings.update(settings)
            logger.debug(f"Using custom TA settings for {ticker}: {current_settings}")

        analysis_results = {
            "technical_score": 0.0,
            "trend_direction": "NEUTRAL",
            "reason": "", # Initialize as empty, will be built up
            "indicators": {}
        }
        
        # Ensure a default reason if no signals modify it
        initial_reason = "Technical indicators show a neutral trend."

        # Fetch data based on the longest period required by any indicator
        max_lookback_needed = max(current_settings["RSI_PERIOD"], current_settings["MACD_SLOW_PERIOD"], 
                                  current_settings["BOLLINGER_BANDS_PERIOD"], current_settings["MOVING_AVERAGE_LONG"],
                                  current_settings["DATA_LOOKBACK_DAYS"])
        
        df = self._get_historical_data(ticker, days=max_lookback_needed)
        if df is None or df.empty or len(df) < max_lookback_needed: # Ensure enough data points after fetching
            logger.warning(f"Not enough historical data for {ticker} for technical analysis (need {max_lookback_needed}, got {len(df) if df is not None else 0}).")
            analysis_results["reason"] = f"Not enough historical data for technical analysis (at least {max_lookback_needed} days needed)."
            return analysis_results

        latest_close = df['close'].iloc[-1]
        
        # Calculate and interpret each indicator
        # RSI
        rsi_val = self._calculate_rsi(df, current_settings["RSI_PERIOD"])
        self._interpret_rsi(rsi_val, analysis_results)

        # MACD
        macd_val, signal_line_val, macd_hist_val = self._calculate_macd(
            df, current_settings["MACD_FAST_PERIOD"], 
            current_settings["MACD_SLOW_PERIOD"], 
            current_settings["MACD_SIGNAL_PERIOD"]
        )
        self._interpret_macd(macd_val, signal_line_val, macd_hist_val, analysis_results)

        # Bollinger Bands
        middle_band_val, upper_band_val, lower_band_val = self._calculate_bollinger_bands(
            df, current_settings["BOLLINGER_BANDS_PERIOD"], 
            current_settings["BOLLINGER_BANDS_DEV"]
        )
        self._interpret_bollinger_bands(latest_close, middle_band_val, upper_band_val, lower_band_val, analysis_results)

        # Moving Averages
        ma_short_val, ma_long_val = self._calculate_moving_averages(
            df, current_settings["MOVING_AVERAGE_SHORT"], 
            current_settings["MOVING_AVERAGE_LONG"]
        )
        self._interpret_moving_averages(latest_close, ma_short_val, ma_long_val, analysis_results)


        # Determine overall trend based on score
        if analysis_results["technical_score"] > 0.3:
            analysis_results["trend_direction"] = "BULLISH"
        elif analysis_results["technical_score"] < -0.3:
            analysis_results["trend_direction"] = "BEARISH"
        else:
            analysis_results["trend_direction"] = "NEUTRAL"
            
        # Ensure reason is not empty if no specific signals were strong enough
        if not analysis_results["reason"].strip():
            analysis_results["reason"] = initial_reason
        else:
            analysis_results["reason"] = analysis_results["reason"].strip()


        logger.info(f"Technical analysis for {ticker}: Score={analysis_results['technical_score']:.2f}, Trend={analysis_results['trend_direction']}")
        return analysis_results