"""
Optimized technical analysis with improved calculations and caching
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug


@dataclass
class TechnicalAnalysis:
    """Technical analysis result with comprehensive metrics."""
    symbol: str
    liquidity_score: float
    momentum_score: float
    volatility_score: float
    volume_score: float
    technical_confidence: float
    bid_ask_spread: float
    avg_volume_20d: float
    rsi: float
    price_trend: str
    support_resistance_score: float


class TechnicalAnalyzer:
    """Optimized technical analyzer with improved performance and caching."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized caching strategy."""
        self.fmp_loader = fmp_loader
        self._price_history_cache: Dict[str, pd.DataFrame] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=1)
        self._max_cache_size = 100
    
    def _get_price_history(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data with intelligent caching."""
        cache_key = f"{symbol}_{days}d"
        now = datetime.now()
        
        # Check cache validity
        if (cache_key in self._price_history_cache and 
            cache_key in self._cache_expiry and
            now < self._cache_expiry[cache_key]):
            return self._price_history_cache[cache_key]
        
        try:
            # Fetch new data using the non-cached method
            hist_data = self.fmp_loader.get_historical_data(symbol, days)
            
            if hist_data is not None and not hist_data.empty:
                # Cache management
                self._manage_cache(cache_key, hist_data, now)
                return hist_data
                
        except Exception as e:
            log_error(f"Error fetching price history for {symbol}: {e}")
        
        return None
    
    def _manage_cache(self, cache_key: str, data: pd.DataFrame, timestamp: datetime) -> None:
        """Intelligent cache management with size limits."""
        # Add to cache
        self._price_history_cache[cache_key] = data
        self._cache_expiry[cache_key] = timestamp + self._cache_duration
        
        # Cleanup if cache is too large
        if len(self._price_history_cache) > self._max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(self._cache_expiry.keys(), key=lambda k: self._cache_expiry[k])
            keys_to_remove = sorted_keys[:self._max_cache_size // 4]  # Remove 25%
            
            for key in keys_to_remove:
                self._price_history_cache.pop(key, None)
                self._cache_expiry.pop(key, None)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Optimized RSI calculation using vectorized operations."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            # Ensure prices are numeric and clean
            clean_prices = pd.to_numeric(prices, errors='coerce').dropna()
            if len(clean_prices) < period + 1:
                return 50.0

            # Vectorized RSI calculation with proper typing
            delta = clean_prices.diff().dropna()

            # Ensure delta is numeric and handle comparisons safely
            delta = pd.to_numeric(delta, errors='coerce').fillna(0.0)

            # Safe comparison operations using explicit float comparisons
            gain = delta.where(delta > 0.0, 0.0)
            loss = (-delta).where(delta < 0.0, 0.0)
            
            # Use exponential moving average for better responsiveness
            alpha = 2.0 / (period + 1)
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            log_debug(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Optimized moving average calculations."""
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
    
    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Optimized volatility calculation with proper normalization."""
        try:
            if len(prices) < 2:
                return 0.5
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return 0.5
            
            # Rolling volatility if enough data
            if len(returns) >= period:
                vol = returns.rolling(period).std().iloc[-1]
            else:
                vol = returns.std()
            
            # Annualize and normalize (assume 252 trading days)
            if not pd.isna(vol):
                annualized_vol = vol * np.sqrt(252)
                # Normalize to 0-1 scale (50% annual vol = 1.0)
                normalized_vol = min(annualized_vol / 0.5, 1.0)
                return max(0.0, normalized_vol)
            
            return 0.5
            
        except Exception as e:
            log_debug(f"Error calculating volatility: {e}")
            return 0.5
    
    def _detect_trend(self, prices: pd.Series, mas: Dict[str, float]) -> str:
        """Improved trend detection with multiple timeframes."""
        try:
            if len(prices) < 5:
                return 'sideways'
            
            current_price = float(prices.iloc[-1])
            
            # Safely get moving averages with proper fallbacks
            sma_5 = mas.get('sma_5')
            sma_10 = mas.get('sma_10')
            sma_20 = mas.get('sma_20')
            
            # Ensure all values are valid floats, use current_price as fallback
            if sma_5 is None or pd.isna(sma_5):
                sma_5 = current_price
            else:
                sma_5 = float(sma_5)
                
            if sma_10 is None or pd.isna(sma_10):
                sma_10 = current_price
            else:
                sma_10 = float(sma_10)
                
            if sma_20 is None or pd.isna(sma_20):
                sma_20 = current_price
            else:
                sma_20 = float(sma_20)
            
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
    
    def _calculate_support_resistance_score(self, prices: pd.Series) -> float:
        """Optimized support/resistance calculation using vectorized operations."""
        try:
            if len(prices) < 10:
                return 0.5
            
            current_price = prices.iloc[-1]
            recent_prices = prices.tail(20)
            
            # Vectorized calculation of nearby levels
            tolerance = 0.02
            price_diffs = np.abs(recent_prices - current_price) / current_price
            nearby_count = np.sum(price_diffs <= tolerance)
            
            # Normalize the score
            level_strength = nearby_count / len(recent_prices)
            return min(level_strength * 2, 1.0)
            
        except Exception as e:
            log_debug(f"Error calculating support/resistance: {e}")
            return 0.5
    
    def analyze_symbol(self, symbol: str, current_price_data: pd.Series) -> Optional[TechnicalAnalysis]:
        """Comprehensive technical analysis with optimized calculations."""
        try:
            # Extract current market data safely
            current_price = float(current_price_data.get('lastSalePrice', 0))
            current_volume = float(current_price_data.get('volume', 0))
            bid_price = float(current_price_data.get('bidPrice', 0))
            ask_price = float(current_price_data.get('askPrice', 0))
            
            if current_price <= 0:
                return None
            
            # Calculate bid-ask spread
            if bid_price > 0 and ask_price > 0 and bid_price < ask_price:
                bid_ask_spread = (ask_price - bid_price) / bid_price
            else:
                bid_ask_spread = 0.05  # Default assumption
            
            # Get historical data
            hist_df = self._get_price_history(symbol)
            
            if hist_df is None or hist_df.empty:
                # Return minimal analysis with current data only
                return self._create_minimal_analysis(symbol, current_price, current_volume, bid_ask_spread)
            
            # Perform comprehensive technical analysis
            return self._perform_full_analysis(symbol, hist_df, current_price, current_volume, bid_ask_spread)
            
        except Exception as e:
            log_error(f"Error in technical analysis for {symbol}: {e}")
            return None
    
    def _create_minimal_analysis(self, symbol: str, current_price: float, 
                                current_volume: float, bid_ask_spread: float) -> TechnicalAnalysis:
        """Create minimal analysis when historical data is unavailable."""
        return TechnicalAnalysis(
            symbol=symbol,
            liquidity_score=0.3,
            momentum_score=0.0,
            volatility_score=0.5,
            volume_score=0.5,
            technical_confidence=0.2,
            bid_ask_spread=bid_ask_spread,
            avg_volume_20d=current_volume,
            rsi=50.0,
            price_trend='sideways',
            support_resistance_score=0.5
        )
    
    def _perform_full_analysis(self, symbol: str, hist_df: pd.DataFrame, 
                              current_price: float, current_volume: float, 
                              bid_ask_spread: float) -> TechnicalAnalysis:
        """Perform comprehensive technical analysis with historical data."""
        close_prices = hist_df['close']
        volumes = hist_df['volume']
        
        # Technical calculations with error handling
        rsi = self._calculate_rsi(close_prices)
        mas = self._calculate_moving_averages(close_prices)
        volatility = self._calculate_volatility(close_prices)
        price_trend = self._detect_trend(close_prices, mas)
        sr_score = self._calculate_support_resistance_score(close_prices)
        
        # Volume analysis with improved calculations
        avg_volume_20d = volumes.tail(20).mean() if len(volumes) >= 20 else current_volume
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
        volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize
        
        # Liquidity score calculation
        liquidity_score = self._calculate_liquidity_score(bid_ask_spread, current_volume, avg_volume_20d)
        
        # Momentum score calculation
        momentum_score = self._calculate_momentum_score(rsi, price_trend, mas, current_price)
        
        # Overall technical confidence
        technical_confidence = self._calculate_technical_confidence(
            liquidity_score, momentum_score, volume_score, volatility, sr_score
        )
        
        return TechnicalAnalysis(
            symbol=symbol,
            liquidity_score=liquidity_score,
            momentum_score=momentum_score,
            volatility_score=volatility,
            volume_score=volume_score,
            technical_confidence=technical_confidence,
            bid_ask_spread=bid_ask_spread,
            avg_volume_20d=avg_volume_20d,
            rsi=rsi,
            price_trend=price_trend,
            support_resistance_score=sr_score
        )
    
    def _calculate_liquidity_score(self, spread: float, current_vol: float, avg_vol: float) -> float:
        """Optimized liquidity score calculation."""
        # Spread component (tighter spreads = higher liquidity)
        spread_score = max(0, 1 - (spread / 0.05))  # 5% spread = 0 score
        
        # Volume component (relative to average)
        vol_ratio = current_vol / max(avg_vol, 1) if avg_vol > 0 else 1.0
        vol_score = min(vol_ratio / 2.0, 1.0)  # Cap at 2x average
        
        # Minimum volume threshold
        min_vol_score = 1.0 if current_vol >= CONFIG.min_volume else 0.5
        
        # Weighted combination
        liquidity = (spread_score * 0.4 + vol_score * 0.4 + min_vol_score * 0.2)
        return max(0.0, min(1.0, liquidity))
    
    def _calculate_momentum_score(self, rsi: float, trend: str, mas: Dict[str, float], current_price: float) -> float:
        """Enhanced momentum score calculation."""
        try:
            # RSI component (-1 to 1 scale)
            rsi_normalized = (rsi - 50) / 50
            rsi_score = max(-1, min(1, rsi_normalized))
            
            # Trend component
            trend_scores = {'uptrend': 0.5, 'sideways': 0, 'downtrend': -0.5}
            trend_score = trend_scores.get(trend, 0)
            
            # Moving average alignment score
            ma_score = 0
            sma_20 = mas.get('sma_20', current_price)
            sma_5 = mas.get('sma_5', current_price)
            ema_12 = mas.get('ema_12', current_price)
            ema_26 = mas.get('ema_26', current_price)
            
            # Price vs MA
            if current_price > sma_20:
                ma_score += 0.3
            elif current_price < sma_20:
                ma_score -= 0.3
            
            # Short vs long MA
            if sma_5 > sma_20:
                ma_score += 0.2
            elif sma_5 < sma_20:
                ma_score -= 0.2
            
            # EMA alignment
            if ema_12 > ema_26:
                ma_score += 0.2
            elif ema_12 < ema_26:
                ma_score -= 0.2
            
            # Combine all components
            momentum = (rsi_score * 0.4 + trend_score * 0.3 + ma_score * 0.3)
            return max(-1, min(1, momentum))
            
        except Exception as e:
            log_debug(f"Error calculating momentum score: {e}")
            return 0.0
    
    def _calculate_technical_confidence(self, liquidity: float, momentum: float, 
                                      volume: float, volatility: float, sr_score: float) -> float:
        """Calculate overall technical confidence with optimized weighting."""
        try:
            # Component scoring
            liquidity_component = liquidity * 0.3          # High liquidity is essential
            momentum_component = abs(momentum) * 0.25      # Strong momentum (either direction)
            volume_component = volume * 0.2                # High volume confirms moves
            volatility_component = (1 - volatility) * 0.15 # Lower volatility for better entries
            sr_component = sr_score * 0.1                  # Support/resistance strength
            
            confidence = (
                liquidity_component + momentum_component + volume_component + 
                volatility_component + sr_component
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            log_debug(f"Error calculating technical confidence: {e}")
            return 0.3
    
    def filter_trades_by_technical(self, symbols_with_prices: List[Tuple[str, pd.Series]], 
                                 min_technical_confidence: float = 0.5) -> List[Tuple[str, TechnicalAnalysis]]:
        """Efficiently filter symbols based on technical analysis."""
        results = []
        
        for symbol, price_data in symbols_with_prices:
            try:
                tech_analysis = self.analyze_symbol(symbol, price_data)
                
                if tech_analysis and tech_analysis.technical_confidence >= min_technical_confidence:
                    results.append((symbol, tech_analysis))
                    log_debug(f"Technical PASS: {symbol} "
                             f"conf={tech_analysis.technical_confidence:.2f} "
                             f"liq={tech_analysis.liquidity_score:.2f} "
                             f"mom={tech_analysis.momentum_score:.2f}")
                elif tech_analysis:
                    log_debug(f"Technical FAIL: {symbol} "
                             f"conf={tech_analysis.technical_confidence:.2f} "
                             f"(threshold: {min_technical_confidence})")
                             
            except Exception as e:
                log_debug(f"Error filtering {symbol}: {e}")
                continue
        
        return results