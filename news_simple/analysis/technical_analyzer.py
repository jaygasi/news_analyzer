"""
Technical analysis for trade validation
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
    """Technical analysis result"""
    symbol: str
    liquidity_score: float      # 0-1 (higher = more liquid)
    momentum_score: float       # -1 to 1 (positive = bullish momentum)
    volatility_score: float     # 0-1 (higher = more volatile)
    volume_score: float         # 0-1 (higher = above average volume)
    technical_confidence: float # 0-1 (overall technical confidence)
    bid_ask_spread: float
    avg_volume_20d: float
    rsi: float
    price_trend: str           # 'uptrend', 'downtrend', 'sideways'
    support_resistance_score: float

class TechnicalAnalyzer:
    """Technical analyzer for trade validation"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
        self.price_history_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=1)
    
    def _get_price_history(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data with caching"""
        cache_key = f"{symbol}_{days}d"
        now = datetime.now()
        
        # Check cache first
        if (cache_key in self.price_history_cache and 
            cache_key in self.cache_expiry and
            now < self.cache_expiry[cache_key]):
            return self.price_history_cache[cache_key]
        
        try:
            # Calculate date range
            end_date = now.strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Get historical data from FMP
            data = self.fmp_loader._make_request(f"historical-price-full/{symbol}", {
                'from': start_date,
                'to': end_date
            })
            
            if data and 'historical' in data:
                df = pd.DataFrame(data['historical'])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    df.set_index('date', inplace=True)
                    
                    # Cache the result
                    self.price_history_cache[cache_key] = df
                    self.cache_expiry[cache_key] = now + self.cache_duration
                    
                    return df
        except Exception as e:
            log_error(f"Error fetching price history for {symbol}: {e}")
        
        return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate key moving averages"""
        try:
            return {
                'sma_5': prices.rolling(5).mean().iloc[-1] if len(prices) >= 5 else prices.iloc[-1],
                'sma_10': prices.rolling(10).mean().iloc[-1] if len(prices) >= 10 else prices.iloc[-1],
                'sma_20': prices.rolling(20).mean().iloc[-1] if len(prices) >= 20 else prices.iloc[-1],
                'ema_12': prices.ewm(span=12).mean().iloc[-1] if len(prices) >= 12 else prices.iloc[-1],
                'ema_26': prices.ewm(span=26).mean().iloc[-1] if len(prices) >= 26 else prices.iloc[-1]
            }
        except:
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            return {
                'sma_5': current_price, 'sma_10': current_price, 'sma_20': current_price,
                'ema_12': current_price, 'ema_26': current_price
            }
    
    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        try:
            if len(prices) < 2:
                return 0.5  # Medium volatility default
            
            returns = prices.pct_change().dropna()
            if len(returns) < period:
                volatility = returns.std()
            else:
                volatility = returns.rolling(period).std().iloc[-1]
            
            # Normalize to 0-1 scale (assume 50% annualized vol = 1.0)
            normalized_vol = min(volatility * np.sqrt(252) / 0.5, 1.0)
            return float(normalized_vol) if not pd.isna(normalized_vol) else 0.5
        except:
            return 0.5
    
    def _detect_trend(self, prices: pd.Series, mas: Dict[str, float]) -> str:
        """Detect price trend using moving averages"""
        try:
            current_price = prices.iloc[-1]
            sma_20 = mas['sma_20']
            sma_5 = mas['sma_5']
            
            # Simple trend detection
            if current_price > sma_20 and sma_5 > sma_20:
                return 'uptrend'
            elif current_price < sma_20 and sma_5 < sma_20:
                return 'downtrend'
            else:
                return 'sideways'
        except:
            return 'sideways'
    
    def _calculate_support_resistance_score(self, prices: pd.Series) -> float:
        """Calculate support/resistance strength score"""
        try:
            if len(prices) < 10:
                return 0.5
            
            current_price = prices.iloc[-1]
            recent_prices = prices.tail(20)
            
            # Find nearby price levels (within 2%)
            tolerance = 0.02
            nearby_levels = recent_prices[
                abs(recent_prices - current_price) / current_price <= tolerance
            ]
            
            # More touches = stronger level
            level_strength = len(nearby_levels) / len(recent_prices)
            return min(level_strength * 2, 1.0)  # Scale to 0-1
        except:
            return 0.5
    
    def analyze_symbol(self, symbol: str, current_price_data: pd.Series) -> Optional[TechnicalAnalysis]:
        """Perform comprehensive technical analysis on a symbol"""
        try:
            # Extract current market data
            current_price = current_price_data.get('lastSalePrice', 0)
            current_volume = current_price_data.get('volume', 0)
            bid_price = current_price_data.get('bidPrice', 0)
            ask_price = current_price_data.get('askPrice', 0)
            
            if current_price <= 0:
                return None
            
            # Calculate bid-ask spread
            if bid_price > 0 and ask_price > 0:
                bid_ask_spread = (ask_price - bid_price) / bid_price
            else:
                bid_ask_spread = 0.05  # Assume 5% if not available
            
            # Get historical data
            hist_df = self._get_price_history(symbol)
            if hist_df is None or hist_df.empty:
                # Create minimal analysis with current data only
                return TechnicalAnalysis(
                    symbol=symbol,
                    liquidity_score=0.3,  # Low due to no historical data
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
            
            # Technical calculations
            close_prices = hist_df['close']
            volumes = hist_df['volume']
            
            # RSI calculation
            rsi = self._calculate_rsi(close_prices)
            
            # Moving averages
            mas = self._calculate_moving_averages(close_prices)
            
            # Volatility
            volatility = self._calculate_volatility(close_prices)
            
            # Trend detection
            price_trend = self._detect_trend(close_prices, mas)
            
            # Support/resistance
            sr_score = self._calculate_support_resistance_score(close_prices)
            
            # Volume analysis
            avg_volume_20d = volumes.tail(20).mean() if len(volumes) >= 20 else current_volume
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
            volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize volume surge
            
            # Liquidity score (based on spread and volume)
            liquidity_score = self._calculate_liquidity_score(bid_ask_spread, current_volume, avg_volume_20d)
            
            # Momentum score (combining RSI and trend)
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
            
        except Exception as e:
            log_error(f"Error in technical analysis for {symbol}: {e}")
            return None
    
    def _calculate_liquidity_score(self, spread: float, current_vol: float, avg_vol: float) -> float:
        """Calculate liquidity score based on spread and volume"""
        # Spread component (lower spread = higher liquidity)
        spread_score = max(0, 1 - (spread / 0.05))  # 5% spread = 0 score
        
        # Volume component
        vol_score = min(current_vol / max(avg_vol, 1), 2.0) / 2.0  # Cap at 2x average
        
        # Minimum volume threshold
        min_volume_score = 1.0 if current_vol >= CONFIG.min_volume else 0.5
        
        return (spread_score * 0.4 + vol_score * 0.4 + min_volume_score * 0.2)
    
    def _calculate_momentum_score(self, rsi: float, trend: str, mas: Dict, current_price: float) -> float:
        """Calculate momentum score"""
        # RSI component (-1 to 1)
        rsi_score = (rsi - 50) / 50  # Convert 0-100 RSI to -1 to 1
        
        # Trend component
        trend_score = {'uptrend': 0.5, 'sideways': 0, 'downtrend': -0.5}[trend]
        
        # Moving average alignment
        ma_score = 0
        if current_price > mas['sma_20']:
            ma_score += 0.3
        if mas['sma_5'] > mas['sma_20']:
            ma_score += 0.2
        if mas['ema_12'] > mas['ema_26']:
            ma_score += 0.2
        
        ma_score = ma_score - 0.35  # Center around 0
        
        # Combine components
        momentum = (rsi_score * 0.4 + trend_score * 0.3 + ma_score * 0.3)
        return max(-1, min(1, momentum))  # Clamp to -1, 1
    
    def _calculate_technical_confidence(self, liquidity: float, momentum: float, 
                                      volume: float, volatility: float, sr_score: float) -> float:
        """Calculate overall technical confidence"""
        # Prefer high liquidity, moderate volatility, strong volume
        confidence = (
            liquidity * 0.3 +           # High liquidity is good
            abs(momentum) * 0.25 +      # Strong momentum (either direction)
            volume * 0.2 +              # High volume is good
            (1 - volatility) * 0.15 +   # Lower volatility is better for entry
            sr_score * 0.1              # Strong S/R levels
        )
        
        return min(confidence, 1.0)
    
    def filter_trades_by_technical(self, symbols_with_prices: List[Tuple[str, pd.Series]], 
                                 min_technical_confidence: float = 0.5) -> List[Tuple[str, TechnicalAnalysis]]:
        """Filter symbols based on technical analysis"""
        results = []
        
        for symbol, price_data in symbols_with_prices:
            tech_analysis = self.analyze_symbol(symbol, price_data)
            
            if tech_analysis and tech_analysis.technical_confidence >= min_technical_confidence:
                results.append((symbol, tech_analysis))
                log_debug(f"Technical PASS: {symbol} confidence={tech_analysis.technical_confidence:.2f} "
                         f"liquidity={tech_analysis.liquidity_score:.2f} "
                         f"momentum={tech_analysis.momentum_score:.2f}")
            elif tech_analysis:
                log_debug(f"Technical FAIL: {symbol} confidence={tech_analysis.technical_confidence:.2f} "
                         f"(threshold: {min_technical_confidence})")
        
        return results