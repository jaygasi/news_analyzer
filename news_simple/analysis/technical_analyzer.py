"""
Optimized technical analysis with modular components
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug
from .technical.indicators import TechnicalIndicators
from .technical.trend_analyzer import TrendAnalyzer
from .technical.momentum_calculator import MomentumCalculator
from .technical.liquidity_analyzer import LiquidityAnalyzer


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
    """Optimized technical analyzer using modular components."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized caching strategy."""
        self.fmp_loader = fmp_loader
        self._price_history_cache: Dict[str, pd.DataFrame] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=1)
        self._max_cache_size = 100
        
        # Initialize component analyzers
        self.indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
        self.momentum_calculator = MomentumCalculator()
        self.liquidity_analyzer = LiquidityAnalyzer()
    
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
            hist_data = self.fmp_loader.get_historical_data(symbol, days)
            
            if hist_data is not None and not hist_data.empty:
                self._manage_cache(cache_key, hist_data, now)
                return hist_data
                
        except Exception as e:
            log_error(f"Error fetching price history for {symbol}: {e}")
        
        return None
    
    def _manage_cache(self, cache_key: str, data: pd.DataFrame, timestamp: datetime) -> None:
        """Intelligent cache management with size limits."""
        self._price_history_cache[cache_key] = data
        self._cache_expiry[cache_key] = timestamp + self._cache_duration
        
        if len(self._price_history_cache) > self._max_cache_size:
            sorted_keys = sorted(self._cache_expiry.keys(), key=lambda k: self._cache_expiry[k])
            keys_to_remove = sorted_keys[:self._max_cache_size // 4]
            
            for key in keys_to_remove:
                self._price_history_cache.pop(key, None)
                self._cache_expiry.pop(key, None)
    
    def analyze_symbol(self, symbol: str, current_price_data: pd.Series) -> Optional[TechnicalAnalysis]:
        """Comprehensive technical analysis using modular components."""
        try:
            # Extract current market data
            current_price = float(current_price_data.get('lastSalePrice', 0))
            current_volume = float(current_price_data.get('volume', 0))
            bid_price = float(current_price_data.get('bidPrice', 0))
            ask_price = float(current_price_data.get('askPrice', 0))
            
            if current_price <= 0:
                return None
            
            # Calculate bid-ask spread
            bid_ask_spread = self.liquidity_analyzer.calculate_bid_ask_spread(bid_price, ask_price)
            
            # Get historical data
            hist_df = self._get_price_history(symbol)
            
            if hist_df is None or hist_df.empty:
                return self._create_minimal_analysis(symbol, current_price, current_volume, bid_ask_spread)
            
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
        # Add all technical indicators
        hist_df = self.indicators.add_all_indicators(hist_df)
        
        close_prices = hist_df['close']
        volumes = hist_df['volume']
        
        # Calculate core indicators
        rsi = self.indicators.calculate_rsi(close_prices)
        mas = self.indicators.calculate_moving_averages(close_prices)
        volatility = self.indicators.calculate_volatility(close_prices)
        
        # Trend analysis
        price_trend = self.trend_analyzer.detect_trend(close_prices, mas)
        sr_score = self.trend_analyzer.calculate_support_resistance_score(close_prices, current_price)
        
        # Volume analysis
        volume_profile = self.liquidity_analyzer.analyze_volume_profile(volumes)
        avg_volume_20d = volume_profile['average_volume']
        volume_score = volume_profile['volume_score']
        
        # Liquidity analysis
        liquidity_score = self.liquidity_analyzer.calculate_liquidity_score(
            bid_ask_spread, current_volume, avg_volume_20d
        )
        
        # Momentum analysis
        momentum_score = self.momentum_calculator.calculate_momentum_score(
            rsi, price_trend, mas, current_price
        )
        
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
    
    def _calculate_technical_confidence(self, liquidity: float, momentum: float, 
                                      volume: float, volatility: float, sr_score: float) -> float:
        """Calculate overall technical confidence with optimized weighting."""
        try:
            # Component weights
            weights = (0.3, 0.25, 0.2, 0.15, 0.1)
            
            confidence = (
                liquidity * weights[0] +
                abs(momentum) * weights[1] +
                volume * weights[2] +
                (1 - volatility) * weights[3] +
                sr_score * weights[4]
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            log_debug(f"Error calculating technical confidence: {e}")
            return 0.3
    
    def filter_trades_by_technical(self, symbols_with_prices: List[Tuple[str, pd.Series]], 
                                 min_technical_confidence: float = 0.5) -> List[Tuple[str, TechnicalAnalysis]]:
        """Filter symbols based on technical analysis criteria."""
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