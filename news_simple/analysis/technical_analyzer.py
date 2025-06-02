"""
Optimized technical analyzer with improved caching and performance
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_debug
from .technical.indicators import TechnicalIndicators
from .technical.trend_analyzer import TrendAnalyzer
from .technical.momentum_calculator import MomentumCalculator
from .technical.liquidity_analyzer import LiquidityAnalyzer


@dataclass
class TechnicalAnalysis:
    """Comprehensive technical analysis result with optimized data structure."""
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
    
    # Additional technical metrics
    trend_strength: float = 0.5
    price_momentum: float = 0.0
    volume_momentum: float = 1.0


class TechnicalAnalyzer:
    """High-performance technical analyzer with advanced caching and optimization."""
    
    # Class-level constants for performance
    CACHE_DURATION_MINUTES = 45
    MAX_CACHE_SIZE = 200
    DEFAULT_HISTORY_DAYS = 60
    
    # Pre-computed confidence weights
    CONFIDENCE_WEIGHTS = {
        'liquidity': 0.30,
        'momentum': 0.25,
        'volume': 0.20,
        'volatility': 0.15,
        'support_resistance': 0.10
    }
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized caching and component setup."""
        self.fmp_loader = fmp_loader
        
        # Enhanced caching system
        self._price_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._analysis_cache: Dict[str, Tuple[TechnicalAnalysis, datetime]] = {}
        
        # Initialize technical components
        self.indicators = TechnicalIndicators()
        self.trend_analyzer = TrendAnalyzer()
        self.momentum_calculator = MomentumCalculator()
        self.liquidity_analyzer = LiquidityAnalyzer()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
    
    def analyze_symbol(self, symbol: str, current_price_data: pd.Series) -> Optional[TechnicalAnalysis]:
        """Optimized technical analysis with intelligent caching."""
        try:
            # Check analysis cache first
            cached_analysis = self._get_cached_analysis(symbol)
            if cached_analysis:
                self.cache_hits += 1
                return cached_analysis
            
            self.cache_misses += 1
            
            # Extract current market data efficiently
            market_data = self._extract_market_data(current_price_data)
            if not market_data['valid']:
                return None
            
            # Get historical data with caching
            historical_df = self._get_cached_price_data(symbol)
            
            if historical_df is None or historical_df.empty:
                return self._create_minimal_analysis(symbol, market_data)
            
            # Perform comprehensive analysis
            analysis = self._perform_comprehensive_analysis(symbol, historical_df, market_data)
            
            # Cache the result
            if analysis:
                self._cache_analysis(symbol, analysis)
            
            return analysis
            
        except Exception as e:
            log_error(f"Technical analysis error for {symbol}: {e}")
            return None
    
    def _get_cached_analysis(self, symbol: str) -> Optional[TechnicalAnalysis]:
        """Check for valid cached analysis."""
        if symbol not in self._analysis_cache:
            return None
        
        analysis, timestamp = self._analysis_cache[symbol]
        age_minutes = (datetime.now() - timestamp).total_seconds() / 60
        
        if age_minutes < self.CACHE_DURATION_MINUTES:
            return analysis
        
        # Remove stale cache entry
        del self._analysis_cache[symbol]
        return None
    
    def _extract_market_data(self, price_data: pd.Series) -> Dict:
        """Extract and validate current market data."""
        try:
            current_price = float(price_data.get('lastSalePrice', 0))
            current_volume = float(price_data.get('volume', 0))
            bid_price = float(price_data.get('bidPrice', 0))
            ask_price = float(price_data.get('askPrice', 0))
            
            # Validate data
            if current_price <= 0:
                return {'valid': False}
            
            # Calculate bid-ask spread
            bid_ask_spread = self.liquidity_analyzer.calculate_bid_ask_spread(bid_price, ask_price)
            
            return {
                'valid': True,
                'current_price': current_price,
                'current_volume': current_volume,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'bid_ask_spread': bid_ask_spread
            }
            
        except (ValueError, TypeError) as e:
            log_debug(f"Error extracting market data: {e}")
            return {'valid': False}
    
    def _get_cached_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical price data with intelligent caching."""
        current_time = datetime.now()
        
        # Check cache validity
        if symbol in self._price_cache:
            cached_df, cache_time = self._price_cache[symbol]
            age_minutes = (current_time - cache_time).total_seconds() / 60
            
            if age_minutes < self.CACHE_DURATION_MINUTES:
                return cached_df
        
        # Fetch new data
        try:
            historical_df = self.fmp_loader.get_historical_data(symbol, self.DEFAULT_HISTORY_DAYS)
            
            if historical_df is not None and not historical_df.empty:
                # Add technical indicators
                historical_df = self.indicators.add_all_indicators(historical_df)
                
                # Cache the result
                self._price_cache[symbol] = (historical_df, current_time)
                
                # Manage cache size
                self._cleanup_price_cache()
                
                return historical_df
                
        except Exception as e:
            log_debug(f"Error fetching price data for {symbol}: {e}")
        
        return None
    
    def _cleanup_price_cache(self) -> None:
        """Clean up old cache entries to manage memory."""
        if len(self._price_cache) <= self.MAX_CACHE_SIZE:
            return
        
        # Remove oldest entries
        sorted_cache = sorted(
            self._price_cache.items(),
            key=lambda x: x[1][1]  # Sort by timestamp
        )
        
        # Keep most recent entries
        keep_count = int(self.MAX_CACHE_SIZE * 0.8)
        self._price_cache = dict(sorted_cache[-keep_count:])
    
    def _create_minimal_analysis(self, symbol: str, market_data: Dict) -> TechnicalAnalysis:
        """Create minimal analysis when historical data is unavailable."""
        return TechnicalAnalysis(
            symbol=symbol,
            liquidity_score=0.4,
            momentum_score=0.0,
            volatility_score=0.5,
            volume_score=0.5,
            technical_confidence=0.25,
            bid_ask_spread=market_data['bid_ask_spread'],
            avg_volume_20d=market_data['current_volume'],
            rsi=50.0,
            price_trend='sideways',
            support_resistance_score=0.5,
            trend_strength=0.5,
            price_momentum=0.0,
            volume_momentum=1.0
        )
    
    def _perform_comprehensive_analysis(self, symbol: str, historical_df: pd.DataFrame, 
                                      market_data: Dict) -> TechnicalAnalysis:
        """Perform comprehensive technical analysis with optimized calculations."""
        try:
            # Extract core data series
            close_prices = historical_df['close']
            volumes = historical_df['volume']
            current_price = market_data['current_price']
            current_volume = market_data['current_volume']
            
            # Calculate core indicators efficiently
            analysis_data = self._calculate_core_indicators(
                close_prices, volumes, current_price, current_volume
            )
            
            # Advanced analysis components
            trend_data = self._analyze_trend_components(historical_df, close_prices, analysis_data['mas'])
            momentum_data = self._analyze_momentum_components(close_prices, volumes, analysis_data['rsi'])
            liquidity_data = self._analyze_liquidity_components(
                market_data, current_volume, analysis_data['avg_volume_20d']
            )
            
            # Calculate technical confidence
            technical_confidence = self._calculate_optimized_confidence(
                liquidity_data['liquidity_score'],
                momentum_data['momentum_score'],
                liquidity_data['volume_score'],
                analysis_data['volatility'],
                trend_data['support_resistance_score']
            )
            
            return TechnicalAnalysis(
                symbol=symbol,
                liquidity_score=liquidity_data['liquidity_score'],
                momentum_score=momentum_data['momentum_score'],
                volatility_score=analysis_data['volatility'],
                volume_score=liquidity_data['volume_score'],
                technical_confidence=technical_confidence,
                bid_ask_spread=market_data['bid_ask_spread'],
                avg_volume_20d=analysis_data['avg_volume_20d'],
                rsi=analysis_data['rsi'],
                price_trend=trend_data['price_trend'],
                support_resistance_score=trend_data['support_resistance_score'],
                trend_strength=trend_data['trend_strength'],
                price_momentum=momentum_data['price_momentum'],
                volume_momentum=momentum_data['volume_momentum']
            )
            
        except Exception as e:
            log_error(f"Error in comprehensive analysis for {symbol}: {e}")
            return self._create_minimal_analysis(symbol, market_data)
    
    def _calculate_core_indicators(self, close_prices: pd.Series, volumes: pd.Series,
                                 current_price: float, current_volume: float) -> Dict:
        """Calculate core technical indicators efficiently."""
        try:
            # Core calculations
            rsi = self.indicators.calculate_rsi(close_prices)
            mas = self.indicators.calculate_moving_averages(close_prices)
            volatility = self.indicators.calculate_volatility(close_prices)
            
            # Volume analysis
            volume_profile = self.liquidity_analyzer.analyze_volume_profile(volumes)
            avg_volume_20d = volume_profile['average_volume']
            
            return {
                'rsi': rsi,
                'mas': mas,
                'volatility': volatility,
                'avg_volume_20d': avg_volume_20d
            }
            
        except Exception as e:
            log_debug(f"Error calculating core indicators: {e}")
            return {
                'rsi': 50.0,
                'mas': {},
                'volatility': 0.5,
                'avg_volume_20d': current_volume
            }
    
    def _analyze_trend_components(self, historical_df: pd.DataFrame, close_prices: pd.Series, 
                                mas: Dict) -> Dict:
        """Analyze trend-related components."""
        try:
            current_price = float(close_prices.iloc[-1])
            
            # Trend detection
            price_trend = self.trend_analyzer.detect_trend(close_prices, mas)
            
            # Trend strength
            trend_strength = self.trend_analyzer.calculate_trend_strength(close_prices, price_trend)
            
            # Support/resistance analysis
            sr_score = self.trend_analyzer.calculate_support_resistance_score(close_prices, current_price)
            
            return {
                'price_trend': price_trend,
                'trend_strength': trend_strength,
                'support_resistance_score': sr_score
            }
            
        except Exception as e:
            log_debug(f"Error analyzing trend components: {e}")
            return {
                'price_trend': 'sideways',
                'trend_strength': 0.5,
                'support_resistance_score': 0.5
            }
    
    def _analyze_momentum_components(self, close_prices: pd.Series, volumes: pd.Series, 
                                   rsi: float) -> Dict:
        """Analyze momentum-related components."""
        try:
            current_price = float(close_prices.iloc[-1])
            
            # Price momentum calculation
            price_momentum_data = self.momentum_calculator.calculate_price_momentum(close_prices)
            price_momentum = price_momentum_data.get('momentum_5d', 0.0)
            
            # Volume momentum
            volume_momentum = self.momentum_calculator.calculate_volume_momentum(volumes)
            
            # Overall momentum score
            momentum_score = self.momentum_calculator.calculate_momentum_score(
                rsi, 'sideways', {}, current_price  # Simplified for performance
            )
            
            return {
                'momentum_score': momentum_score,
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum
            }
            
        except Exception as e:
            log_debug(f"Error analyzing momentum components: {e}")
            return {
                'momentum_score': 0.0,
                'price_momentum': 0.0,
                'volume_momentum': 1.0
            }
    
    def _analyze_liquidity_components(self, market_data: Dict, current_volume: float,
                                    avg_volume_20d: float) -> Dict:
        """Analyze liquidity-related components."""
        try:
            # Liquidity score
            liquidity_score = self.liquidity_analyzer.calculate_liquidity_score(
                market_data['bid_ask_spread'], current_volume, avg_volume_20d
            )
            
            # Volume score
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
            volume_score = min(volume_ratio / 2.0, 1.0)
            
            return {
                'liquidity_score': liquidity_score,
                'volume_score': volume_score
            }
            
        except Exception as e:
            log_debug(f"Error analyzing liquidity components: {e}")
            return {
                'liquidity_score': 0.4,
                'volume_score': 0.5
            }
    
    def _calculate_optimized_confidence(self, liquidity: float, momentum: float, 
                                      volume: float, volatility: float, sr_score: float) -> float:
        """Calculate technical confidence using pre-computed weights."""
        try:
            weights = self.CONFIDENCE_WEIGHTS
            
            # Vectorized confidence calculation
            components = [
                liquidity * weights['liquidity'],
                abs(momentum) * weights['momentum'],
                volume * weights['volume'],
                (1 - volatility) * weights['volatility'],
                sr_score * weights['support_resistance']
            ]
            
            confidence = sum(components)
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            log_debug(f"Error calculating confidence: {e}")
            return 0.3
    
    def _cache_analysis(self, symbol: str, analysis: TechnicalAnalysis) -> None:
        """Cache analysis result with memory management."""
        current_time = datetime.now()
        self._analysis_cache[symbol] = (analysis, current_time)
        
        # Manage cache size
        if len(self._analysis_cache) > self.MAX_CACHE_SIZE:
            # Remove oldest entries
            sorted_cache = sorted(
                self._analysis_cache.items(),
                key=lambda x: x[1][1]
            )
            keep_count = int(self.MAX_CACHE_SIZE * 0.8)
            self._analysis_cache = dict(sorted_cache[-keep_count:])
    
    def filter_trades_by_technical(self, symbols_with_prices: List[Tuple[str, pd.Series]], 
                                 min_technical_confidence: float = 0.5) -> List[Tuple[str, TechnicalAnalysis]]:
        """Optimized technical filtering with batch processing."""
        results = []
        
        # Process in batches for better performance
        batch_size = 20
        for i in range(0, len(symbols_with_prices), batch_size):
            batch = symbols_with_prices[i:i + batch_size]
            
            for symbol, price_data in batch:
                try:
                    tech_analysis = self.analyze_symbol(symbol, price_data)
                    
                    if tech_analysis and tech_analysis.technical_confidence >= min_technical_confidence:
                        results.append((symbol, tech_analysis))
                        log_debug(f"Technical PASS: {symbol} conf={tech_analysis.technical_confidence:.2f}")
                    elif tech_analysis:
                        log_debug(f"Technical FAIL: {symbol} conf={tech_analysis.technical_confidence:.2f}")
                        
                except Exception as e:
                    log_debug(f"Error filtering {symbol}: {e}")
                    continue
        
        # Sort by confidence for better trade selection
        results.sort(key=lambda x: x[1].technical_confidence, reverse=True)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get analyzer performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'price_cache_size': len(self._price_cache),
            'analysis_cache_size': len(self._analysis_cache),
            'max_cache_size': self.MAX_CACHE_SIZE,
            'cache_duration_minutes': self.CACHE_DURATION_MINUTES
        }
    
    def clear_cache(self) -> None:
        """Clear all caches (useful for testing or memory management)."""
        self._price_cache.clear()
        self._analysis_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        log_info("Technical analyzer caches cleared")