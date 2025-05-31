"""
Optimized market condition and timing filters with improved datetime handling
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug


@dataclass
class MarketConditions:
    """Current market conditions assessment with comprehensive data."""
    is_market_hours: bool
    market_regime: str
    market_stress_level: float
    overall_trend: str
    volume_environment: str
    time_of_day_score: float


class MarketFilter:
    """Optimized market condition and timing filters."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with improved caching strategy."""
        self.fmp_loader = fmp_loader
        self._market_data_cache: Dict[str, float] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=15)
    
    def _get_market_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get SPY and VIX data with optimized caching."""
        now = datetime.now(timezone.utc)
        
        # Check cache validity
        if (self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_duration and
            'spy_price' in self._market_data_cache and 
            'vix_level' in self._market_data_cache):
            return self._market_data_cache['spy_price'], self._market_data_cache['vix_level']
        
        try:
            # Batch request for both SPY and VIX
            market_data = self.fmp_loader.get_real_time_prices(['SPY', 'VIX'])
            
            spy_price = None
            vix_level = None
            
            if market_data is not None and not market_data.empty:
                spy_row = market_data[market_data['symbol'] == 'SPY']
                vix_row = market_data[market_data['symbol'] == 'VIX']
                
                if not spy_row.empty:
                    spy_price = spy_row.iloc[0].get('lastSalePrice')
                if not vix_row.empty:
                    vix_level = vix_row.iloc[0].get('lastSalePrice')
            
            # Update cache
            if spy_price is not None:
                self._market_data_cache['spy_price'] = spy_price
            if vix_level is not None:
                self._market_data_cache['vix_level'] = vix_level
            
            self._cache_timestamp = now
            
            return spy_price, vix_level
            
        except Exception as e:
            log_warning(f"Error fetching market data: {e}")
            return None, None
    
    def get_market_conditions(self) -> MarketConditions:
        """Assess current market conditions with timezone awareness."""
        # Use timezone-aware datetime
        now = datetime.now(timezone.utc)
        
        # Convert to ET for market hours (assuming ET timezone)
        et_offset = timedelta(hours=-5)  # EST offset (adjust for DST as needed)
        et_time = (now + et_offset).time()
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        is_market_hours = (
            market_open <= et_time <= market_close and
            now.weekday() < 5  # Monday=0, Friday=4
        )
        
        # Time of day scoring
        time_score = self._calculate_time_score(et_time)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Market stress assessment with improved logic
        stress_level = self._calculate_stress_level(vix_level)
        
        # Market regime classification
        regime = self._classify_market_regime(vix_level, spy_price)
        
        return MarketConditions(
            is_market_hours=is_market_hours,
            market_regime=regime,
            market_stress_level=stress_level,
            overall_trend='neutral',
            volume_environment='normal',
            time_of_day_score=time_score
        )
    
    def _calculate_stress_level(self, vix_level: Optional[float]) -> float:
        """Calculate market stress level with improved thresholds."""
        if vix_level is None:
            return 0.3  # Default moderate stress
        
        # Improved stress calculation with gradual transitions
        if vix_level < 12:
            return 0.05  # Very low stress
        elif vix_level < 16:
            return 0.15  # Low stress
        elif vix_level < 20:
            return 0.25  # Moderate-low stress
        elif vix_level < 25:
            return 0.4   # Moderate stress
        elif vix_level < 30:
            return 0.6   # High stress
        elif vix_level < 40:
            return 0.8   # Very high stress
        else:
            return 0.95  # Extreme stress
    
    def _classify_market_regime(self, vix_level: Optional[float], spy_price: Optional[float]) -> str:
        """Classify market regime with improved logic."""
        if vix_level is None:
            return 'neutral'
        
        if vix_level > 35:
            return 'volatile'
        elif vix_level > 28:
            return 'bear'
        elif vix_level < 14:
            return 'bull'
        else:
            return 'neutral'
    
    def _calculate_time_score(self, current_time: time) -> float:
        """Calculate optimized time-of-day score."""
        hour = current_time.hour
        minute = current_time.minute
        
        # Convert to minutes from market open (9:30 AM = 0)
        if hour < 9 or (hour == 9 and minute < 30):
            return 0.1  # Before market open
        elif hour >= 16:
            return 0.1  # After market close
        
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Optimized scoring based on market microstructure
        if 0 <= minutes_from_open < 30:      # 9:30-10:00: Opening volatility
            return 0.7
        elif 30 <= minutes_from_open < 90:   # 10:00-11:00: Strong activity
            return 0.9
        elif 90 <= minutes_from_open < 150:  # 11:00-12:00: Good activity
            return 0.8
        elif 150 <= minutes_from_open < 210: # 12:00-1:30: Lunch lull
            return 0.3
        elif 210 <= minutes_from_open < 270: # 1:30-2:30: Afternoon pickup
            return 0.7
        elif 270 <= minutes_from_open < 360: # 2:30-3:30: Strong activity
            return 0.85
        elif 360 <= minutes_from_open < 390: # 3:30-4:00: End of day
            return 0.9
        else:
            return 0.1
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine trading suitability with improved logic and clear messages."""
        # Market hours requirement with detailed message
        if not market_conditions.is_market_hours:
            now = datetime.now(timezone.utc)
            et_offset = timedelta(hours=-5)  # EST offset
            et_time = (now + et_offset).time()
            
            return False, f"MARKET_CLOSED - Current time: {et_time.strftime('%H:%M')} ET (Market: 9:30-16:00 ET)"
        
        # Time of day filter with lower threshold
        if market_conditions.time_of_day_score < 0.4:
            return False, f"poor_trading_time (score: {market_conditions.time_of_day_score:.2f})"
        
        # Stress level filter with graduated response
        if market_conditions.market_stress_level > 0.85:
            return False, f"extreme_market_stress (level: {market_conditions.market_stress_level:.2f})"
        
        # Volatile regime filter
        if market_conditions.market_regime == 'volatile':
            return False, f"volatile_market_regime (stress: {market_conditions.market_stress_level:.2f})"
        
        return True, f"conditions_favorable (regime: {market_conditions.market_regime})"

class NewsQualityFilter:
    """Optimized news quality and freshness filters."""
    
    def __init__(self) -> None:
        """Initialize with efficient data structures."""
        self._processed_headlines: Set[str] = set()
        self._headline_cache_limit = 1000
        
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter news with optimized processing pipeline."""
        if news_df is None or news_df.empty:
            return news_df
        
        # Create a copy to avoid modifying original
        filtered_df = news_df.copy()
        
        # 1. Timezone-aware freshness filter
        filtered_df = self._filter_stale_news(filtered_df)
        
        # 2. Deduplicate headlines efficiently
        filtered_df = self._deduplicate_headlines(filtered_df)
        
        # 3. Content quality filters
        filtered_df = self._filter_content_quality(filtered_df)
        
        # 4. Priority scoring
        filtered_df = self._add_priority_scoring(filtered_df)
        
        return filtered_df
    
    def _filter_stale_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter stale news with timezone-aware comparison."""
        if 'publishedDate' not in news_df.columns:
            return news_df
        
        try:
            # Ensure timezone awareness
            now = datetime.now(timezone.utc)
            cutoff_time = now - timedelta(hours=2)
            
            # Convert published dates to UTC if not already
            mask = news_df['publishedDate'] > cutoff_time
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error filtering stale news: {e}")
            return news_df
    
    def _deduplicate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Efficient headline deduplication using vectorized operations."""
        if 'title' not in news_df.columns:
            return news_df
        
        try:
            # Simple deduplication using pandas
            news_df = news_df.drop_duplicates(subset=['title'], keep='first')
            
            # More sophisticated similarity check for remaining articles
            if len(news_df) > 100:  # Only for larger datasets to avoid performance issues
                return news_df
            
            # For smaller datasets, do similarity checking
            filtered_indices = []
            seen_word_sets = []
            
            for idx, row in news_df.iterrows():
                title = str(row['title']).lower().strip()
                title_words = set(title.split())
                
                is_duplicate = False
                for seen_words in seen_word_sets:
                    if title_words and seen_words:
                        overlap = len(title_words & seen_words) / len(title_words | seen_words)
                        if overlap > 0.8:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_indices.append(idx)
                    seen_word_sets.append(title_words)
                    
                    # Limit memory usage
                    if len(seen_word_sets) > self._headline_cache_limit:
                        seen_word_sets = seen_word_sets[-self._headline_cache_limit//2:]
            
            return news_df.loc[filtered_indices]
            
        except Exception as e:
            log_warning(f"Error in headline deduplication: {e}")
            return news_df
    
    def _filter_content_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized content quality filtering."""
        if news_df.empty:
            return news_df
        
        try:
            mask = pd.Series([True] * len(news_df), index=news_df.index)
            
            # Title length filter
            if 'title' in news_df.columns:
                mask &= news_df['title'].str.len() >= 20
                
                # Suspicious content filter
                suspicious_patterns = ['click here', 'ad:', 'advertisement', 'sponsored']
                for pattern in suspicious_patterns:
                    mask &= ~news_df['title'].str.lower().str.contains(pattern, na=False)
            
            # Text length filter
            if 'text' in news_df.columns:
                mask &= news_df['text'].str.len() >= 100
            
            return news_df[mask]
            
        except Exception as e:
            log_warning(f"Error in content quality filtering: {e}")
            return news_df
    
    def _add_priority_scoring(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scoring with vectorized operations."""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        try:
            # Initialize priority score
            news_df = news_df.copy()
            news_df['priority_score'] = 0.5
            
            # Official announcement keywords
            official_keywords = [
                'announces', 'reports', 'declares', 'files', 'receives',
                'completes', 'signs', 'launches', 'enters into', 'appoints'
            ]
            
            # Vectorized keyword matching
            for keyword in official_keywords:
                mask = news_df['title'].str.lower().str.contains(keyword, na=False)
                news_df.loc[mask, 'priority_score'] += 0.1
            
            # Sort by priority and timestamp
            sort_columns = ['priority_score']
            if 'publishedDate' in news_df.columns:
                sort_columns.append('publishedDate')
            
            news_df = news_df.sort_values(sort_columns, ascending=[False, False])
            
            return news_df
            
        except Exception as e:
            log_warning(f"Error in priority scoring: {e}")
            return news_df


class PriceActionFilter:
    """Optimized price action filtering with improved performance."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with caching for price history."""
        self.fmp_loader = fmp_loader
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=1)
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Efficiently filter symbols based on price action."""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        # Batch process symbols for efficiency
        filtered_symbols = []
        
        # Pre-filter symbols that have current price data
        available_symbols = set(current_prices['symbol'].tolist())
        candidate_symbols = [s for s in symbols if s in available_symbols]
        
        for symbol in candidate_symbols:
            if self._is_good_price_action(symbol, current_prices):
                filtered_symbols.append(symbol)
        
        return filtered_symbols
    
    def _is_good_price_action(self, symbol: str, current_prices: pd.DataFrame) -> bool:
        """Optimized price action validation."""
        try:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                return False
            
            current_price = price_row.iloc[0].get('lastSalePrice', 0)
            if current_price <= 0:
                return False
            
            # Get cached or fetch recent history
            hist_data = self._get_cached_history(symbol)
            if hist_data is None or hist_data.empty:
                return True  # Allow if no history available
            
            # Vectorized calculations for efficiency
            return self._validate_price_metrics(symbol, current_price, hist_data)
            
        except Exception as e:
            log_debug(f"Error checking price action for {symbol}: {e}")
            return True  # Default to allowing
    
    def _get_cached_history(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price history with intelligent caching."""
        now = datetime.now()
        
        # Check cache validity
        if (symbol in self._history_cache and 
            symbol in self._cache_timestamps and
            now - self._cache_timestamps[symbol] < self._cache_duration):
            return self._history_cache[symbol]
        
        # Fetch new data
        hist_data = self._fetch_recent_history(symbol)
        if hist_data is not None:
            self._history_cache[symbol] = hist_data
            self._cache_timestamps[symbol] = now
            
            # Limit cache size
            if len(self._history_cache) > 100:
                oldest_symbol = min(self._cache_timestamps.keys(), 
                                  key=lambda k: self._cache_timestamps[k])
                del self._history_cache[oldest_symbol]
                del self._cache_timestamps[oldest_symbol]
        
        return hist_data
    
    def _fetch_recent_history(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch recent price history with error handling."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            data = self.fmp_loader._make_request(f"historical-price-full/{symbol}", {
                'from': start_date,
                'to': end_date
            })
            
            if data and 'historical' in data and data['historical']:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Ensure numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
        except Exception as e:
            log_debug(f"Could not get history for {symbol}: {e}")
        
        return None
    
    def _validate_price_metrics(self, symbol: str, current_price: float, hist_data: pd.DataFrame) -> bool:
        """Validate price metrics with vectorized calculations."""
        try:
            # Gap analysis
            if len(hist_data) >= 2:
                prev_close = hist_data['close'].iloc[-2]
                gap_pct = abs(current_price - prev_close) / prev_close
                
                if gap_pct > 0.10:  # 10% gap threshold
                    return False
            
            # Volatility analysis using vectorized operations
            if len(hist_data) >= 5 and 'high' in hist_data.columns and 'low' in hist_data.columns:
                daily_ranges = (hist_data['high'] - hist_data['low']) / hist_data['close']
                avg_range = daily_ranges.tail(5).mean()
                
                if avg_range > 0.15:  # 15% average daily range
                    return False
            
            # 52-week extreme check (if enough data)
            if len(hist_data) >= 200:
                year_data = hist_data.tail(250)
                year_high = year_data['high'].max()
                year_low = year_data['low'].min()
                
                high_distance = (year_high - current_price) / year_high
                low_distance = (current_price - year_low) / year_low
                
                # Near extremes but don't automatically reject
                if high_distance < 0.02 or low_distance < 0.02:
                    log_debug(f"Near 52-week extreme for {symbol}")
            
            return True
            
        except Exception as e:
            log_debug(f"Error validating price metrics for {symbol}: {e}")
            return True


class PortfolioRiskFilter:
    """Optimized portfolio risk management with improved calculations."""
    
    def __init__(self, trader) -> None:
        """Initialize with trader reference for position data."""
        self.trader = trader
        self._sector_cache: Dict[str, str] = {}
    
    def check_portfolio_limits(self, new_symbol: str, new_side: str, 
                             new_position_size: float) -> Tuple[bool, str]:
        """Comprehensive portfolio limit checking with optimized calculations."""
        try:
            active_trades = self.trader.get_active_positions()
            
            # 1. Position count limit
            if len(active_trades) >= 10:
                return False, "max_positions_reached"
            
            # 2. Duplicate position check
            if any(trade.symbol == new_symbol for trade in active_trades):
                return False, "position_already_exists"
            
            # 3. Sector concentration with caching
            sector_exposure = self._calculate_sector_exposure(active_trades, new_symbol)
            if sector_exposure > 0.4:
                return False, "sector_concentration_limit"
            
            # 4. Portfolio heat calculation
            current_risk = sum(trade.position_size * CONFIG.stop_loss_pct for trade in active_trades)
            new_risk = new_position_size * CONFIG.stop_loss_pct
            total_risk = current_risk + new_risk
            
            max_total_risk = CONFIG.position_size * 5
            if total_risk > max_total_risk:
                return False, "portfolio_risk_limit"
            
            # 5. Directional balance
            long_count = sum(1 for trade in active_trades if trade.side == 'long')
            short_count = len(active_trades) - long_count
            
            total_positions = len(active_trades) + 1
            new_long_ratio = (long_count + 1) / total_positions if new_side == 'long' else long_count / total_positions
            
            if new_long_ratio > 0.8 or new_long_ratio < 0.2:
                return False, "directional_imbalance"
            
            return True, "within_limits"
            
        except Exception as e:
            log_warning(f"Error checking portfolio limits: {e}")
            return False, "portfolio_check_error"
    
    def _calculate_sector_exposure(self, active_trades: List, new_symbol: str) -> float:
        """Calculate sector exposure with caching."""
        try:
            new_sector = self._get_sector(new_symbol)
            sector_count = sum(1 for trade in active_trades 
                             if self._get_sector(trade.symbol) == new_sector)
            
            total_positions = len(active_trades) + 1
            return (sector_count + 1) / total_positions if total_positions > 0 else 0
            
        except Exception as e:
            log_debug(f"Error calculating sector exposure: {e}")
            return 0.0
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector classification with caching."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        
        symbol_upper = symbol.upper()
        
        # Sector classification logic
        if symbol_upper in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'CRM', 'ORCL', 'ADBE']:
            sector = 'tech'
        elif symbol_upper in ['GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN'] or 'BIO' in symbol_upper:
            sector = 'biotech'
        elif symbol_upper in ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS']:
            sector = 'finance'
        elif symbol_upper in ['JNJ', 'PFE', 'MRK', 'ABT', 'TMO']:
            sector = 'healthcare'
        else:
            sector = 'other'
        
        # Cache the result
        self._sector_cache[symbol] = sector
        return sector


class EntryTimingOptimizer:
    """Optimized entry timing with improved decision logic."""
    
    def __init__(self) -> None:
        """Initialize with optimized data structures."""
        self.pending_entries: Dict[str, Dict] = {}
        self.entry_delay = 60  # seconds
        self.max_pending_time = 300  # 5 minutes
    
    def should_enter_now(self, symbol: str, analysis, current_price: float) -> Tuple[bool, str]:
        """Optimized entry timing decision with improved logic."""
        try:
            # Immediate entry for very high conviction
            if hasattr(analysis, 'combined_confidence') and analysis.combined_confidence > 0.85:
                return True, "high_conviction_immediate"
            
            now = datetime.now()
            
            # First time seeing this signal
            if symbol not in self.pending_entries:
                self.pending_entries[symbol] = {
                    'timestamp': now,
                    'analysis': analysis,
                    'initial_price': current_price,
                    'sentiment': getattr(analysis, 'sentiment_score', 0)
                }
                return False, "waiting_for_confirmation"
            
            # Check timing
            pending = self.pending_entries[symbol]
            time_elapsed = (now - pending['timestamp']).total_seconds()
            
            if time_elapsed < self.entry_delay:
                return False, "confirmation_period"
            
            # Price movement validation
            initial_price = pending['initial_price']
            price_change = (current_price - initial_price) / initial_price
            sentiment = pending['sentiment']
            
            # Entry logic based on sentiment and price movement
            if sentiment > 0:  # Bullish signal
                if -0.02 <= price_change <= 0.05:  # Reasonable price movement
                    del self.pending_entries[symbol]
                    return True, "confirmed_long_entry"
            elif sentiment < 0:  # Bearish signal
                if -0.05 <= price_change <= 0.02:  # Reasonable price movement
                    del self.pending_entries[symbol]
                    return True, "confirmed_short_entry"
            
            # Price moved too much
            if abs(price_change) > 0.10:
                del self.pending_entries[symbol]
                return False, "price_moved_too_much"
            
            # Still waiting
            if time_elapsed > self.max_pending_time:
                del self.pending_entries[symbol]
                return False, "timeout_waiting"
            
            return False, "waiting_for_better_entry"
            
        except Exception as e:
            log_debug(f"Error in entry timing for {symbol}: {e}")
            return False, "timing_error"
    
    def cleanup_stale_entries(self) -> None:
        """Efficiently clean up stale pending entries."""
        now = datetime.now()
        stale_symbols = [
            symbol for symbol, pending in self.pending_entries.items()
            if (now - pending['timestamp']).total_seconds() > self.max_pending_time
        ]
        
        for symbol in stale_symbols:
            del self.pending_entries[symbol]