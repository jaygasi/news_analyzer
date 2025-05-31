"""
Market condition and timing filters for trade quality improvement
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import CONFIG
from utils.simple_logger import log_info, log_warning, log_debug

@dataclass
class MarketConditions:
    """Current market conditions assessment"""
    is_market_hours: bool
    market_regime: str  # 'bull', 'bear', 'neutral', 'volatile'
    market_stress_level: float  # 0-1 (0=calm, 1=stressed)
    overall_trend: str  # 'up', 'down', 'sideways'
    volume_environment: str  # 'high', 'normal', 'low'
    time_of_day_score: float  # 0-1 (1=best trading time)

class MarketFilter:
    """Market condition and timing filters"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
        self.spy_cache = {}
        self.vix_cache = {}
        self.cache_expiry = None
        self.cache_duration = timedelta(minutes=15)
    
    def _get_market_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get SPY and VIX data with caching"""
        now = datetime.now()
        
        if (self.cache_expiry and now < self.cache_expiry and 
            'spy_price' in self.spy_cache and 'vix_level' in self.vix_cache):
            return self.spy_cache['spy_price'], self.vix_cache['vix_level']
        
        try:
            # Get SPY current price
            spy_data = self.fmp_loader.get_real_time_prices(['SPY'])
            spy_price = None
            if spy_data is not None and not spy_data.empty:
                spy_price = spy_data.iloc[0].get('lastSalePrice', None)
            
            # Get VIX level
            vix_data = self.fmp_loader.get_real_time_prices(['VIX'])
            vix_level = None
            if vix_data is not None and not vix_data.empty:
                vix_level = vix_data.iloc[0].get('lastSalePrice', None)
            
            # Cache results
            if spy_price:
                self.spy_cache['spy_price'] = spy_price
            if vix_level:
                self.vix_cache['vix_level'] = vix_level
            self.cache_expiry = now + self.cache_duration
            
            return spy_price, vix_level
            
        except Exception as e:
            log_warning(f"Error fetching market data: {e}")
            return None, None
    
    def get_market_conditions(self) -> MarketConditions:
        """Assess current market conditions"""
        now = datetime.now()
        current_time = now.time()
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)
        lunch_start = time(12, 0)
        lunch_end = time(13, 30)
        
        is_market_hours = (market_open <= current_time <= market_close and
                          now.weekday() < 5)  # Monday=0, Friday=4
        
        # Time of day scoring (avoid lunch hour, prefer morning/late afternoon)
        time_score = self._calculate_time_score(current_time)
        
        # Get market indicators
        spy_price, vix_level = self._get_market_data()
        
        # Market stress assessment
        stress_level = 0.3  # Default moderate
        if vix_level:
            if vix_level < 15:
                stress_level = 0.1  # Low stress
            elif vix_level < 25:
                stress_level = 0.3  # Moderate stress
            elif vix_level < 35:
                stress_level = 0.6  # High stress
            else:
                stress_level = 0.9  # Very high stress
        
        # Market regime (simplified)
        regime = 'neutral'
        if vix_level:
            if vix_level > 30:
                regime = 'volatile'
            elif vix_level > 25:
                regime = 'bear'
            elif vix_level < 15:
                regime = 'bull'
        
        return MarketConditions(
            is_market_hours=is_market_hours,
            market_regime=regime,
            market_stress_level=stress_level,
            overall_trend='neutral',  # Could be enhanced with SPY trend analysis
            volume_environment='normal',  # Could be enhanced with volume analysis
            time_of_day_score=time_score
        )
    
    def _calculate_time_score(self, current_time: time) -> float:
        """Calculate time-of-day score (1=best trading time, 0=worst)"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Convert to minutes from market open (9:30 AM = 0)
        minutes_from_open = (hour - 9) * 60 + (minute - 30)
        
        # Scoring based on typical market patterns
        if 0 <= minutes_from_open < 60:  # First hour: high activity
            return 0.9
        elif 60 <= minutes_from_open < 150:  # 10:30-12:00: good activity
            return 0.8
        elif 150 <= minutes_from_open < 210:  # 12:00-1:30: lunch lull
            return 0.3
        elif 210 <= minutes_from_open < 300:  # 1:30-3:00: moderate activity
            return 0.7
        elif 300 <= minutes_from_open < 390:  # 3:00-4:00: end of day activity
            return 0.8
        else:
            return 0.1  # Outside market hours
    
    def should_trade_now(self, market_conditions: MarketConditions) -> Tuple[bool, str]:
        """Determine if current conditions are suitable for trading"""
        
        # Market hours requirement
        if not market_conditions.is_market_hours:
            return False, "outside_market_hours"
        
        # Time of day filter
        if market_conditions.time_of_day_score < 0.5:
            return False, "poor_trading_time"
        
        # Market stress filter
        if market_conditions.market_stress_level > 0.8:
            return False, "high_market_stress"
        
        # Volatile market regime
        if market_conditions.market_regime == 'volatile':
            return False, "volatile_market_regime"
        
        return True, "conditions_favorable"

class NewsQualityFilter:
    """News quality and freshness filters"""
    
    def __init__(self):
        self.processed_headlines = set()
        self.headline_cache_limit = 1000
        
    def filter_news_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter news for quality and freshness"""
        if news_df is None or news_df.empty:
            return news_df
        
        original_count = len(news_df)
        
        # 1. Remove stale news (older than 2 hours)
        if 'publishedDate' in news_df.columns:
            news_df['publishedDate'] = pd.to_datetime(news_df['publishedDate'])
            cutoff_time = datetime.now() - timedelta(hours=2)
            news_df = news_df[news_df['publishedDate'] > cutoff_time]
        
        # 2. Remove duplicate/similar headlines
        news_df = self._deduplicate_headlines(news_df)
        
        # 3. Filter low-quality sources and content
        news_df = self._filter_content_quality(news_df)
        
        # 4. Prioritize official announcements
        news_df = self._prioritize_official_news(news_df)
        
        filtered_count = len(news_df)
        if filtered_count < original_count:
            log_debug(f"News quality filter: {original_count} → {filtered_count} articles")
        
        return news_df
    
    def _deduplicate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate and very similar headlines"""
        if 'title' not in news_df.columns:
            return news_df
        
        # Simple deduplication based on title similarity
        unique_news = []
        seen_headlines = set()
        
        for _, row in news_df.iterrows():
            title = str(row['title']).lower().strip()
            
            # Create a simplified version for comparison
            title_words = set(title.split())
            is_duplicate = False
            
            for seen_title in seen_headlines:
                seen_words = set(seen_title.split())
                
                # If 80%+ words overlap, consider it duplicate
                if len(title_words & seen_words) / len(title_words | seen_words) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_news.append(row)
                seen_headlines.add(title)
                
                # Limit cache size
                if len(seen_headlines) > self.headline_cache_limit:
                    seen_headlines.clear()
        
        return pd.DataFrame(unique_news) if unique_news else pd.DataFrame()
    
    def _filter_content_quality(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-quality content"""
        if news_df.empty:
            return news_df
        
        # Remove articles with poor content indicators
        quality_filters = []
        
        if 'title' in news_df.columns:
            # Remove very short titles
            quality_filters.append(news_df['title'].str.len() >= 20)
            
            # Remove articles with suspicious patterns
            suspicious_patterns = ['click here', 'ad:', 'advertisement', 'sponsored']
            for pattern in suspicious_patterns:
                quality_filters.append(~news_df['title'].str.lower().str.contains(pattern, na=False))
        
        if 'text' in news_df.columns:
            # Remove very short articles
            quality_filters.append(news_df['text'].str.len() >= 100)
        
        # Apply all filters
        final_filter = pd.Series([True] * len(news_df))
        for filt in quality_filters:
            final_filter &= filt
        
        return news_df[final_filter]
    
    def _prioritize_official_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Add priority scoring for official announcements"""
        if news_df.empty or 'title' not in news_df.columns:
            return news_df
        
        # Keywords that indicate official announcements
        official_keywords = [
            'announces', 'reports', 'declares', 'files', 'receives',
            'completes', 'signs', 'launches', 'enters into', 'appoints'
        ]
        
        # Add priority score
        news_df['priority_score'] = 0.5  # Default priority
        
        for keyword in official_keywords:
            mask = news_df['title'].str.lower().str.contains(keyword, na=False)
            news_df.loc[mask, 'priority_score'] += 0.1
        
        # Sort by priority (highest first)
        news_df = news_df.sort_values('priority_score', ascending=False)
        
        return news_df

class PriceActionFilter:
    """Price action and technical filters"""
    
    def __init__(self, fmp_loader):
        self.fmp_loader = fmp_loader
    
    def filter_price_action(self, symbols: List[str], current_prices: pd.DataFrame) -> List[str]:
        """Filter symbols based on price action quality"""
        if not symbols or current_prices is None or current_prices.empty:
            return symbols
        
        filtered_symbols = []
        
        for symbol in symbols:
            if self._is_good_price_action(symbol, current_prices):
                filtered_symbols.append(symbol)
            else:
                log_debug(f"Price action filter removed: {symbol}")
        
        return filtered_symbols
    
    def _is_good_price_action(self, symbol: str, current_prices: pd.DataFrame) -> bool:
        """Check if symbol has good price action for entry"""
        price_row = current_prices[current_prices['symbol'] == symbol]
        if price_row.empty:
            return False
        
        current_price = price_row.iloc[0].get('lastSalePrice', 0)
        if current_price <= 0:
            return False
        
        try:
            # Get recent price history for gap and volatility analysis
            hist_data = self._get_recent_history(symbol)
            if hist_data is None or hist_data.empty:
                return True  # Allow if no history available
            
            # 1. Check for excessive gaps (avoid stocks that just gapped >10%)
            if len(hist_data) >= 2:
                prev_close = hist_data['close'].iloc[-2]
                gap_pct = abs(current_price - prev_close) / prev_close
                
                if gap_pct > 0.10:  # 10% gap
                    log_debug(f"Large gap detected for {symbol}: {gap_pct:.2%}")
                    return False
            
            # 2. Avoid extremely volatile stocks (daily range >15%)
            if 'high' in hist_data.columns and 'low' in hist_data.columns:
                recent_ranges = (hist_data['high'] - hist_data['low']) / hist_data['close']
                avg_range = recent_ranges.tail(5).mean()
                
                if avg_range > 0.15:  # 15% average daily range
                    log_debug(f"High volatility for {symbol}: {avg_range:.2%}")
                    return False
            
            # 3. Avoid stocks at 52-week extremes without strong conviction
            if len(hist_data) >= 250:  # Roughly 1 year of data
                year_high = hist_data['high'].tail(250).max()
                year_low = hist_data['low'].tail(250).min()
                
                # If within 5% of 52-week high/low, require extra caution
                high_distance = (year_high - current_price) / year_high
                low_distance = (current_price - year_low) / year_low
                
                if high_distance < 0.05 or low_distance < 0.05:
                    log_debug(f"Near 52-week extreme for {symbol}")
                    # Don't automatically reject, but could lower confidence
            
            return True
            
        except Exception as e:
            log_warning(f"Error checking price action for {symbol}: {e}")
            return True  # Default to allowing if analysis fails
    
    def _get_recent_history(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get recent price history"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            data = self.fmp_loader._make_request(f"historical-price-full/{symbol}", {
                'from': start_date,
                'to': end_date
            })
            
            if data and 'historical' in data:
                df = pd.DataFrame(data['historical'])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    return df.sort_values('date')
            
        except Exception as e:
            log_debug(f"Could not get history for {symbol}: {e}")
        
        return None

class PortfolioRiskFilter:
    """Portfolio-level risk management filters"""
    
    def __init__(self, trader):
        self.trader = trader
    
    def check_portfolio_limits(self, new_symbol: str, new_side: str, 
                             new_position_size: float) -> Tuple[bool, str]:
        """Check if new trade violates portfolio limits"""
        
        active_trades = self.trader.get_active_positions()
        
        # 1. Maximum number of positions
        max_positions = 10  # Configurable limit
        if len(active_trades) >= max_positions:
            return False, "max_positions_reached"
        
        # 2. Check if already have position in same symbol
        if any(trade.symbol == new_symbol for trade in active_trades):
            return False, "position_already_exists"
        
        # 3. Sector concentration limits (simplified by symbol patterns)
        sector_exposure = self._calculate_sector_exposure(active_trades, new_symbol)
        if sector_exposure > 0.4:  # Max 40% in one sector
            return False, "sector_concentration_limit"
        
        # 4. Portfolio heat (total risk exposure)
        total_risk = sum(trade.position_size * CONFIG.stop_loss_pct for trade in active_trades)
        total_risk += new_position_size * CONFIG.stop_loss_pct
        
        max_total_risk = CONFIG.position_size * 5  # Max risk = 5 full positions
        if total_risk > max_total_risk:
            return False, "portfolio_risk_limit"
        
        # 5. Long/Short balance
        long_count = sum(1 for trade in active_trades if trade.side == 'long')
        short_count = sum(1 for trade in active_trades if trade.side == 'short')
        
        # Don't allow more than 80% in one direction
        total_positions = len(active_trades) + 1
        if new_side == 'long':
            long_ratio = (long_count + 1) / total_positions
        else:
            long_ratio = long_count / total_positions
        
        if long_ratio > 0.8 or long_ratio < 0.2:
            return False, "directional_imbalance"
        
        return True, "within_limits"
    
    def _calculate_sector_exposure(self, active_trades: List, new_symbol: str) -> float:
        """Calculate sector exposure (simplified)"""
        # Simple sector classification based on symbol patterns
        tech_patterns = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
        biotech_patterns = ['GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN']
        finance_patterns = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        
        # Count positions in same sector as new symbol
        new_sector = self._get_simple_sector(new_symbol)
        sector_count = sum(1 for trade in active_trades 
                          if self._get_simple_sector(trade.symbol) == new_sector)
        
        total_positions = len(active_trades) + 1
        return sector_count / total_positions if total_positions > 0 else 0
    
    def _get_simple_sector(self, symbol: str) -> str:
        """Simple sector classification"""
        symbol = symbol.upper()
        
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'CRM', 'ORCL']
        if any(tech in symbol for tech in tech_symbols):
            return 'tech'
        
        biotech_symbols = ['GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN']
        if any(bio in symbol for bio in biotech_symbols) or 'BIO' in symbol:
            return 'biotech'
        
        finance_symbols = ['JPM', 'BAC', 'WFC', 'C', 'GS']
        if any(fin in symbol for fin in finance_symbols):
            return 'finance'
        
        return 'other'

class EntryTimingOptimizer:
    """Optimize entry timing after news detection"""
    
    def __init__(self):
        self.pending_entries = {}  # symbol -> {timestamp, analysis, prices}
        self.entry_delay = 60  # Wait 1 minute for initial reaction
        
    def should_enter_now(self, symbol: str, analysis, current_price: float) -> Tuple[bool, str]:
        """Determine optimal entry timing"""
        
        # For immediate high-conviction signals
        if analysis.combined_confidence > 0.85:
            return True, "high_conviction_immediate"
        
        # For other signals, wait for price confirmation
        now = datetime.now()
        
        if symbol not in self.pending_entries:
            # First time seeing this signal - wait for confirmation
            self.pending_entries[symbol] = {
                'timestamp': now,
                'analysis': analysis,
                'initial_price': current_price
            }
            return False, "waiting_for_confirmation"
        
        # Check if enough time has passed
        pending = self.pending_entries[symbol]
        time_elapsed = (now - pending['timestamp']).total_seconds()
        
        if time_elapsed < self.entry_delay:
            return False, "confirmation_period"
        
        # Check price movement direction aligns with sentiment
        initial_price = pending['initial_price']
        price_change = (current_price - initial_price) / initial_price
        
        sentiment = analysis.sentiment_score
        
        # For long signals, prefer entry on slight pullback or continued strength
        if sentiment > 0:
            if -0.02 <= price_change <= 0.05:  # -2% to +5% move
                del self.pending_entries[symbol]
                return True, "confirmed_long_entry"
        
        # For short signals, prefer entry on slight bounce or continued weakness  
        elif sentiment < 0:
            if -0.05 <= price_change <= 0.02:  # -5% to +2% move
                del self.pending_entries[symbol]
                return True, "confirmed_short_entry"
        
        # If price moved too much against us, skip
        if abs(price_change) > 0.1:  # 10% move
            del self.pending_entries[symbol]
            return False, "price_moved_too_much"
        
        return False, "waiting_for_better_entry"
    
    def cleanup_stale_entries(self):
        """Remove stale pending entries"""
        now = datetime.now()
        stale_symbols = []
        
        for symbol, pending in self.pending_entries.items():
            if (now - pending['timestamp']).total_seconds() > 300:  # 5 minutes
                stale_symbols.append(symbol)
        
        for symbol in stale_symbols:
            del self.pending_entries[symbol]