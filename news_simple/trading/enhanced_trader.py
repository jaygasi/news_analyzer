"""
Optimized trader with comprehensive filtering and improved performance
"""
import pandas as pd
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import csv
from pathlib import Path
from config import CONFIG
from analysis.enhanced_news_analyzer import EnhancedNewsAnalysis
from analysis.technical_analyzer import TechnicalAnalysis
from analysis.market_filters import (
    MarketFilter, NewsQualityFilter, PriceActionFilter, 
    PortfolioRiskFilter, EntryTimingOptimizer
)
from utils.simple_logger import log_info, log_error, log_warning


@dataclass
class EnhancedTrade:
    """Enhanced trade record with comprehensive data and type hints."""
    id: str
    symbol: str
    side: str
    entry_price: float
    position_size: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
    # News analysis data
    news_title: str = ""
    news_confidence: float = 0.0
    combined_confidence: float = 0.0
    finbert_score: float = 0.0
    keyword_score: float = 0.0
    topic: str = ""
    
    # Technical analysis data
    technical_confidence: float = 0.0
    liquidity_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    rsi: float = 50.0
    price_trend: str = ""
    bid_ask_spread: float = 0.0
    
    # Market condition data
    market_regime: str = ""
    market_stress: float = 0.0
    time_score: float = 0.0
    entry_timing: str = ""


class EnhancedTrader:
    """Optimized trader with comprehensive filtering and performance improvements."""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized components and error handling."""
        self.fmp_loader = fmp_loader
        self.active_trades: Dict[str, EnhancedTrade] = {}
        self.trade_log_file = CONFIG.trade_log_path / CONFIG.trade_log_file
        self.trade_counter = 0
        
        # Initialize filter components
        self._initialize_filters()
        
        # Setup trade logging
        self._setup_trade_log()
    
    def _initialize_filters(self) -> None:
        """Initialize all filter components with error handling."""
        try:
            self.market_filter = MarketFilter(self.fmp_loader)
            self.news_filter = NewsQualityFilter()
            self.price_filter = PriceActionFilter(self.fmp_loader)
            self.portfolio_filter = PortfolioRiskFilter(self)
            self.timing_optimizer = EntryTimingOptimizer()
        except Exception as e:
            log_error(f"Error initializing filters: {e}")
            raise
    
    def _setup_trade_log(self) -> None:
        """Setup comprehensive trade log CSV with proper headers."""
        try:
            if not self.trade_log_file.exists():
                headers = [
                    # Trade basics
                    'trade_id', 'symbol', 'side', 'entry_price', 'position_size',
                    'entry_time', 'exit_price', 'exit_time', 'exit_reason', 'pnl',
                    
                    # News analysis
                    'news_title', 'news_confidence', 'combined_confidence',
                    'finbert_score', 'keyword_score', 'topic',
                    
                    # Technical analysis
                    'technical_confidence', 'liquidity_score', 'momentum_score',
                    'volume_score', 'rsi', 'price_trend', 'bid_ask_spread',
                    
                    # Market conditions
                    'market_regime', 'market_stress', 'time_score', 'entry_timing'
                ]
                
                with open(self.trade_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    
        except Exception as e:
            log_error(f"Error setting up trade log: {e}")
            raise
    
    def _log_trade(self, trade: EnhancedTrade) -> None:
        """Log comprehensive trade data to CSV with error handling."""
        try:
            with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Convert dataclass to ordered list
                trade_data = [
                    trade.id, trade.symbol, trade.side, trade.entry_price,
                    trade.position_size, trade.entry_time, trade.exit_price,
                    trade.exit_time, trade.exit_reason, trade.pnl,
                    
                    trade.news_title, trade.news_confidence, trade.combined_confidence,
                    trade.finbert_score, trade.keyword_score, trade.topic,
                    
                    trade.technical_confidence, trade.liquidity_score, trade.momentum_score,
                    trade.volume_score, trade.rsi, trade.price_trend, trade.bid_ask_spread,
                    
                    trade.market_regime, trade.market_stress, trade.time_score, trade.entry_timing
                ]
                
                writer.writerow(trade_data)
                
        except Exception as e:
            log_error(f"Error logging trade {trade.id}: {e}")
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID with timestamp."""
        self.trade_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d')
        return f"T{timestamp}_{self.trade_counter:04d}"
    
    def pre_filter_news(self, news_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply news quality filters before analysis with comprehensive validation."""
        if news_df is None or news_df.empty:
            return news_df
        
        try:
            # Apply news quality filters
            filtered_news = self.news_filter.filter_news_quality(news_df)
            
            if filtered_news is not None:
                original_count = len(news_df)
                filtered_count = len(filtered_news)
                
                if filtered_count < original_count:
                    log_info(f"News pre-filter: {original_count} -> {filtered_count} articles")
                
                return filtered_news
            else:
                log_warning("News filter returned None")
                return pd.DataFrame()  # Return empty DataFrame instead of None
                
        except Exception as e:
            log_error(f"Error in news pre-filtering: {e}")
            return news_df  # Return original on error
    
    def _determine_position_size(self, analysis: EnhancedNewsAnalysis) -> float:
        """Enhanced dynamic position sizing with risk management."""
        try:
            base_size = CONFIG.position_size
            
            # Start with combined confidence
            confidence_multiplier = analysis.combined_confidence
            
            # Technical adjustments
            if analysis.technical_analysis:
                tech = analysis.technical_analysis
                
                # Volatility adjustment (reduce size for high volatility)
                volatility_factor = max(0.5, 1.0 - (tech.volatility_score * 0.3))
                
                # Liquidity adjustment (reduce size for poor liquidity)
                liquidity_factor = max(0.5, tech.liquidity_score)
                
                # Momentum alignment bonus
                momentum_boost = 1.0
                sentiment = analysis.sentiment_score
                momentum = tech.momentum_score
                
                if abs(sentiment) > 0.3 and abs(momentum) > 0.2:
                    if (sentiment > 0) == (momentum > 0):
                        momentum_boost = 1.15  # Aligned signals
                
                confidence_multiplier *= volatility_factor * liquidity_factor * momentum_boost
            
            # Topic-based multipliers
            topic_multipliers = {
                'earnings': 1.1,
                'biotech': 1.2,
                'ma': 1.15,
                'analyst': 0.9,
                'general': 0.95
            }
            
            topic_mult = topic_multipliers.get(analysis.topic, 1.0)
            confidence_multiplier *= topic_mult
            
            # Apply position sizing bounds
            final_multiplier = max(
                CONFIG.min_position_multiplier,
                min(CONFIG.max_position_multiplier, confidence_multiplier)
            )
            
            return base_size * final_multiplier
            
        except Exception as e:
            log_error(f"Error calculating position size: {e}")
            return CONFIG.position_size * CONFIG.min_position_multiplier
    
    def process_news_signals(self, analyses: List[EnhancedNewsAnalysis], 
                           current_prices: pd.DataFrame) -> None:
        """Process news signals with comprehensive filtering pipeline."""
        if not analyses or current_prices is None or current_prices.empty:
            return
        
        try:
            # 1. Market condition check
            if not self._check_market_conditions():
                return
            
            # 2. Price action filtering
            filtered_analyses = self._apply_price_action_filter(analyses, current_prices)
            
            # 3. Process individual signals
            trades_created = self._process_filtered_signals(filtered_analyses, current_prices)
            
            # 4. Cleanup and logging
            self.timing_optimizer.cleanup_stale_entries()
            
            if trades_created > 0:
                log_info(f"[TRADES] Created {trades_created} trades from {len(analyses)} signals")
            elif analyses:
                log_info(f"[FILTER] No trades created from {len(analyses)} signals")
                
        except Exception as e:
            log_error(f"Error processing news signals: {e}")
    
    def _check_market_conditions(self) -> bool:
        """Check if market conditions are favorable for trading."""
        try:
            market_conditions = self.market_filter.get_market_conditions()
            should_trade, reason = self.market_filter.should_trade_now(market_conditions)
            
            if not should_trade:
                log_info(f"Market conditions unfavorable: {reason}")
                return False
            
            log_info(f"Market conditions favorable: regime={market_conditions.market_regime}, "
                    f"stress={market_conditions.market_stress_level:.2f}")
            return True
            
        except Exception as e:
            log_error(f"Error checking market conditions: {e}")
            return False
    
    def _apply_price_action_filter(self, analyses: List[EnhancedNewsAnalysis], 
                                  current_prices: pd.DataFrame) -> List[EnhancedNewsAnalysis]:
        """Apply price action filters to analysis list."""
        try:
            candidate_symbols = [a.symbol for a in analyses]
            good_symbols = self.price_filter.filter_price_action(candidate_symbols, current_prices)
            good_symbols_set = set(good_symbols)
            
            filtered_analyses = [a for a in analyses if a.symbol in good_symbols_set]
            
            if len(filtered_analyses) < len(analyses):
                log_info(f"Price action filter: {len(analyses)} -> {len(filtered_analyses)} signals")
            
            return filtered_analyses
            
        except Exception as e:
            log_error(f"Error applying price action filter: {e}")
            return analyses
    
    def _process_filtered_signals(self, analyses: List[EnhancedNewsAnalysis], 
                                 current_prices: pd.DataFrame) -> int:
        """Process filtered signals and create trades."""
        trades_created = 0
        
        for analysis in analyses:
            try:
                if self._process_single_signal(analysis, current_prices):
                    trades_created += 1
            except Exception as e:
                log_error(f"Error processing signal for {analysis.symbol}: {e}")
        
        return trades_created
    
    def _process_single_signal(self, analysis: EnhancedNewsAnalysis, 
                              current_prices: pd.DataFrame) -> bool:
        """Process a single trading signal with comprehensive validation."""
        symbol = analysis.symbol
        
        # Skip if no technical analysis
        if analysis.technical_analysis is None:
            log_warning(f"Skipping {symbol} - no technical analysis")
            return False
        
        # Determine trade direction
        side = self._determine_trade_side(analysis)
        if not side:
            return False
        
        # Calculate position size
        position_size = self._determine_position_size(analysis)
        
        # Portfolio risk check
        if not self._check_portfolio_limits(symbol, side, position_size):
            return False
        
        # Get current price
        current_price = self._get_current_price(symbol, current_prices)
        if current_price <= 0:
            return False
        
        # Entry timing check
        if not self._check_entry_timing(symbol, analysis, current_price):
            return False
        
        # Technical validation
        if not self._validate_technical_alignment(analysis):
            return False
        
        # Create and execute trade
        return self._create_trade(analysis, side, position_size, current_price)
    
    def _determine_trade_side(self, analysis: EnhancedNewsAnalysis) -> Optional[str]:
        """Determine trade direction based on sentiment and confidence."""
        sentiment = analysis.sentiment_score
        confidence = analysis.combined_confidence
        min_confidence = CONFIG.min_confidence_score
        if confidence < min_confidence:
            return None
        
        if sentiment > 0.25:
            return 'long'
        elif sentiment < -0.25:
            return 'short'
        
        return None
    
    def _check_portfolio_limits(self, symbol: str, side: str, position_size: float) -> bool:
        """Check portfolio risk limits."""
        try:
            within_limits, reason = self.portfolio_filter.check_portfolio_limits(
                symbol, side, position_size
            )
            
            if not within_limits:
                log_info(f"Portfolio limit violation for {symbol}: {reason}")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking portfolio limits for {symbol}: {e}")
            return False
    
    def _get_current_price(self, symbol: str, current_prices: pd.DataFrame) -> float:
        """Get current price for symbol with validation."""
        try:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                return 0.0
            
            current_price = float(price_row.iloc[0].get('lastSalePrice', 0))
            return current_price if current_price > 0 else 0.0
            
        except Exception as e:
            log_error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _check_entry_timing(self, symbol: str, analysis: EnhancedNewsAnalysis, 
                           current_price: float) -> bool:
        """Check entry timing optimization."""
        try:
            should_enter, timing_reason = self.timing_optimizer.should_enter_now(
                symbol, analysis, current_price
            )
            
            if not should_enter:
                log_info(f"Entry timing not optimal for {symbol}: {timing_reason}")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking entry timing for {symbol}: {e}")
            return False
    
    def _validate_technical_alignment(self, analysis: EnhancedNewsAnalysis) -> bool:
        """Validate technical and sentiment alignment."""
        try:
            sentiment = analysis.sentiment_score
            tech = analysis.technical_analysis
            
            if not tech:
                return False
            
            momentum = tech.momentum_score
            
            # Skip if sentiment and momentum strongly disagree
            if sentiment > 0.5 and momentum < -0.3:
                log_info(f"Skipping {analysis.symbol}: bullish news but bearish momentum")
                return False
            elif sentiment < -0.5 and momentum > 0.3:
                log_info(f"Skipping {analysis.symbol}: bearish news but bullish momentum")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error validating technical alignment: {e}")
            return False
    
    def _create_trade(self, analysis: EnhancedNewsAnalysis, side: str, 
                     position_size: float, current_price: float) -> bool:
        """Create and log a new trade."""
        try:
            # Get market conditions for logging
            market_conditions = self.market_filter.get_market_conditions()
            
            # Calculate entry price with slippage
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            
            # Safe access to technical analysis data with defaults
            tech = analysis.technical_analysis
            if tech is not None:
                tech_confidence = tech.technical_confidence
                liquidity_score = tech.liquidity_score
                momentum_score = tech.momentum_score
                volume_score = tech.volume_score
                rsi = tech.rsi
                price_trend = tech.price_trend
                bid_ask_spread = tech.bid_ask_spread
            else:
                # Default values when technical analysis is not available
                tech_confidence = 0.0
                liquidity_score = 0.0
                momentum_score = 0.0
                volume_score = 0.0
                rsi = 50.0
                price_trend = "unknown"
                bid_ask_spread = 0.05
            
            # Create trade record
            trade = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                # News data
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                # Technical data (safely accessed)
                technical_confidence=tech_confidence,
                liquidity_score=liquidity_score,
                momentum_score=momentum_score,
                volume_score=volume_score,
                rsi=rsi,
                price_trend=price_trend,
                bid_ask_spread=bid_ask_spread,
                
                # Market conditions
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing="trade_created"
            )
            
            # Add to active trades
            self.active_trades[analysis.symbol] = trade
            
            # Log trade
            self._log_trade(trade)
            
            # Log entry message
            log_info(f"[ENTRY] {side} {analysis.symbol} @ ${entry_price:.2f} "
                    f"size=${position_size:.0f} | "
                    f"conf={analysis.combined_confidence:.2f} | "
                    f"regime={market_conditions.market_regime}")
            
            return True
            
        except Exception as e:
            log_error(f"Error creating trade for {analysis.symbol}: {e}")
            return False
    
    def check_exits(self, current_prices: pd.DataFrame) -> None:
        """Enhanced exit logic with dynamic adjustments."""
        if current_prices is None or current_prices.empty or not self.active_trades:
            return
        
        try:
            # Get market conditions for exit adjustments
            market_conditions = self.market_filter.get_market_conditions()
            stress_multiplier = self._calculate_stress_multiplier(market_conditions)
            
            trades_to_close = []
            
            for symbol, trade in self.active_trades.items():
                try:
                    if self._check_trade_exit(trade, current_prices, stress_multiplier):
                        trades_to_close.append(symbol)
                except Exception as e:
                    log_error(f"Error checking exit for {symbol}: {e}")
            
            # Close trades
            for symbol in trades_to_close:
                del self.active_trades[symbol]
                
        except Exception as e:
            log_error(f"Error in exit checking: {e}")
    
    def _calculate_stress_multiplier(self, market_conditions) -> float:
        """Calculate stress-based exit multiplier."""
        stress_level = market_conditions.market_stress_level
        
        if stress_level > 0.8:
            return 0.7  # Very tight stops
        elif stress_level > 0.6:
            return 0.8  # Tight stops
        else:
            return 1.0  # Normal stops
    
    def _check_trade_exit(self, trade: EnhancedTrade, current_prices: pd.DataFrame, 
                         stress_multiplier: float) -> bool:
        """Check if a trade should be exited."""
        # Get current price
        current_price = self._get_current_price(trade.symbol, current_prices)
        if current_price <= 0:
            return False
        
        # Calculate P&L percentage
        if trade.side == 'long':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
        
        # Dynamic stop/target adjustments
        stop_loss_pct = CONFIG.stop_loss_pct * stress_multiplier
        take_profit_pct = CONFIG.take_profit_pct
        
        # Adjust for trade quality
        if trade.liquidity_score < 0.5:
            stop_loss_pct *= 0.8
            take_profit_pct *= 0.8
        
        if trade.combined_confidence > 0.9:
            take_profit_pct *= 1.2
        
        # Check exit conditions
        exit_reason = None
        
        if pnl_pct <= -stop_loss_pct:
            exit_reason = f'stop_loss_{stress_multiplier:.1f}x'
        elif pnl_pct >= take_profit_pct:
            exit_reason = 'take_profit'
        elif stress_multiplier < 0.8 and pnl_pct > 0.02:  # Stress protection
            exit_reason = 'market_stress_protect'
        
        if exit_reason:
            # Execute exit
            trade.exit_price = current_price
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_reason = exit_reason
            trade.pnl = trade.position_size * pnl_pct
            
            # Log exit
            self._log_trade(trade)
            
            duration_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
            log_info(f"[EXIT] {trade.side} {trade.symbol} @ ${current_price:.2f} "
                    f"P&L=${trade.pnl:.2f} ({exit_reason}) | "
                    f"duration={duration_minutes:.1f}min")
            
            return True
        
        return False
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get list of active trades."""
        return list(self.active_trades.values())
    
    def get_daily_pnl(self) -> float:
        """Calculate today's P&L with error handling."""
        if not self.trade_log_file.exists():
            return 0.0
        
        try:
            today = datetime.now(timezone.utc).date()
            
            # Read trade log efficiently
            df = pd.read_csv(self.trade_log_file)
            
            if 'exit_time' not in df.columns or 'pnl' not in df.columns:
                return 0.0
            
            # Convert to datetime with timezone awareness
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', utc=True)
            
            # Filter for today's completed trades
            today_trades = df[
                (df['exit_time'].notna()) & 
                (df['exit_time'].dt.date == today)
            ]
            
            return float(today_trades['pnl'].fillna(0).sum())
            
        except Exception as e:
            log_error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def get_filter_performance(self) -> Dict[str, Any]:
        """Analyze filter effectiveness with comprehensive metrics."""
        if not self.trade_log_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.trade_log_file)
            completed_trades = df[df['exit_time'].notna()].copy()
            
            if completed_trades.empty:
                return {}
            
            # Ensure numeric columns
            completed_trades['pnl'] = pd.to_numeric(completed_trades['pnl'], errors='coerce')
            completed_trades['market_stress'] = pd.to_numeric(completed_trades['market_stress'], errors='coerce')
            completed_trades['time_score'] = pd.to_numeric(completed_trades['time_score'], errors='coerce')
            completed_trades['technical_confidence'] = pd.to_numeric(completed_trades['technical_confidence'], errors='coerce')
            
            performance_data = {}
            
            # Performance by market regime
            if 'market_regime' in completed_trades.columns:
                regime_perf = completed_trades.groupby('market_regime')['pnl'].agg(['count', 'sum', 'mean'])
                performance_data['regime_performance'] = regime_perf.to_dict()
            
            # Performance by entry timing
            if 'entry_timing' in completed_trades.columns:
                timing_perf = completed_trades.groupby('entry_timing')['pnl'].agg(['count', 'sum', 'mean'])
                performance_data['timing_performance'] = timing_perf.to_dict()
            
            # Performance by technical confidence buckets
            if 'technical_confidence' in completed_trades.columns:
                completed_trades['tech_conf_bucket'] = pd.cut(
                    completed_trades['technical_confidence'],
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['low', 'medium', 'high'],
                    include_lowest=True
                )
                tech_perf = completed_trades.groupby('tech_conf_bucket')['pnl'].agg(['count', 'sum', 'mean'])
                performance_data['technical_confidence_performance'] = tech_perf.to_dict()
            
            # Average metrics
            performance_data.update({
                'avg_market_stress': completed_trades['market_stress'].mean(),
                'avg_time_score': completed_trades['time_score'].mean(),
                'total_trades': len(completed_trades),
                'win_rate': (completed_trades['pnl'] > 0).mean(),
                'avg_pnl': completed_trades['pnl'].mean()
            })
            
            return performance_data
            
        except Exception as e:
            log_error(f"Error analyzing filter performance: {e}")
            return {}