"""
Optimized trader with comprehensive filtering and improved performance
"""
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
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
    
    # Trade execution status
    trade_type: str = "LIVE"  # "LIVE", "PAPER", "RECOMMENDATION"
    execution_status: str = "PENDING"  # "PENDING", "EXECUTED", "REJECTED"
    rejection_reason: str = ""
    
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
        if self.trade_log_file.exists():
            return
        
        try:
            headers = [
                # Trade basics
                'trade_id', 'symbol', 'side', 'entry_price', 'position_size',
                'entry_time', 'exit_price', 'exit_time', 'exit_reason', 'pnl',
                
                # Trade execution status (NEW COLUMNS)
                'trade_type', 'execution_status', 'rejection_reason',
                
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
                csv.writer(f).writerow(headers)
                
            log_info(f"Created new trade log with {len(headers)} columns including new execution tracking")
                    
        except Exception as e:
            log_error(f"Error setting up trade log: {e}")
            raise
    
    def _log_trade(self, trade: EnhancedTrade) -> None:
        """Log comprehensive trade data to CSV with error handling."""
        try:
            trade_data = [
                trade.id, trade.symbol, trade.side, trade.entry_price,
                trade.position_size, trade.entry_time, trade.exit_price,
                trade.exit_time, trade.exit_reason, trade.pnl,
                
                # Trade execution status (NEW COLUMNS)
                trade.trade_type, trade.execution_status, trade.rejection_reason,
                
                trade.news_title, trade.news_confidence, trade.combined_confidence,
                trade.finbert_score, trade.keyword_score, trade.topic,
                
                trade.technical_confidence, trade.liquidity_score, trade.momentum_score,
                trade.volume_score, trade.rsi, trade.price_trend, trade.bid_ask_spread,
                
                trade.market_regime, trade.market_stress, trade.time_score, trade.entry_timing
            ]
            
            with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(trade_data)
                
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
            filtered_news = self.news_filter.filter_news_quality(news_df)
            
            if filtered_news is not None:
                original_count = len(news_df)
                filtered_count = len(filtered_news)
                
                if filtered_count < original_count:
                    log_info(f"News pre-filter: {original_count} -> {filtered_count} articles")
                
                return filtered_news
            else:
                log_warning("News filter returned None")
                return pd.DataFrame()
                
        except Exception as e:
            log_error(f"Error in news pre-filtering: {e}")
            return news_df
    
    def _determine_position_size(self, analysis: EnhancedNewsAnalysis) -> float:
        """Enhanced dynamic position sizing with risk management."""
        try:
            base_size = CONFIG.position_size
            confidence_multiplier = analysis.combined_confidence
            
            # Technical adjustments
            if analysis.technical_analysis:
                tech = analysis.technical_analysis
                
                # Calculate adjustment factors
                volatility_factor = max(0.5, 1.0 - (tech.volatility_score * 0.3))
                liquidity_factor = max(0.5, tech.liquidity_score)
                
                # Momentum alignment bonus
                momentum_boost = self._calculate_momentum_boost(analysis.sentiment_score, tech.momentum_score)
                
                confidence_multiplier *= volatility_factor * liquidity_factor * momentum_boost
            
            # Topic-based multipliers
            topic_multipliers = {
                'earnings': 1.1, 'biotech': 1.2, 'ma': 1.15,
                'analyst': 0.9, 'general': 0.95
            }
            
            topic_mult = topic_multipliers.get(analysis.topic, 1.0)
            confidence_multiplier *= topic_mult
            
            # Apply bounds
            final_multiplier = max(
                CONFIG.min_position_multiplier,
                min(CONFIG.max_position_multiplier, confidence_multiplier)
            )
            
            return base_size * final_multiplier
            
        except Exception as e:
            log_error(f"Error calculating position size: {e}")
            return CONFIG.position_size * CONFIG.min_position_multiplier
    
    def _calculate_momentum_boost(self, sentiment_score: float, momentum_score: float) -> float:
        """Calculate momentum alignment boost."""
        if abs(sentiment_score) > 0.3 and abs(momentum_score) > 0.2:
            if (sentiment_score > 0) == (momentum_score > 0):
                return 1.15  # Aligned signals
        return 1.0
    
    def process_news_signals(self, analyses: List[EnhancedNewsAnalysis], 
                           current_prices: pd.DataFrame) -> None:
        """Process news signals with deduplication and ALWAYS log recommendations to CSV for backtesting."""
        if not analyses or current_prices is None or current_prices.empty:
            return
        
        try:
            # STEP 1: Deduplicate signals by symbol (take highest confidence per symbol)
            deduplicated_analyses = self._deduplicate_signals_by_symbol(analyses)
            
            if len(deduplicated_analyses) < len(analyses):
                log_info(f"Deduplicated signals: {len(analyses)} -> {len(deduplicated_analyses)} (removed duplicates)")
            
            # STEP 2: Always log all deduplicated recommendations regardless of market conditions
            recommendations_logged = self._log_all_recommendations(deduplicated_analyses, current_prices)
            
            # STEP 3: Then check if we should actually execute trades
            market_favorable = self._check_market_conditions()
            
            if not market_favorable:
                log_info(f"[RECOMMENDATIONS] Logged {recommendations_logged} unique trading recommendations for backtesting")
                return
            
            # STEP 4: If market conditions are favorable, process for actual execution
            filtered_analyses = self._apply_price_action_filter(deduplicated_analyses, current_prices)
            trades_created = self._process_filtered_signals(filtered_analyses, current_prices, is_live=True)
            
            self.timing_optimizer.cleanup_stale_entries()
            
            if trades_created > 0:
                log_info(f"[TRADES] Created {trades_created} LIVE trades from {len(deduplicated_analyses)} signals")
            elif deduplicated_analyses:
                log_info(f"[FILTER] No LIVE trades created from {len(deduplicated_analyses)} signals (but recommendations logged)")
                
        except Exception as e:
            log_error(f"Error processing news signals: {e}")
    
    def _deduplicate_signals_by_symbol(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Deduplicate signals by symbol, keeping the highest confidence signal for each symbol."""
        if not analyses:
            return analyses
        
        try:
            # Group by symbol and keep highest confidence
            symbol_best_signals = {}
            
            for analysis in analyses:
                symbol = analysis.symbol
                current_confidence = analysis.combined_confidence
                
                if symbol not in symbol_best_signals:
                    symbol_best_signals[symbol] = analysis
                else:
                    # Keep the analysis with higher confidence
                    existing_confidence = symbol_best_signals[symbol].combined_confidence
                    if current_confidence > existing_confidence:
                        log_info(f"Updated {symbol} signal: conf {existing_confidence:.3f} -> {current_confidence:.3f}")
                        symbol_best_signals[symbol] = analysis
                    else:
                        log_info(f"Kept existing {symbol} signal: conf {existing_confidence:.3f} > {current_confidence:.3f}")
            
            deduplicated = list(symbol_best_signals.values())
            
            # Sort by confidence descending
            deduplicated.sort(key=lambda x: x.combined_confidence, reverse=True)
            
            return deduplicated
            
        except Exception as e:
            log_error(f"Error deduplicating signals: {e}")
            return analyses
    
    def _log_all_recommendations(self, analyses: List[EnhancedNewsAnalysis], 
                               current_prices: pd.DataFrame) -> int:
        """Log all trading recommendations to CSV regardless of market conditions."""
        recommendations_count = 0
        
        try:
            # Get market conditions for logging
            market_conditions = self.market_filter.get_market_conditions()
            
            for analysis in analyses:
                try:
                    recommendation = self._create_recommendation(analysis, current_prices, market_conditions)
                    if recommendation:
                        self._log_trade(recommendation)
                        recommendations_count += 1
                        
                        log_info(f"[RECOMMENDATION] {recommendation.side.upper()} {analysis.symbol} @ ${recommendation.entry_price:.2f} "
                                f"conf={analysis.combined_confidence:.2f} (logged for backtesting)")
                        
                except Exception as e:
                    log_error(f"Error creating recommendation for {analysis.symbol}: {e}")
                    continue
            
            return recommendations_count
            
        except Exception as e:
            log_error(f"Error logging recommendations: {e}")
            return 0
    
    def _create_recommendation(self, analysis: EnhancedNewsAnalysis, current_prices: pd.DataFrame, 
                             market_conditions) -> Optional[EnhancedTrade]:
        """Create a recommendation trade record for logging."""
        try:
            # Determine trade direction
            side = self._determine_trade_side(analysis)
            if not side:
                return None
            
            # Get current price
            current_price = self._get_current_price(analysis.symbol, current_prices)
            if current_price <= 0:
                return None
            
            # Calculate position size
            position_size = self._determine_position_size(analysis)
            
            # Calculate entry price with slippage
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            
            # Extract technical data safely
            tech_data = self._extract_technical_data(analysis.technical_analysis)
            
            # Determine trade type based on market conditions
            trade_type = "RECOMMENDATION"  # Always log as recommendation first
            execution_status = "PENDING"
            rejection_reason = ""
            
            # Check various rejection reasons for logging
            if not market_conditions.is_market_hours:
                rejection_reason = "MARKET_CLOSED"
            elif market_conditions.market_stress_level > 0.85:
                rejection_reason = "HIGH_MARKET_STRESS"
            elif market_conditions.market_regime == 'volatile':
                rejection_reason = "VOLATILE_REGIME"
            
            # Create recommendation record
            recommendation = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                # Trade execution info
                trade_type=trade_type,
                execution_status=execution_status,
                rejection_reason=rejection_reason,
                
                # News data
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                # Technical data
                **tech_data,
                
                # Market conditions
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing="recommendation_logged"
            )
            
            return recommendation
            
        except Exception as e:
            log_error(f"Error creating recommendation for {analysis.symbol}: {e}")
            return None
    
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
                                 current_prices: pd.DataFrame, is_live: bool = False) -> int:
        """Process filtered signals and create trades."""
        trades_created = 0
        
        for analysis in analyses:
            try:
                if self._process_single_signal(analysis, current_prices, is_live):
                    trades_created += 1
            except Exception as e:
                log_error(f"Error processing signal for {analysis.symbol}: {e}")
        
        return trades_created
    
    def _process_single_signal(self, analysis: EnhancedNewsAnalysis, 
                              current_prices: pd.DataFrame, is_live: bool = False) -> bool:
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
        
        # Portfolio risk check (only for live trades)
        if is_live and not self._check_portfolio_limits(symbol, side, position_size):
            return False
        
        # Get current price
        current_price = self._get_current_price(symbol, current_prices)
        if current_price <= 0:
            return False
        
        # Entry timing check (only for live trades)
        if is_live and not self._check_entry_timing(symbol, analysis, current_price):
            return False
        
        # Technical validation
        if not self._validate_technical_alignment(analysis):
            return False
        
        # Create and execute trade
        return self._create_trade(analysis, side, position_size, current_price, is_live)
    
    def _determine_trade_side(self, analysis: EnhancedNewsAnalysis) -> Optional[str]:
        """Determine trade direction based on sentiment and confidence."""
        sentiment = analysis.sentiment_score
        confidence = analysis.combined_confidence
        
        if confidence < CONFIG.min_confidence_score:
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
                     position_size: float, current_price: float, is_live: bool = False) -> bool:
        """Create and log a new trade."""
        try:
            # Get market conditions for logging
            market_conditions = self.market_filter.get_market_conditions()
            
            # Calculate entry price with slippage
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            
            # Extract technical data safely
            tech_data = self._extract_technical_data(analysis.technical_analysis)
            
            # Determine trade type and status
            trade_type = "LIVE" if is_live else "PAPER"
            execution_status = "EXECUTED" if is_live else "SIMULATED"
            
            # Create trade record
            trade = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                # Trade execution info
                trade_type=trade_type,
                execution_status=execution_status,
                rejection_reason="",
                
                # News data
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                # Technical data
                **tech_data,
                
                # Market conditions
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing="trade_created"
            )
            
            # Add to active trades only if live
            if is_live:
                self.active_trades[analysis.symbol] = trade
            
            # Always log trade
            self._log_trade(trade)
            
            # Log entry message
            trade_type_label = "LIVE" if is_live else "SIMULATED"
            log_info(f"[{trade_type_label}] {side} {analysis.symbol} @ ${entry_price:.2f} "
                    f"size=${position_size:.0f} | "
                    f"conf={analysis.combined_confidence:.2f} | "
                    f"regime={market_conditions.market_regime}")
            
            return True
            
        except Exception as e:
            log_error(f"Error creating trade for {analysis.symbol}: {e}")
            return False
    
    def _extract_technical_data(self, technical_analysis: Optional[TechnicalAnalysis]) -> Dict[str, Any]:
        """Extract technical analysis data safely."""
        if technical_analysis is not None:
            return {
                'technical_confidence': technical_analysis.technical_confidence,
                'liquidity_score': technical_analysis.liquidity_score,
                'momentum_score': technical_analysis.momentum_score,
                'volume_score': technical_analysis.volume_score,
                'rsi': technical_analysis.rsi,
                'price_trend': technical_analysis.price_trend,
                'bid_ask_spread': technical_analysis.bid_ask_spread
            }
        else:
            return {
                'technical_confidence': 0.0,
                'liquidity_score': 0.0,
                'momentum_score': 0.0,
                'volume_score': 0.0,
                'rsi': 50.0,
                'price_trend': "unknown",
                'bid_ask_spread': 0.05
            }
    
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
                self.active_trades.pop(symbol, None)
                
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
        pnl_pct = self._calculate_pnl_percentage(trade, current_price)
        
        # Dynamic stop/target adjustments
        stop_loss_pct, take_profit_pct = self._calculate_dynamic_levels(trade, stress_multiplier)
        
        # Check exit conditions
        exit_reason = self._determine_exit_reason(pnl_pct, stop_loss_pct, take_profit_pct, stress_multiplier)
        
        if exit_reason:
            self._execute_exit(trade, current_price, exit_reason, pnl_pct)
            return True
        
        return False
    
    def _calculate_pnl_percentage(self, trade: EnhancedTrade, current_price: float) -> float:
        """Calculate P&L percentage for a trade."""
        if trade.side == 'long':
            return (current_price - trade.entry_price) / trade.entry_price
        else:
            return (trade.entry_price - current_price) / trade.entry_price
    
    def _calculate_dynamic_levels(self, trade: EnhancedTrade, stress_multiplier: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        stop_loss_pct = CONFIG.stop_loss_pct * stress_multiplier
        take_profit_pct = CONFIG.take_profit_pct
        
        # Adjust for trade quality
        if trade.liquidity_score < 0.5:
            stop_loss_pct *= 0.8
            take_profit_pct *= 0.8
        
        if trade.combined_confidence > 0.9:
            take_profit_pct *= 1.2
        
        return stop_loss_pct, take_profit_pct
    
    def _determine_exit_reason(self, pnl_pct: float, stop_loss_pct: float, 
                             take_profit_pct: float, stress_multiplier: float) -> Optional[str]:
        """Determine exit reason based on conditions."""
        if pnl_pct <= -stop_loss_pct:
            return f'stop_loss_{stress_multiplier:.1f}x'
        elif pnl_pct >= take_profit_pct:
            return 'take_profit'
        elif stress_multiplier < 0.8 and pnl_pct > 0.02:  # Stress protection
            return 'market_stress_protect'
        
        return None
    
    def _execute_exit(self, trade: EnhancedTrade, current_price: float, 
                     exit_reason: str, pnl_pct: float) -> None:
        """Execute trade exit and log details."""
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
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get list of active trades."""
        return list(self.active_trades.values())
    
    def get_daily_pnl(self) -> float:
        """Calculate today's P&L with error handling."""
        if not self.trade_log_file.exists():
            return 0.0
        
        try:
            today = datetime.now(timezone.utc).date()
            
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
            numeric_columns = ['pnl', 'market_stress', 'time_score', 'technical_confidence']
            for col in numeric_columns:
                if col in completed_trades.columns:
                    completed_trades[col] = pd.to_numeric(completed_trades[col], errors='coerce')
            
            performance_data = {}
            
            # Performance analytics
            performance_data.update(self._calculate_regime_performance(completed_trades))
            performance_data.update(self._calculate_timing_performance(completed_trades))
            performance_data.update(self._calculate_technical_performance(completed_trades))
            performance_data.update(self._calculate_summary_metrics(completed_trades))
            
            return performance_data
            
        except Exception as e:
            log_error(f"Error analyzing filter performance: {e}")
            return {}
    
    def _calculate_regime_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance by market regime."""
        if 'market_regime' not in df.columns:
            return {}
        
        regime_perf = df.groupby('market_regime')['pnl'].agg(['count', 'sum', 'mean'])
        return {'regime_performance': regime_perf.to_dict()}
    
    def _calculate_timing_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance by entry timing."""
        if 'entry_timing' not in df.columns:
            return {}
        
        timing_perf = df.groupby('entry_timing')['pnl'].agg(['count', 'sum', 'mean'])
        return {'timing_performance': timing_perf.to_dict()}
    
    def _calculate_technical_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance by technical confidence buckets."""
        if 'technical_confidence' not in df.columns:
            return {}
        
        df['tech_conf_bucket'] = pd.cut(
            df['technical_confidence'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high'],
            include_lowest=True
        )
        tech_perf = df.groupby('tech_conf_bucket')['pnl'].agg(['count', 'sum', 'mean'])
        return {'technical_confidence_performance': tech_perf.to_dict()}
    
    def _calculate_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary performance metrics."""
        return {
            'avg_market_stress': df['market_stress'].mean() if 'market_stress' in df.columns else 0,
            'avg_time_score': df['time_score'].mean() if 'time_score' in df.columns else 0,
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean() if 'pnl' in df.columns else 0,
            'avg_pnl': df['pnl'].mean() if 'pnl' in df.columns else 0
        }