"""
Enhanced trader with modular components
"""
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
import csv
import json
import time
from pathlib import Path
from config import CONFIG
from analysis.enhanced_news_analyzer import EnhancedNewsAnalysis
from analysis.technical_analyzer import TechnicalAnalysis
from analysis.market_filters import (
    MarketConditionFilter, NewsQualityFilter, PriceActionFilter, 
    PortfolioRiskFilter, EntryTimingOptimizer
)
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from .trade_models import EnhancedTrade, ProcessedArticle
from .recommendation_tracker import RecommendationTracker
from .position_manager import PositionManager


class EnhancedTrader:
    """Enhanced trader with modular components"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with modular components"""
        self.fmp_loader = fmp_loader
        self.trade_log_file = CONFIG.trade_log_path / CONFIG.trade_log_file
        self.trade_counter = 0
        
        # Initialize components
        self._initialize_components()
        
        # Tracking systems
        self.recommendation_tracker = RecommendationTracker()
        self.position_manager = PositionManager(fmp_loader)
        self.processed_symbols_today: Set[str] = set()
        self.daily_symbol_reset_time = datetime.now(timezone.utc).date()
        
        # Article combination settings
        self.article_combination_window = timedelta(hours=1.5)
        self.breaking_news_window = timedelta(minutes=10)
        
        self._setup_trade_log()
    
    def _initialize_components(self) -> None:
        """Initialize filter components"""
        try:
            self.market_filter = MarketConditionFilter(self.fmp_loader)
            self.news_filter = NewsQualityFilter()
            self.price_filter = PriceActionFilter(self.fmp_loader)
            self.portfolio_filter = PortfolioRiskFilter(self)
            self.timing_optimizer = EntryTimingOptimizer()
        except Exception as e:
            log_error(f"Error initializing filters: {e}")
            raise
    
    def _setup_trade_log(self) -> None:
        """Setup trade log CSV"""
        if self.trade_log_file.exists():
            return
        
        try:
            headers = [
                # Trade basics
                'trade_id', 'symbol', 'side', 'entry_price', 'position_size',
                'entry_time', 'exit_price', 'exit_time', 'exit_reason', 'pnl',
                
                # Trade execution status
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
                
            log_info(f"Created trade log with {len(headers)} columns")
                    
        except Exception as e:
            log_error(f"Error setting up trade log: {e}")
            raise
    
    def _log_trade(self, trade: EnhancedTrade) -> None:
        """Log trade data to CSV"""
        try:
            trade_data = [
                trade.id, trade.symbol, trade.side, trade.entry_price,
                trade.position_size, trade.entry_time, trade.exit_price,
                trade.exit_time, trade.exit_reason, trade.pnl,
                
                trade.trade_type, trade.execution_status, trade.rejection_reason,
                
                trade.news_title[:100], trade.news_confidence, trade.combined_confidence,
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
        """Generate unique trade ID"""
        self.trade_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d')
        return f"T{timestamp}_{self.trade_counter:04d}"
    
    def _reset_daily_tracking_if_needed(self) -> None:
        """Reset daily symbol tracking if new day"""
        current_date = datetime.now(timezone.utc).date()
        
        if current_date != self.daily_symbol_reset_time:
            self.processed_symbols_today.clear()
            self.daily_symbol_reset_time = current_date
            log_info("Reset daily symbol tracking for new day")
    
    def pre_filter_news(self, news_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply news quality filters"""
        if news_df is None or news_df.empty:
            return news_df
        
        try:
            log_info(f"Pre-filter input: {len(news_df)} articles")
            
            filtered_news = self.news_filter.filter_news_quality(news_df)
            
            if filtered_news is not None:
                original_count = len(news_df)
                filtered_count = len(filtered_news)
                
                if filtered_count < original_count:
                    log_info(f"News pre-filter: {original_count} -> {filtered_count} articles")
                    
                    if filtered_count == 0:
                        log_warning("ALL ARTICLES FILTERED OUT")
                        self._debug_filter_rejections(news_df)
                
                return filtered_news
            else:
                log_warning("News filter returned None")
                return pd.DataFrame()
                
        except Exception as e:
            log_error(f"Error in news pre-filtering: {e}")
            return news_df
    
    def _debug_filter_rejections(self, news_df: pd.DataFrame) -> None:
        """Debug why articles are being filtered out"""
        current_time = datetime.now(timezone.utc)
        
        log_warning("=== FILTER REJECTION ANALYSIS ===")
        
        for idx, row in news_df.iterrows():
            symbol = row.get('symbol', 'UNKNOWN')
            title = str(row.get('title', ''))
            text = str(row.get('text', ''))
            published_date = row.get('publishedDate')
            
            rejection_reasons = []
            
            # Check basic requirements
            if len(title) < 20:
                rejection_reasons.append(f"Title too short ({len(title)} chars)")
            
            if len(text) < 100:
                rejection_reasons.append(f"Text too short ({len(text)} chars)")
            
            # Check staleness
            if published_date:
                try:
                    pub_time = pd.to_datetime(published_date, utc=True)
                    time_diff = current_time - pub_time
                    cutoff_hours = 24 if CONFIG.testing_mode else 2
                    
                    if time_diff.total_seconds() / 3600 > cutoff_hours:
                        rejection_reasons.append(f"Too old ({time_diff.total_seconds()/3600:.1f}h > {cutoff_hours}h)")
                except:
                    rejection_reasons.append("Invalid published date")
            
            # Log analysis
            title_short = title[:40] + "..." if len(title) > 40 else title
            if rejection_reasons:
                log_warning(f"  {symbol}: {title_short}")
                for reason in rejection_reasons:
                    log_warning(f"    ❌ {reason}")
            else:
                log_warning(f"  {symbol}: {title_short} ✅ (should pass)")
    
    def process_news_signals(self, analyses: List[EnhancedNewsAnalysis], 
                           current_prices: pd.DataFrame) -> None:
        """Process news signals with optimized flow"""
        if not analyses or current_prices is None or current_prices.empty:
            return
        
        try:
            # Reset daily tracking if needed
            self._reset_daily_tracking_if_needed()
            
            # Deduplicate signals
            deduplicated_analyses = self._deduplicate_signals(analyses)
            
            # Filter recent recommendations
            filtered_analyses = self._filter_recent_recommendations(deduplicated_analyses)
            
            # Log recommendations
            recommendations_logged = self._log_all_recommendations(filtered_analyses, current_prices)
            
            # Check market conditions
            market_favorable = self._check_market_conditions()
            
            if not market_favorable:
                log_info(f"[RECOMMENDATIONS] Logged {recommendations_logged} (market unfavorable)")
                return
            
            # Process for live trading
            trades_created = self._process_filtered_signals(filtered_analyses, current_prices, is_live=True)
            
            # Mark symbols as processed
            for analysis in filtered_analyses:
                self.recommendation_tracker.mark_recommended(analysis.symbol)
                self.processed_symbols_today.add(analysis.symbol)
            
            self.timing_optimizer.cleanup_stale_entries()
            
            if trades_created > 0:
                log_info(f"[TRADES] Created {trades_created} LIVE trades")
            elif filtered_analyses:
                log_info(f"[FILTER] No LIVE trades created (recommendations logged)")
                
        except Exception as e:
            log_error(f"Error processing news signals: {e}")
    
    def _deduplicate_signals(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Deduplicate signals by symbol"""
        if not analyses:
            return analyses
        
        # Simple deduplication - keep highest confidence per symbol
        symbol_best = {}
        
        for analysis in analyses:
            symbol = analysis.symbol
            if symbol not in symbol_best or analysis.combined_confidence > symbol_best[symbol].combined_confidence:
                symbol_best[symbol] = analysis
        
        result = sorted(symbol_best.values(), key=lambda x: x.combined_confidence, reverse=True)
        
        if len(result) < len(analyses):
            log_info(f"Deduplication: {len(analyses)} -> {len(result)} signals")
        
        return result
    
    def _filter_recent_recommendations(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Filter out symbols with recent recommendations"""
        filtered = []
        
        for analysis in analyses:
            # Check recommendation tracker
            if not self.recommendation_tracker.should_recommend(analysis.symbol):
                continue
            
            # Check daily processing
            if analysis.symbol in self.processed_symbols_today:
                log_debug(f"Already processed {analysis.symbol} today")
                continue
            
            filtered.append(analysis)
        
        return filtered
    
    def _log_all_recommendations(self, analyses: List[EnhancedNewsAnalysis], 
                               current_prices: pd.DataFrame) -> int:
        """Log all recommendations"""
        recommendations_count = 0
        
        try:
            market_conditions = self.market_filter.get_market_conditions()
            
            for analysis in analyses:
                try:
                    recommendation = self._create_recommendation(analysis, current_prices, market_conditions)
                    if recommendation:
                        self._log_trade(recommendation)
                        recommendations_count += 1
                        
                        log_info(f"[RECOMMENDATION] {recommendation.side.upper()} {analysis.symbol} @ ${recommendation.entry_price:.2f} "
                                f"conf={analysis.combined_confidence:.2f}")
                        
                except Exception as e:
                    log_error(f"Error creating recommendation for {analysis.symbol}: {e}")
                    continue
            
            return recommendations_count
            
        except Exception as e:
            log_error(f"Error logging recommendations: {e}")
            return 0
    
    def _create_recommendation(self, analysis: EnhancedNewsAnalysis, current_prices: pd.DataFrame, 
                             market_conditions) -> Optional[EnhancedTrade]:
        """Create recommendation trade record"""
        try:
            side = self._determine_trade_side(analysis)
            if not side:
                return None
            
            current_price = self._get_current_price(analysis.symbol, current_prices)
            if current_price <= 0:
                return None
            
            position_size = self._determine_position_size(analysis)
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            tech_data = self._extract_technical_data(analysis.technical_analysis)
            
            # Determine rejection reason
            rejection_reason = ""
            if not market_conditions.is_market_hours:
                rejection_reason = "MARKET_CLOSED"
            elif market_conditions.market_stress_level > 0.85:
                rejection_reason = "HIGH_MARKET_STRESS"
            elif market_conditions.market_regime == 'volatile':
                rejection_reason = "VOLATILE_REGIME"
            
            recommendation = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                trade_type="RECOMMENDATION",
                execution_status="PENDING",
                rejection_reason=rejection_reason,
                
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                **tech_data,
                
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
        """Check if market conditions favor trading"""
        try:
            market_conditions = self.market_filter.get_market_conditions()
            should_trade, reason = self.market_filter.should_trade_now(market_conditions)
            
            if not should_trade:
                log_info(f"Market conditions unfavorable: {reason}")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking market conditions: {e}")
            return False
    
    def _process_filtered_signals(self, analyses: List[EnhancedNewsAnalysis], 
                                 current_prices: pd.DataFrame, is_live: bool = False) -> int:
        """Process filtered signals for trading"""
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
        """Process single trading signal"""
        symbol = analysis.symbol
        
        # Validation checks
        if analysis.technical_analysis is None:
            log_warning(f"Skipping {symbol} - no technical analysis")
            return False
        
        side = self._determine_trade_side(analysis)
        if not side:
            return False
        
        position_size = self._determine_position_size(analysis)
        
        if is_live and not self._check_portfolio_limits(symbol, position_size):
            return False
        
        current_price = self._get_current_price(symbol, current_prices)
        if current_price <= 0:
            return False
        
        if is_live and not self._check_entry_timing(symbol, analysis):
            return False
        
        if not self._validate_technical_alignment(analysis):
            return False
        
        return self._create_trade(analysis, side, position_size, current_price, is_live)
    
    def _determine_trade_side(self, analysis: EnhancedNewsAnalysis) -> Optional[str]:
        """Determine trade direction"""
        sentiment = analysis.sentiment_score
        confidence = analysis.combined_confidence
        
        if confidence < CONFIG.min_confidence_score:
            return None
        
        if sentiment > 0.20:
            return 'long'
        elif sentiment < -0.20:
            return 'short'
        
        return None
    
    def _determine_position_size(self, analysis: EnhancedNewsAnalysis) -> float:
        """Calculate position size with risk management"""
        try:
            base_size = CONFIG.position_size
            confidence_multiplier = analysis.combined_confidence
            
            if analysis.technical_analysis:
                tech = analysis.technical_analysis
                volatility_factor = max(0.5, 1.0 - (tech.volatility_score * 0.25))
                liquidity_factor = max(0.6, tech.liquidity_score)
                confidence_multiplier *= volatility_factor * liquidity_factor
            
            # Topic multipliers
            topic_multipliers = {
                'earnings': 1.08, 'biotech': 1.15, 'ma': 1.12,
                'analyst': 0.92, 'general': 0.96
            }
            
            topic_mult = topic_multipliers.get(analysis.topic, 1.0)
            confidence_multiplier *= topic_mult
            
            final_multiplier = max(
                CONFIG.min_position_multiplier,
                min(CONFIG.max_position_multiplier, confidence_multiplier)
            )
            
            return base_size * final_multiplier
            
        except Exception as e:
            log_error(f"Error calculating position size: {e}")
            return CONFIG.position_size * CONFIG.min_position_multiplier
    
    def _check_portfolio_limits(self, symbol: str, position_size: float) -> bool:
        """Check portfolio risk limits"""
        try:
            within_limits, reason = self.portfolio_filter.check_portfolio_limits(
                symbol, position_size
            )
            
            if not within_limits:
                log_info(f"Portfolio limit violation for {symbol}: {reason}")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking portfolio limits for {symbol}: {e}")
            return False
    
    def _get_current_price(self, symbol: str, current_prices: pd.DataFrame) -> float:
        """Get current price for symbol"""
        try:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if price_row.empty:
                return 0.0
            
            current_price = float(price_row.iloc[0].get('lastSalePrice', 0))
            return current_price if current_price > 0 else 0.0
            
        except (ValueError, IndexError, KeyError) as e:
            log_error(f"Error getting current price for {symbol}: {e}")
            return 0.0
    
    def _check_entry_timing(self, symbol: str, analysis: EnhancedNewsAnalysis) -> bool:
        """Check entry timing"""
        try:
            should_enter, timing_reason = self.timing_optimizer.should_enter_now(
                symbol, analysis
            )
            
            if not should_enter:
                log_info(f"Entry timing not optimal for {symbol}: {timing_reason}")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking entry timing for {symbol}: {e}")
            return False
    
    def _validate_technical_alignment(self, analysis: EnhancedNewsAnalysis) -> bool:
        """Validate technical and sentiment alignment"""
        try:
            sentiment = analysis.sentiment_score
            tech = analysis.technical_analysis
            
            if not tech:
                return False
            
            momentum = tech.momentum_score
            
            # Check for major conflicts
            if sentiment > 0.4 and momentum < -0.4:
                log_info(f"Skipping {analysis.symbol}: strong bullish news but strong bearish momentum")
                return False
            elif sentiment < -0.4 and momentum > 0.4:
                log_info(f"Skipping {analysis.symbol}: strong bearish news but strong bullish momentum")
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error validating technical alignment: {e}")
            return False
    
    def _create_trade(self, analysis: EnhancedNewsAnalysis, side: str, 
                     position_size: float, current_price: float, is_live: bool = False) -> bool:
        """Create and log trade"""
        try:
            market_conditions = self.market_filter.get_market_conditions()
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            tech_data = self._extract_technical_data(analysis.technical_analysis)
            
            trade_type = "LIVE" if is_live else "PAPER"
            execution_status = "EXECUTED" if is_live else "SIMULATED"
            
            trade = EnhancedTrade(
                id=self._generate_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                trade_type=trade_type,
                execution_status=execution_status,
                rejection_reason="",
                
                news_title=analysis.title[:100],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                **tech_data,
                
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing="trade_created"
            )
            
            if is_live:
                self.position_manager.add_position(trade)
            
            self._log_trade(trade)
            
            trade_type_label = "LIVE" if is_live else "SIMULATED"
            log_info(f"[{trade_type_label}] {side} {analysis.symbol} @ ${entry_price:.2f} "
                    f"size=${position_size:.0f} | conf={analysis.combined_confidence:.2f}")
            
            return True
            
        except Exception as e:
            log_error(f"Error creating trade for {analysis.symbol}: {e}")
            return False
    
    def _extract_technical_data(self, technical_analysis: Optional[TechnicalAnalysis]) -> Dict[str, Any]:
        """Extract technical analysis data with safe defaults"""
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
        """Check position exits using position manager"""
        self.position_manager.check_exits(current_prices)
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get active trades"""
        return self.position_manager.get_active_positions()
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L"""
        if not self.trade_log_file.exists():
            return 0.0
        
        try:
            today = datetime.now(timezone.utc).date()
            df = pd.read_csv(self.trade_log_file)
            
            if 'exit_time' not in df.columns or 'pnl' not in df.columns:
                return 0.0
            
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', utc=True)
            
            today_trades = df[
                (df['exit_time'].notna()) & 
                (df['exit_time'].dt.date == today)
            ]
            
            return float(today_trades['pnl'].fillna(0).sum())
            
        except Exception as e:
            log_error(f"Error calculating daily P&L: {e}")
            return 0.0
    
    def get_filter_performance(self) -> Dict[str, Any]:
        """Analyze filter effectiveness"""
        if not self.trade_log_file.exists():
            return {}
        
        try:
            df = pd.read_csv(self.trade_log_file)
            
            if df.empty or 'exit_time' not in df.columns:
                return {}
            
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
            if 'market_regime' in completed_trades.columns:
                regime_perf = completed_trades.groupby('market_regime')['pnl'].agg(['count', 'sum', 'mean'])
                performance_data['regime_performance'] = regime_perf.to_dict()
            
            return performance_data
            
        except Exception as e:
            log_error(f"Error analyzing filter performance: {e}")
            return {}


# Create alias for backwards compatibility
OptimizedEnhancedTrader = EnhancedTrader