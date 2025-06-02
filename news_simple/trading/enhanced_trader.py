"""
Optimized enhanced trader with improved performance and robust error handling
"""
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Set
from datetime import datetime, timezone, timedelta
import csv
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
    """High-performance trader with optimized processing and robust risk management."""
    
    # Class-level constants for performance
    TRADE_LOG_HEADERS = [
        'trade_id', 'symbol', 'side', 'entry_price', 'position_size',
        'entry_time', 'exit_price', 'exit_time', 'exit_reason', 'pnl',
        'trade_type', 'execution_status', 'rejection_reason',
        'news_title', 'news_confidence', 'combined_confidence',
        'finbert_score', 'keyword_score', 'topic',
        'technical_confidence', 'liquidity_score', 'momentum_score',
        'volume_score', 'rsi', 'price_trend', 'bid_ask_spread',
        'market_regime', 'market_stress', 'time_score', 'entry_timing'
    ]
    
    # Position sizing multipliers by topic
    TOPIC_MULTIPLIERS = {
        'earnings': 1.08,
        'biotech': 1.15,
        'ma': 1.12,
        'analyst': 0.92,
        'general': 0.96
    }
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized configuration and error handling."""
        self.fmp_loader = fmp_loader
        self.trade_log_file = CONFIG.trade_log_path / CONFIG.trade_log_file
        self.trade_counter = 0
        
        # Initialize components with error handling
        self._initialize_components_safely()
        
        # Tracking systems
        self.recommendation_tracker = RecommendationTracker()
        self.position_manager = PositionManager(fmp_loader)
        
        # Daily tracking with automatic reset
        self.processed_symbols_today: Set[str] = set()
        self.daily_reset_date = datetime.now(timezone.utc).date()
        
        # Performance tracking
        self.performance_stats = {
            'total_analyses': 0,
            'trades_created': 0,
            'recommendations_logged': 0,
            'filter_rejections': 0
        }
        
        self._setup_trade_log_safely()
    
    def _initialize_components_safely(self) -> None:
        """Initialize filter components with robust error handling."""
        try:
            self.market_filter = MarketConditionFilter(self.fmp_loader)
            self.news_filter = NewsQualityFilter()
            self.price_filter = PriceActionFilter(self.fmp_loader)
            self.portfolio_filter = PortfolioRiskFilter(self)
            self.timing_optimizer = EntryTimingOptimizer()
            log_info("All trading components initialized successfully")
        except Exception as e:
            log_error(f"Error initializing trading components: {e}")
            raise
    
    def _setup_trade_log_safely(self) -> None:
        """Setup trade log with robust error handling."""
        try:
            if self.trade_log_file.exists():
                # Validate existing log file
                self._validate_existing_log()
                return
            
            # Create new log file
            with open(self.trade_log_file, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.TRADE_LOG_HEADERS)
            
            log_info(f"Created trade log with {len(self.TRADE_LOG_HEADERS)} columns")
            
        except Exception as e:
            log_error(f"Error setting up trade log: {e}")
            raise
    
    def _validate_existing_log(self) -> None:
        """Validate existing log file structure."""
        try:
            with open(self.trade_log_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                
                # Check if headers match expected structure
                if set(headers) != set(self.TRADE_LOG_HEADERS):
                    log_warning("Trade log headers don't match expected structure")
                
        except Exception as e:
            log_warning(f"Error validating trade log: {e}")
    
    def _log_trade_safely(self, trade: EnhancedTrade) -> None:
        """Log trade data with comprehensive error handling."""
        try:
            trade_data = [
                trade.id, trade.symbol, trade.side, trade.entry_price,
                trade.position_size, trade.entry_time, trade.exit_price,
                trade.exit_time, trade.exit_reason, trade.pnl,
                trade.trade_type, trade.execution_status, trade.rejection_reason,
                trade.news_title[:150], trade.news_confidence, trade.combined_confidence,
                trade.finbert_score, trade.keyword_score, trade.topic,
                trade.technical_confidence, trade.liquidity_score, trade.momentum_score,
                trade.volume_score, trade.rsi, trade.price_trend, trade.bid_ask_spread,
                trade.market_regime, trade.market_stress, trade.time_score, trade.entry_timing
            ]
            
            with open(self.trade_log_file, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(trade_data)
                
        except Exception as e:
            log_error(f"Error logging trade {trade.id}: {e}")
    
    def _generate_unique_trade_id(self) -> str:
        """Generate unique trade ID with timestamp."""
        self.trade_counter += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"T{timestamp}_{self.trade_counter:04d}"
    
    def _reset_daily_tracking_if_needed(self) -> None:
        """Reset daily symbol tracking if new day."""
        current_date = datetime.now(timezone.utc).date()
        
        if current_date != self.daily_reset_date:
            self.processed_symbols_today.clear()
            self.daily_reset_date = current_date
            log_info("Daily symbol tracking reset for new trading day")
    
    def pre_filter_news(self, news_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply optimized news quality filters with comprehensive logging."""
        if news_df is None or news_df.empty:
            return news_df
        
        try:
            initial_count = len(news_df)
            log_info(f"Pre-filtering {initial_count} news articles")
            
            # Apply quality filters
            filtered_news = self.news_filter.filter_news_quality(news_df)
            
            if filtered_news is not None and not filtered_news.empty:
                final_count = len(filtered_news)
                filter_rate = ((initial_count - final_count) / initial_count * 100)
                
                log_info(f"News pre-filter: {initial_count} → {final_count} articles ({filter_rate:.1f}% filtered)")
                
                # Get filter statistics for monitoring
                filter_stats = self.news_filter.get_filter_statistics(news_df, filtered_news)
                if filter_stats:
                    log_debug(f"Filter stats: {filter_stats}")
                
                return filtered_news
            else:
                log_warning("All articles filtered out by quality filters")
                self._analyze_filter_rejections(news_df)
                return pd.DataFrame()
                
        except Exception as e:
            log_error(f"Error in news pre-filtering: {e}")
            return news_df
    
    def _analyze_filter_rejections(self, news_df: pd.DataFrame) -> None:
        """Analyze why articles were rejected for debugging."""
        log_warning("=== FILTER REJECTION ANALYSIS ===")
        
        try:
            current_time = datetime.now(timezone.utc)
            
            for idx, row in news_df.head(10).iterrows():  # Analyze first 10 for performance
                symbol = row.get('symbol', 'UNKNOWN')
                title = str(row.get('title', ''))[:100]
                
                issues = []
                
                # Check basic requirements
                if len(title.strip()) < 10:
                    issues.append(f"Short title ({len(title)} chars)")
                
                if 'text' in row:
                    text_len = len(str(row['text']))
                    if text_len < 30:
                        issues.append(f"Short content ({text_len} chars)")
                
                # Check time criteria
                if 'publishedDate' in row:
                    try:
                        pub_date = pd.to_datetime(row['publishedDate'], utc=True)
                        age_hours = (current_time - pub_date).total_seconds() / 3600
                        time_limit = 48 if CONFIG.testing_mode else 8
                        
                        if age_hours > time_limit:
                            issues.append(f"Too old ({age_hours:.1f}h > {time_limit}h)")
                    except:
                        issues.append("Invalid date")
                
                if issues:
                    log_warning(f"  {symbol}: {title}... - Issues: {', '.join(issues)}")
                else:
                    log_warning(f"  {symbol}: {title}... - No obvious issues found")
            
        except Exception as e:
            log_error(f"Error analyzing filter rejections: {e}")
    
    def process_news_signals(self, analyses: List[EnhancedNewsAnalysis], 
                           current_prices: pd.DataFrame) -> None:
        """Optimized news signal processing with comprehensive tracking."""
        if not analyses or current_prices is None or current_prices.empty:
            return
        
        try:
            start_time = time.time()
            
            # Reset daily tracking
            self._reset_daily_tracking_if_needed()
            
            # Process signals through optimized pipeline
            processing_stats = self._process_signals_pipeline(analyses, current_prices)
            
            # Update performance stats
            self.performance_stats['total_analyses'] += len(analyses)
            self.performance_stats['trades_created'] += processing_stats.get('trades_created', 0)
            self.performance_stats['recommendations_logged'] += processing_stats.get('recommendations', 0)
            
            # Log performance metrics
            processing_time = time.time() - start_time
            log_info(f"Signal processing completed in {processing_time:.2f}s: {processing_stats}")
            
        except Exception as e:
            log_error(f"Error processing news signals: {e}")
    
    def _process_signals_pipeline(self, analyses: List[EnhancedNewsAnalysis], 
                                current_prices: pd.DataFrame) -> Dict[str, int]:
        """Optimized signal processing pipeline."""
        # Stage 1: Deduplication
        deduplicated = self._deduplicate_signals_optimized(analyses)
        
        # Stage 2: Filter recent recommendations
        filtered = self._filter_recent_recommendations_optimized(deduplicated)
        
        # Stage 3: Log all recommendations
        recommendations_count = self._log_recommendations_batch(filtered, current_prices)
        
        # Stage 4: Check market conditions
        market_favorable = self._check_market_conditions_cached()
        
        if not market_favorable:
            return {
                'total_signals': len(analyses),
                'after_dedup': len(deduplicated),
                'after_filter': len(filtered),
                'recommendations': recommendations_count,
                'trades_created': 0,
                'market_favorable': False
            }
        
        # Stage 5: Create live trades
        trades_created = self._create_live_trades_batch(filtered, current_prices)
        
        # Stage 6: Update tracking
        self._update_processing_tracking(filtered)
        
        return {
            'total_signals': len(analyses),
            'after_dedup': len(deduplicated),
            'after_filter': len(filtered),
            'recommendations': recommendations_count,
            'trades_created': trades_created,
            'market_favorable': True
        }
    
    def _deduplicate_signals_optimized(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Optimized signal deduplication using dict for O(1) lookups."""
        if not analyses:
            return analyses
        
        # Use dict for efficient deduplication
        symbol_best = {}
        
        for analysis in analyses:
            symbol = analysis.symbol
            if (symbol not in symbol_best or 
                analysis.combined_confidence > symbol_best[symbol].combined_confidence):
                symbol_best[symbol] = analysis
        
        result = sorted(symbol_best.values(), key=lambda x: x.combined_confidence, reverse=True)
        
        if len(result) < len(analyses):
            log_debug(f"Deduplication: {len(analyses)} → {len(result)} signals")
        
        return result
    
    def _filter_recent_recommendations_optimized(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Optimized filtering of recent recommendations."""
        if not analyses:
            return analyses
        
        filtered = []
        
        for analysis in analyses:
            # Efficient recommendation checking
            if (self.recommendation_tracker.should_recommend(analysis.symbol) and
                analysis.symbol not in self.processed_symbols_today):
                filtered.append(analysis)
            else:
                log_debug(f"Skipping {analysis.symbol} - recently processed")
        
        return filtered
    
    def _log_recommendations_batch(self, analyses: List[EnhancedNewsAnalysis], 
                                 current_prices: pd.DataFrame) -> int:
        """Batch logging of recommendations for efficiency."""
        if not analyses:
            return 0
        
        try:
            market_conditions = self.market_filter.get_market_conditions()
            recommendations_logged = 0
            
            # Process in batches for better performance
            batch_size = 10
            for i in range(0, len(analyses), batch_size):
                batch = analyses[i:i + batch_size]
                
                for analysis in batch:
                    try:
                        recommendation = self._create_recommendation_optimized(
                            analysis, current_prices, market_conditions
                        )
                        if recommendation:
                            self._log_trade_safely(recommendation)
                            recommendations_logged += 1
                            
                            log_info(f"[REC] {recommendation.side.upper()} {analysis.symbol} "
                                   f"@ ${recommendation.entry_price:.2f} conf={analysis.combined_confidence:.2f}")
                            
                    except Exception as e:
                        log_error(f"Error creating recommendation for {analysis.symbol}: {e}")
            
            return recommendations_logged
            
        except Exception as e:
            log_error(f"Error in batch recommendation logging: {e}")
            return 0
    
    def _create_recommendation_optimized(self, analysis: EnhancedNewsAnalysis, 
                                       current_prices: pd.DataFrame, 
                                       market_conditions) -> Optional[EnhancedTrade]:
        """Create recommendation with optimized data extraction."""
        try:
            # Determine trade direction
            side = self._determine_trade_side_optimized(analysis)
            if not side:
                return None
            
            # Get current price efficiently
            current_price = self._get_current_price_optimized(analysis.symbol, current_prices)
            if current_price <= 0:
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size_optimized(analysis)
            
            # Create entry price with small buffer
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            
            # Extract technical data
            tech_data = self._extract_technical_data_optimized(analysis.technical_analysis)
            
            # Determine rejection reason
            rejection_reason = self._get_rejection_reason(market_conditions)
            
            return EnhancedTrade(
                id=self._generate_unique_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                trade_type="RECOMMENDATION",
                execution_status="PENDING",
                rejection_reason=rejection_reason,
                
                news_title=analysis.title[:150],
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
            
        except Exception as e:
            log_error(f"Error creating recommendation for {analysis.symbol}: {e}")
            return None
    
    def _determine_trade_side_optimized(self, analysis: EnhancedNewsAnalysis) -> Optional[str]:
        """Optimized trade direction determination."""
        sentiment = analysis.sentiment_score
        confidence = analysis.combined_confidence
        
        if confidence < CONFIG.min_confidence_score:
            return None
        
        if sentiment > 0.20:
            return 'long'
        elif sentiment < -0.20:
            return 'short'
        
        return None
    
    def _get_current_price_optimized(self, symbol: str, current_prices: pd.DataFrame) -> float:
        """Optimized current price retrieval."""
        try:
            # Use boolean indexing for efficiency
            price_mask = current_prices['symbol'] == symbol
            price_rows = current_prices[price_mask]
            
            if price_rows.empty:
                return 0.0
            
            current_price = float(price_rows.iloc[0].get('lastSalePrice', 0))
            return current_price if current_price > 0 else 0.0
            
        except (ValueError, IndexError, KeyError):
            return 0.0
    
    def _calculate_position_size_optimized(self, analysis: EnhancedNewsAnalysis) -> float:
        """Optimized position size calculation."""
        try:
            base_size = CONFIG.position_size
            confidence_multiplier = analysis.combined_confidence
            
            # Technical adjustments
            if analysis.technical_analysis:
                tech = analysis.technical_analysis
                volatility_factor = max(0.6, 1.0 - (tech.volatility_score * 0.2))
                liquidity_factor = max(0.7, tech.liquidity_score)
                confidence_multiplier *= volatility_factor * liquidity_factor
            
            # Topic multiplier
            topic_mult = self.TOPIC_MULTIPLIERS.get(analysis.topic, 1.0)
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
    
    def _extract_technical_data_optimized(self, technical_analysis: Optional[TechnicalAnalysis]) -> Dict[str, Any]:
        """Optimized technical data extraction with safe defaults."""
        if technical_analysis:
            return {
                'technical_confidence': technical_analysis.technical_confidence,
                'liquidity_score': technical_analysis.liquidity_score,
                'momentum_score': technical_analysis.momentum_score,
                'volume_score': technical_analysis.volume_score,
                'rsi': technical_analysis.rsi,
                'price_trend': technical_analysis.price_trend,
                'bid_ask_spread': technical_analysis.bid_ask_spread
            }
        
        # Safe defaults
        return {
            'technical_confidence': 0.0,
            'liquidity_score': 0.3,
            'momentum_score': 0.0,
            'volume_score': 0.5,
            'rsi': 50.0,
            'price_trend': "sideways",
            'bid_ask_spread': 0.05
        }
    
    def _get_rejection_reason(self, market_conditions) -> str:
        """Get rejection reason based on market conditions."""
        if not market_conditions.is_market_hours:
            return "MARKET_CLOSED"
        elif market_conditions.market_stress_level > 0.85:
            return "HIGH_MARKET_STRESS"
        elif market_conditions.market_regime == 'volatile':
            return "VOLATILE_REGIME"
        else:
            return ""
    
    def _check_market_conditions_cached(self) -> bool:
        """Check market conditions with caching for performance."""
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
    
    def _create_live_trades_batch(self, analyses: List[EnhancedNewsAnalysis], 
                                current_prices: pd.DataFrame) -> int:
        """Create live trades in optimized batches."""
        trades_created = 0
        
        for analysis in analyses:
            try:
                if self._create_live_trade_optimized(analysis, current_prices):
                    trades_created += 1
            except Exception as e:
                log_error(f"Error creating live trade for {analysis.symbol}: {e}")
        
        return trades_created
    
    def _create_live_trade_optimized(self, analysis: EnhancedNewsAnalysis, 
                                   current_prices: pd.DataFrame) -> bool:
        """Create optimized live trade with comprehensive validation."""
        symbol = analysis.symbol
        
        # Quick validation checks
        if not analysis.technical_analysis:
            log_debug(f"Skipping {symbol} - no technical analysis")
            return False
        
        side = self._determine_trade_side_optimized(analysis)
        if not side:
            return False
        
        position_size = self._calculate_position_size_optimized(analysis)
        
        # Portfolio limit checks
        if not self._check_portfolio_limits_fast(symbol, position_size):
            return False
        
        current_price = self._get_current_price_optimized(symbol, current_prices)
        if current_price <= 0:
            return False
        
        # Create and log trade
        return self._execute_trade_creation(analysis, side, position_size, current_price)
    
    def _check_portfolio_limits_fast(self, symbol: str, position_size: float) -> bool:
        """Fast portfolio limit checking."""
        try:
            within_limits, reason = self.portfolio_filter.check_portfolio_limits(symbol, position_size)
            
            if not within_limits:
                log_debug(f"Portfolio limit: {symbol} - {reason}")
                self.performance_stats['filter_rejections'] += 1
                return False
            
            return True
            
        except Exception as e:
            log_error(f"Error checking portfolio limits for {symbol}: {e}")
            return False
    
    def _execute_trade_creation(self, analysis: EnhancedNewsAnalysis, side: str, 
                              position_size: float, current_price: float) -> bool:
        """Execute trade creation with full logging."""
        try:
            market_conditions = self.market_filter.get_market_conditions()
            entry_price = current_price * (1.001 if side == 'long' else 0.999)
            tech_data = self._extract_technical_data_optimized(analysis.technical_analysis)
            
            trade = EnhancedTrade(
                id=self._generate_unique_trade_id(),
                symbol=analysis.symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                entry_time=datetime.now(timezone.utc),
                
                trade_type="LIVE",
                execution_status="EXECUTED",
                rejection_reason="",
                
                news_title=analysis.title[:150],
                news_confidence=analysis.confidence,
                combined_confidence=analysis.combined_confidence,
                finbert_score=analysis.finbert_score,
                keyword_score=analysis.keyword_score,
                topic=analysis.topic,
                
                **tech_data,
                
                market_regime=market_conditions.market_regime,
                market_stress=market_conditions.market_stress_level,
                time_score=market_conditions.time_of_day_score,
                entry_timing="trade_executed"
            )
            
            # Add to position manager
            self.position_manager.add_position(trade)
            
            # Log trade
            self._log_trade_safely(trade)
            
            log_info(f"[LIVE] {side.upper()} {analysis.symbol} @ ${entry_price:.2f} "
                    f"size=${position_size:.0f} conf={analysis.combined_confidence:.2f}")
            
            return True
            
        except Exception as e:
            log_error(f"Error executing trade creation for {analysis.symbol}: {e}")
            return False
    
    def _update_processing_tracking(self, filtered_analyses: List[EnhancedNewsAnalysis]) -> None:
        """Update processing tracking efficiently."""
        for analysis in filtered_analyses:
            self.recommendation_tracker.mark_recommended(analysis.symbol)
            self.processed_symbols_today.add(analysis.symbol)
        
        # Periodic cleanup
        self.timing_optimizer.cleanup_stale_entries()
    
    def check_exits(self, current_prices: pd.DataFrame) -> None:
        """Check position exits using position manager."""
        try:
            self.position_manager.check_exits(current_prices)
        except Exception as e:
            log_error(f"Error checking exits: {e}")
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get active positions safely."""
        try:
            return self.position_manager.get_active_positions()
        except Exception as e:
            log_error(f"Error getting active positions: {e}")
            return []
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L with error handling."""
        try:
            if not self.trade_log_file.exists():
                return 0.0
            
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Get position manager stats
            portfolio_summary = self.position_manager.get_portfolio_summary()
            
            # Combine with trader stats
            return {
                **self.performance_stats,
                **portfolio_summary,
                'daily_pnl': self.get_daily_pnl(),
                'processed_symbols_today': len(self.processed_symbols_today),
                'recommendation_tracker_stats': self.recommendation_tracker.get_stats(),
                'timing_optimizer_stats': self.timing_optimizer.get_timing_statistics()
            }
            
        except Exception as e:
            log_error(f"Error getting performance summary: {e}")
            return self.performance_stats.copy()