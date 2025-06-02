"""
Optimized trader with enhanced deduplication and fixed JSON serialization
"""
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import csv
import json
import time # Import the time module
from pathlib import Path
from config import CONFIG
from analysis.enhanced_news_analyzer import EnhancedNewsAnalysis
from analysis.technical_analyzer import TechnicalAnalysis
from analysis.market_filters import (
    MarketFilter, NewsQualityFilter, PriceActionFilter, 
    PortfolioRiskFilter, EntryTimingOptimizer
)
from utils.simple_logger import log_info, log_error, log_warning, log_debug


@dataclass
class EnhancedTrade:
    """Enhanced trade record with comprehensive data"""
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
    trade_type: str = "LIVE"
    execution_status: str = "PENDING"
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


@dataclass
class ProcessedArticle:
    """Optimized processed article record with JSON-serializable fields"""
    article_hash: str
    symbol: str
    title_hash: str
    processed_time: str  # Store as ISO string instead of datetime
    sentiment_score: float
    combined_confidence: float
    was_traded: int = 0  # Use int instead of bool (0 or 1)
    trade_side: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedArticle':
        """Create from dictionary with safe type conversion."""
        return cls(
            article_hash=str(data.get('article_hash', '')),
            symbol=str(data.get('symbol', '')),
            title_hash=str(data.get('title_hash', '')),
            processed_time=str(data.get('processed_time', datetime.now(timezone.utc).isoformat())),
            sentiment_score=float(data.get('sentiment_score', 0.0)),
            combined_confidence=float(data.get('combined_confidence', 0.0)),
            was_traded=int(data.get('was_traded', 0)),
            trade_side=str(data.get('trade_side', ''))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @property
    def processed_datetime(self) -> datetime:
        """Get processed time as datetime object."""
        try:
            return datetime.fromisoformat(self.processed_time.replace('Z', '+00:00'))
        except:
            return datetime.now(timezone.utc)


class OptimizedRecommendationTracker:
    """Optimized recommendation tracker with memory-efficient storage"""
    
    def __init__(self):
        self.recent_recommendations: Dict[str, float] = {}  # symbol -> timestamp
        self.recommendation_window_hours = 3  # Reduced from 4 hours
        self.max_tracking = 500  # Reduced memory footprint
        self._last_cleanup = time.time()
        self._cleanup_interval = 1800  # 30 minutes
    
    def should_recommend(self, symbol: str) -> bool:
        """Check if we should recommend this symbol with automatic cleanup."""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_recommendations()
        
        if symbol not in self.recent_recommendations:
            return True
        
        last_recommendation = self.recent_recommendations[symbol]
        time_since_last = current_time - last_recommendation
        
        should_recommend = time_since_last >= (self.recommendation_window_hours * 3600)
        
        if not should_recommend:
            log_debug(f"Skipping {symbol} - recommended {time_since_last/3600:.1f}h ago")
        
        return should_recommend
    
    def mark_recommended(self, symbol: str) -> None:
        """Mark symbol as recommended with memory management."""
        current_time = time.time()
        self.recent_recommendations[symbol] = current_time
        
        # Immediate cleanup if over limit
        if len(self.recent_recommendations) > self.max_tracking:
            self._cleanup_old_recommendations()
    
    def _cleanup_old_recommendations(self) -> None:
        """Optimized cleanup of old recommendations."""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours
        
        # Remove old entries in one pass
        old_symbols = [
            symbol for symbol, timestamp in self.recent_recommendations.items()
            if timestamp < cutoff_time
        ]
        
        for symbol in old_symbols:
            del self.recent_recommendations[symbol]
        
        # If still over limit, remove oldest entries
        if len(self.recent_recommendations) > self.max_tracking:
            sorted_items = sorted(
                self.recent_recommendations.items(), 
                key=lambda x: x[1]
            )
            keep_count = self.max_tracking // 2
            self.recent_recommendations = dict(sorted_items[-keep_count:])
        
        self._last_cleanup = current_time
        
        if old_symbols:
            log_debug(f"Cleaned up {len(old_symbols)} old recommendations")


class OptimizedEnhancedTrader:
    """Optimized trader with enhanced deduplication and fixed serialization"""
    
    def __init__(self, fmp_loader) -> None:
        """Initialize with optimized components and error handling"""
        self.fmp_loader = fmp_loader
        self.active_trades: Dict[str, EnhancedTrade] = {}
        self.trade_log_file = CONFIG.trade_log_path / CONFIG.trade_log_file
        self.trade_counter = 0
        
        # Optimized tracking systems
        self.recommendation_tracker = OptimizedRecommendationTracker()
        self.processed_symbols_today: Set[str] = set()
        self.daily_symbol_reset_time = datetime.now(timezone.utc).date()
        
        # Optimized article combination settings
        self.article_combination_window = timedelta(hours=1.5)  # Reduced
        self.breaking_news_window = timedelta(minutes=10)       # Reduced
        
        self._initialize_components()
        self._setup_trade_log()
    
    def _initialize_components(self) -> None:
        """Initialize filter components with error handling"""
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
        """Setup trade log CSV with optimized headers"""
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
        """Log trade data to CSV with optimized serialization"""
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
        """Generate unique trade ID efficiently"""
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
        """Apply optimized news quality filters"""
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
            
            # Check for spam patterns
            title_lower = title.lower()
            spam_patterns = ['click here', 'ad:', 'advertisement', 'sponsored']
            for pattern in spam_patterns:
                if pattern in title_lower:
                    rejection_reasons.append(f"Contains spam: '{pattern}'")
            
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
        """Process news signals with optimized deduplication"""
        if not analyses or current_prices is None or current_prices.empty:
            return
        
        try:
            # Reset daily tracking if needed
            self._reset_daily_tracking_if_needed()
            
            # Optimized deduplication by symbol
            deduplicated_analyses = self._optimized_deduplicate_signals(analyses)
            
            if len(deduplicated_analyses) < len(analyses):
                log_info(f"Deduplication: {len(analyses)} -> {len(deduplicated_analyses)} signals")
            
            # Filter out recent recommendations
            filtered_analyses = self._filter_recent_recommendations(deduplicated_analyses)
            
            if len(filtered_analyses) < len(deduplicated_analyses):
                log_info(f"Recent filter: {len(deduplicated_analyses)} -> {len(filtered_analyses)} signals")
            
            # Always log recommendations
            recommendations_logged = self._log_all_recommendations(filtered_analyses, current_prices)
            
            # Check market conditions for live trading
            market_favorable = self._check_market_conditions()
            
            if not market_favorable:
                log_info(f"[RECOMMENDATIONS] Logged {recommendations_logged} (market unfavorable)")
                return
            
            # Process for live trading
            trades_created = self._process_filtered_signals(filtered_analyses, current_prices, is_live=True)
            
            # Mark symbols as recommended
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
    
    def _optimized_deduplicate_signals(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Optimized deduplication with improved article combination"""
        if not analyses:
            return analyses
        
        try:
            # Group by symbol efficiently
            symbol_groups: Dict[str, List[EnhancedNewsAnalysis]] = {}
            current_time = datetime.now(timezone.utc)
            
            for analysis in analyses:
                symbol = analysis.symbol
                
                # Check time window
                try:
                    article_time = analysis.timestamp.to_pydatetime() if hasattr(analysis.timestamp, 'to_pydatetime') else current_time
                    time_diff = current_time - article_time
                    
                    if time_diff <= self.article_combination_window:
                        if symbol not in symbol_groups:
                            symbol_groups[symbol] = []
                        symbol_groups[symbol].append(analysis)
                    else:
                        log_debug(f"Excluding stale article for {symbol}: {time_diff.total_seconds()/3600:.1f}h old")
                except Exception:
                    # Include if timestamp processing fails
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(analysis)
            
            # Create combined analyses efficiently
            combined_analyses = []
            for symbol, articles in symbol_groups.items():
                try:
                    if len(articles) == 1:
                        combined_analyses.append(articles[0])
                    else:
                        # Combine multiple articles
                        combined_analysis = self._create_optimized_combined_analysis(symbol, articles)
                        if combined_analysis:
                            combined_analyses.append(combined_analysis)
                            log_debug(f"Combined {len(articles)} articles for {symbol}")
                        
                except Exception as e:
                    log_error(f"Error combining articles for {symbol}: {e}")
                    # Fallback to best analysis
                    best_analysis = max(articles, key=lambda x: x.combined_confidence)
                    combined_analyses.append(best_analysis)
            
            # Sort by confidence
            combined_analyses.sort(key=lambda x: x.combined_confidence, reverse=True)
            
            return combined_analyses
            
        except Exception as e:
            log_error(f"Error in optimized deduplication: {e}")
            # Fallback to simple deduplication
            return self._simple_deduplicate_fallback(analyses)
    
    def _create_optimized_combined_analysis(self, symbol: str, articles: List[EnhancedNewsAnalysis]) -> EnhancedNewsAnalysis:
        """Create optimized combined analysis with better scoring"""
        # Use highest confidence article as base
        base_analysis = max(articles, key=lambda x: x.combined_confidence)
        
        # Calculate weighted sentiment efficiently
        weighted_sum = sum(a.sentiment_score * a.combined_confidence for a in articles)
        total_weight = sum(a.combined_confidence for a in articles)
        
        enhanced_sentiment = weighted_sum / total_weight if total_weight > 0 else base_analysis.sentiment_score
        
        # Calculate optimized combination bonus
        article_count = len(articles)
        count_bonus = min(1.0 + (article_count - 1) * 0.08, 1.25)  # Max 25% bonus
        
        # Check for theme consistency
        positive_count = sum(1 for a in articles if a.sentiment_score > 0.2)
        negative_count = sum(1 for a in articles if a.sentiment_score < -0.2)
        
        consistency_ratio = max(positive_count, negative_count) / article_count
        theme_bonus = 1.15 if consistency_ratio >= 0.7 else 1.0
        
        combined_confidence = min(
            base_analysis.combined_confidence * count_bonus * theme_bonus, 
            1.0
        )
        
        combined_title = f"[{article_count} ARTICLES] {base_analysis.title}"
        
        # Create combined analysis
        combined_analysis = EnhancedNewsAnalysis(
            symbol=symbol,
            title=combined_title,
            content=base_analysis.content,
            sentiment_score=enhanced_sentiment,
            confidence=base_analysis.confidence,
            topic=base_analysis.topic,
            timestamp=base_analysis.timestamp,
            finbert_score=base_analysis.finbert_score,
            keyword_score=base_analysis.keyword_score,
            gemini_score=base_analysis.gemini_score,
            gemini_confidence=base_analysis.gemini_confidence,
            gemini_reasoning=base_analysis.gemini_reasoning,
            technical_analysis=base_analysis.technical_analysis,
            strategy_signals=base_analysis.strategy_signals,
            best_strategy=base_analysis.best_strategy,
            combined_confidence=combined_confidence
        )
        
        log_info(f"COMBINED: {symbol} ({article_count} articles) "
                f"sentiment={enhanced_sentiment:.3f} conf={combined_confidence:.3f}")
        
        return combined_analysis
    
    def _simple_deduplicate_fallback(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Simple fallback deduplication for error cases"""
        symbol_best = {}
        
        for analysis in analyses:
            symbol = analysis.symbol
            if symbol not in symbol_best or analysis.combined_confidence > symbol_best[symbol].combined_confidence:
                symbol_best[symbol] = analysis
        
        return sorted(symbol_best.values(), key=lambda x: x.combined_confidence, reverse=True)
    
    def _filter_recent_recommendations(self, analyses: List[EnhancedNewsAnalysis]) -> List[EnhancedNewsAnalysis]:
        """Filter out symbols with recent recommendations efficiently"""
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
        """Log all recommendations with optimized processing"""
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
        """Create recommendation trade record with optimized data extraction"""
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
            
            # Determine rejection reason efficiently
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
        """Process filtered signals for trading with optimized flow"""
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
        """Process single trading signal with optimized checks"""
        symbol = analysis.symbol
        
        # Quick validation checks
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
        """Determine trade direction with optimized logic"""
        sentiment = analysis.sentiment_score
        confidence = analysis.combined_confidence
        
        if confidence < CONFIG.min_confidence_score:
            return None
        
        # More aggressive thresholds for better signal capture
        if sentiment > 0.20:
            return 'long'
        elif sentiment < -0.20:
            return 'short'
        
        return None
    
    def _determine_position_size(self, analysis: EnhancedNewsAnalysis) -> float:
        """Calculate position size with optimized risk management"""
        try:
            base_size = CONFIG.position_size
            confidence_multiplier = analysis.combined_confidence
            
            if analysis.technical_analysis:
                tech = analysis.technical_analysis
                volatility_factor = max(0.5, 1.0 - (tech.volatility_score * 0.25))
                liquidity_factor = max(0.6, tech.liquidity_score)
                confidence_multiplier *= volatility_factor * liquidity_factor
            
            # Optimized topic multipliers
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
        """Check portfolio risk limits with optimized logic"""
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
        """Get current price for symbol with optimized lookup"""
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
        """Check entry timing with optimized logic"""
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
        """Validate technical and sentiment alignment with optimized checks"""
        try:
            sentiment = analysis.sentiment_score
            tech = analysis.technical_analysis
            
            if not tech:
                return False
            
            momentum = tech.momentum_score
            
            # Optimized alignment check - more permissive
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
        """Create and log trade with optimized data handling"""
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
                self.active_trades[analysis.symbol] = trade
            
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
        """Check position exits with optimized logic"""
        if current_prices is None or current_prices.empty or not self.active_trades:
            return
        
        try:
            market_conditions = self.market_filter.get_market_conditions()
            stress_multiplier = self._calculate_stress_multiplier(market_conditions)
            
            trades_to_close = []
            
            for symbol, trade in self.active_trades.items():
                try:
                    if self._check_trade_exit(trade, current_prices, stress_multiplier):
                        trades_to_close.append(symbol)
                except Exception as e:
                    log_error(f"Error checking exit for {symbol}: {e}")
            
            for symbol in trades_to_close:
                self.active_trades.pop(symbol, None)
                
        except Exception as e:
            log_error(f"Error in exit checking: {e}")
    
    def _calculate_stress_multiplier(self, market_conditions) -> float:
        """Calculate stress-based exit multiplier with optimized logic"""
        stress_level = market_conditions.market_stress_level
        
        if stress_level > 0.8:
            return 0.65
        elif stress_level > 0.6:
            return 0.8
        else:
            return 1.0
    
    def _check_trade_exit(self, trade: EnhancedTrade, current_prices: pd.DataFrame, 
                         stress_multiplier: float) -> bool:
        """Check if trade should be exited with optimized calculations"""
        current_price = self._get_current_price(trade.symbol, current_prices)
        if current_price <= 0:
            return False
        
        pnl_pct = self._calculate_pnl_percentage(trade, current_price)
        stop_loss_pct, take_profit_pct = self._calculate_dynamic_levels(trade, stress_multiplier)
        exit_reason = self._determine_exit_reason(pnl_pct, stop_loss_pct, take_profit_pct, stress_multiplier)
        
        if exit_reason:
            self._execute_exit(trade, current_price, exit_reason, pnl_pct)
            return True
        
        return False
    
    def _calculate_pnl_percentage(self, trade: EnhancedTrade, current_price: float) -> float:
        """Calculate P&L percentage efficiently"""
        if trade.side == 'long':
            return (current_price - trade.entry_price) / trade.entry_price
        else:
            return (trade.entry_price - current_price) / trade.entry_price
    
    def _calculate_dynamic_levels(self, trade: EnhancedTrade, stress_multiplier: float) -> Tuple[float, float]:
        """Calculate dynamic stop/target levels with optimized logic"""
        stop_loss_pct = CONFIG.stop_loss_pct * stress_multiplier
        take_profit_pct = CONFIG.take_profit_pct
        
        # Adjust for liquidity
        if trade.liquidity_score < 0.5:
            stop_loss_pct *= 0.85
            take_profit_pct *= 0.85
        
        # Adjust for confidence
        if trade.combined_confidence > 0.8:
            take_profit_pct *= 1.15
        
        return stop_loss_pct, take_profit_pct
    
    def _determine_exit_reason(self, pnl_pct: float, stop_loss_pct: float, 
                             take_profit_pct: float, stress_multiplier: float) -> Optional[str]:
        """Determine exit reason with optimized logic"""
        if pnl_pct <= -stop_loss_pct:
            return f'stop_loss_{stress_multiplier:.1f}x'
        elif pnl_pct >= take_profit_pct:
            return 'take_profit'
        elif stress_multiplier < 0.8 and pnl_pct > 0.015:
            return 'market_stress_protect'
        
        return None
    
    def _execute_exit(self, trade: EnhancedTrade, current_price: float, 
                     exit_reason: str, pnl_pct: float) -> None:
        """Execute trade exit with optimized logging"""
        trade.exit_price = current_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_reason = exit_reason
        trade.pnl = trade.position_size * pnl_pct
        
        self._log_trade(trade)
        
        duration_minutes = (trade.exit_time - trade.entry_time).total_seconds() / 60
        log_info(f"[EXIT] {trade.side} {trade.symbol} @ ${current_price:.2f} "
                f"P&L=${trade.pnl:.2f} ({exit_reason}) | duration={duration_minutes:.1f}min")
    
    def get_active_positions(self) -> List[EnhancedTrade]:
        """Get active trades efficiently"""
        return list(self.active_trades.values())
    
    def get_daily_pnl(self) -> float:
        """Calculate daily P&L with optimized data access"""
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
        """Analyze filter effectiveness with optimized data processing"""
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
EnhancedTrader = OptimizedEnhancedTrader