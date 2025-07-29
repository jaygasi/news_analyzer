"""
Profit-Maximizing Financial News Analyzer - Main Application
Implements profit-first learning with multi-horizon predictions and dynamic position sizing
Python 3.13.3 compatible
"""
import asyncio
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
import pytz
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from config import Config
from database.article_tracker import ArticleTracker
from data_loaders.news_fetcher import NewsFetcher
from core.ticker_filter import TickerFilterEngine, FilterCriteria
from core.profit_maximizing_engine import ProfitMaximizingEngine
from core.market_regime_detector import MarketRegimeDetector
from core.position_sizing_engine import PositionSizingEngine
from tools.profit_learning_system import ProfitLearningSystem
from tools.uncertainty_estimator import UncertaintyEstimator
from analysis.multi_horizon_predictor import MultiHorizonPredictor
from analysis.risk_assessment_engine import RiskAssessmentEngine
from analysis.exit_timing_optimizer import ExitTimingOptimizer
from data_loaders.earnings_integration_manager import EarningsIntegrationManager
from analysis.multi_llm_analyzer import MultiLLMAnalyzer
from analysis.technical_analyzer_simple import TechnicalAnalyzer
from analysis.enhanced_price_tracker import EnhancedPriceTracker
from output.profit_csv_logger import ProfitCSVLogger
from core.ticker_aggregator import TickerAggregator
from utils.simple_logger import log_info, log_error, log_debug, log_warning


@dataclass
class ProfitPrediction:
    """Multi-horizon profit prediction with uncertainty quantification"""
    ticker: str
    
    # Profit predictions at multiple time horizons (in percentage points)
    profit_15min: float = 0.0
    profit_1h: float = 0.0  
    profit_4h: float = 0.0
    profit_eod: float = 0.0
    
    # Risk assessment
    downside_risk: float = 0.0      # Maximum expected loss
    upside_potential: float = 0.0   # Maximum expected gain
    volatility_forecast: float = 0.0 # Expected price volatility
    
    # Timing optimization
    optimal_entry_delay: int = 0     # Minutes to wait before entering
    optimal_exit_time: int = 0       # Minutes to optimal exit
    stop_loss_level: float = 0.0     # Exit if loss exceeds this %
    take_profit_level: float = 0.0   # Exit if profit exceeds this %
    
    # Confidence and uncertainty
    prediction_confidence: float = 0.0        # How sure we are (0-1)
    model_uncertainty: float = 0.0            # Model's uncertainty estimate
    similar_historical_cases: int = 0         # Number of similar past cases
    epistemic_uncertainty: float = 0.0       # Knowledge uncertainty
    aleatoric_uncertainty: float = 0.0       # Data uncertainty
    
    # Market context
    market_regime: str = "unknown"            # Current market regime
    regime_confidence: float = 0.0            # Confidence in regime detection
    
    # Supporting analysis
    news_impact_score: float = 0.0
    technical_momentum: float = 0.0
    earnings_catalyst: bool = False
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "v1.0"


@dataclass  
class TradingDecision:
    """Enhanced trading decision with profit optimization"""
    ticker: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'WAIT'
    
    # Position sizing and risk management
    position_size: float = 0.0              # Percentage of portfolio
    kelly_fraction: float = 0.0             # Kelly criterion result
    risk_adjusted_size: float = 0.0         # Final position size after risk adjustment
    
    # Entry and exit strategy
    recommended_entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_time: int = 0                  # Maximum minutes to hold
    
    # Profit expectations
    expected_profit: float = 0.0            # Expected profit percentage
    expected_return_1h: float = 0.0
    expected_return_4h: float = 0.0
    expected_return_eod: float = 0.0
    
    # Risk metrics
    value_at_risk: float = 0.0              # VaR at 95% confidence
    maximum_drawdown: float = 0.0           # Maximum expected drawdown
    sharpe_ratio_forecast: float = 0.0      # Expected Sharpe ratio
    
    # Decision rationale
    primary_catalyst: str = ""
    confidence_level: float = 0.0
    reasoning: str = ""
    
    # Supporting predictions
    profit_prediction: Optional[ProfitPrediction] = None
    
    # Metadata
    decision_timestamp: datetime = field(default_factory=datetime.now)
    article_count: int = 0


class ProfitMaximizingTradingSystem:
    """Main application implementing profit-first learning and optimization"""

    def __init__(self) -> None:
        """Initialize the profit-maximizing trading system"""
        self.cycle_count = 0
        self.last_successful_run = None
        self.total_realized_profit = 0.0
        self.total_opportunity_cost = 0.0
        
        # Performance tracking
        self.successful_trades = 0
        self.failed_trades = 0
        self.skipped_opportunities = 0
        
        log_info("🚀 Initializing Profit-Maximizing Trading System...")
        self._initialize_components()
        self._log_startup_summary()

    def _initialize_components(self) -> None:
        """Initialize all system components with profit-first focus"""
        
        # Core data and filtering components
        log_info("📊 Initializing data components...")
        self.article_tracker = ArticleTracker()
        self.news_fetcher = NewsFetcher()
        self.ticker_aggregator = TickerAggregator()
        
        # Fundamental filtering
        if Config.ENABLE_FUNDAMENTAL_FILTERING:
            self.ticker_filter = TickerFilterEngine(self.news_fetcher)
            log_info("🔍 Fundamental filtering enabled")
        else:
            self.ticker_filter = None
            log_info("🔍 Fundamental filtering disabled")

        # Earnings analysis
        if Config.ENABLE_EARNINGS_EVENTS:
            self.earnings_manager = EarningsIntegrationManager(self.news_fetcher)
            log_info("🎙️ Earnings analysis enabled")
        else:
            self.earnings_manager = None
            log_info("🎙️ Earnings analysis disabled")

        # AI Analysis Components - Profit-Focused
        log_info("🧠 Initializing AI prediction components...")
        self.multi_horizon_predictor = MultiHorizonPredictor()
        self.market_regime_detector = MarketRegimeDetector()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.risk_assessment = RiskAssessmentEngine()
        self.exit_timing_optimizer = ExitTimingOptimizer()
        
        # Traditional analysis (for ensemble)
        self.llm_analyzer = MultiLLMAnalyzer()
        if Config.ENABLE_TECHNICAL_ANALYSIS:
            self.technical_analyzer = TechnicalAnalyzer(self.news_fetcher)
        else:
            self.technical_analyzer = None

        # Decision and Learning Systems
        log_info("🎯 Initializing decision and learning systems...")
        self.profit_engine = ProfitMaximizingEngine(
            self.multi_horizon_predictor,
            self.market_regime_detector,
            self.uncertainty_estimator
        )
        self.position_sizing = PositionSizingEngine()
        self.learning_system = ProfitLearningSystem()
        
        # Tracking and Logging
        log_info("📈 Initializing tracking components...")
        self.price_tracker = EnhancedPriceTracker(self.news_fetcher)
        self.csv_logger = ProfitCSVLogger()
        
        log_info("✅ All components initialized successfully")

    def _log_startup_summary(self) -> None:
        """Log enhanced startup summary with profit-focused metrics"""
        log_info("=" * 90)
        log_info("🎯 PROFIT-MAXIMIZING TRADING SYSTEM INITIALIZED")
        log_info("=" * 90)
        
        # System capabilities
        log_info("🚀 Core Capabilities:")
        log_info("   📊 Multi-horizon profit prediction (15min, 1h, 4h, EOD)")
        log_info("   🎯 Dynamic position sizing with Kelly criterion")
        log_info("   🧠 Market regime detection and adaptation")
        log_info("   📈 Uncertainty-aware predictions with confidence intervals")
        log_info("   ⏰ Optimal entry/exit timing optimization")
        log_info("   💰 Continuous profit-based learning")
        
        # Model status
        model_info = self.multi_horizon_predictor.get_model_status()
        if model_info.get('available', False):
            log_info(f"   🤖 Multi-Horizon Predictor: {model_info.get('accuracy', 'N/A')} accuracy")
            log_info(f"      Architecture: {model_info.get('architecture', 'Ensemble')}")
        else:
            log_info("   🤖 Multi-Horizon Predictor: Initializing...")
        
        # Learning system status
        learning_status = self.learning_system.get_system_status()
        if learning_status.get('available', False):
            log_info(f"   🎓 Learning System: {learning_status.get('total_trades_analyzed', 0)} trades analyzed")
            log_info(f"      Models trained: {', '.join(learning_status.get('trained_models', []))}")
        else:
            log_info("   🎓 Learning System: Ready for first training")
            
        # Risk management
        log_info("⚠️ Risk Management:")
        log_info(f"   💰 Max position size: {Config.MAX_POSITION_SIZE:.1%}")
        log_info(f"   📉 Max portfolio risk: {Config.MAX_PORTFOLIO_RISK:.1%}")
        log_info(f"   🛡️ Stop loss tolerance: {Config.DEFAULT_STOP_LOSS:.1%}")
        log_info(f"   🎯 Take profit target: {Config.DEFAULT_TAKE_PROFIT:.1%}")
        
        # Trading thresholds
        log_info("🎯 Trading Thresholds:")
        log_info(f"   📊 Min profit expectation: {Config.MIN_EXPECTED_PROFIT:.2%}")
        log_info(f"   🎲 Min prediction confidence: {Config.MIN_PREDICTION_CONFIDENCE:.1%}")
        log_info(f"   📈 Min Sharpe ratio: {Config.MIN_SHARPE_RATIO:.2f}")
        
        log_info("=" * 90)

    async def run(self) -> None:
        """Main execution loop with profit optimization"""
        log_info("🔄 Starting profit-maximizing analysis loop...")
        
        while True:
            cycle_start = time.time()
            self.cycle_count += 1
            
            try:
                log_info(f"🔄 Cycle {self.cycle_count}: Profit Analysis Starting...")
                
                # Execute profit analysis cycle
                await self._execute_profit_cycle()
                
                # Calculate cycle performance
                cycle_duration = time.time() - cycle_start
                log_info(f"✅ Cycle {self.cycle_count} completed in {cycle_duration:.1f}s")
                
                # Update system performance metrics
                self._update_performance_metrics()
                
                # Adaptive learning update
                if self.cycle_count % Config.LEARNING_UPDATE_FREQUENCY == 0:
                    await self._trigger_learning_update()
                
                self.last_successful_run = datetime.now()
                
                # Wait for next cycle
                log_info(f"⏱️ Waiting {Config.ANALYSIS_INTERVAL_MINUTES} minutes until next cycle...")
                await asyncio.sleep(Config.ANALYSIS_INTERVAL_MINUTES * 60)
                
            except Exception as e:
                log_error(f"💥 Cycle {self.cycle_count} failed: {e}")
                traceback.print_exc()
                
                # Exponential backoff on errors
                wait_time = min(300, 60 * (2 ** min(3, self.cycle_count % 4)))
                log_info(f"⏱️ Error recovery: waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

    async def _execute_profit_cycle(self) -> None:
        """Execute one complete profit analysis cycle"""
        
        # Step 1: Fetch and filter news
        log_debug("📰 Fetching latest financial news...")
        articles = await self._fetch_filtered_articles()
        
        if not articles:
            log_info("📰 No new articles found, skipping cycle")
            return
            
        log_info(f"📰 Processing {len(articles)} new articles")

        # Step 2: Group articles by ticker  
        log_debug("🎯 Grouping articles by ticker...")
        ticker_groups = self.ticker_aggregator.group_by_ticker(articles)
        log_info(f"🎯 Found {len(ticker_groups)} tickers to analyze")

        # Step 3: Apply fundamental filtering
        if self.ticker_filter:
            log_debug("🔍 Applying fundamental filters...")
            filtered_tickers = await self._apply_fundamental_filtering(ticker_groups)
        else:
            filtered_tickers = ticker_groups

        if not filtered_tickers:
            log_info("🔍 No tickers passed fundamental filtering")
            return

        log_info(f"🔍 {len(filtered_tickers)} tickers passed fundamental filtering")

        # Step 4: Market regime detection
        log_debug("🌍 Detecting current market regime...")
        market_regime = await self._detect_market_regime()
        log_info(f"🌍 Current market regime: {market_regime['regime']} (confidence: {market_regime['confidence']:.1%})")

        # Step 5: Multi-horizon profit predictions
        log_debug("🔮 Generating profit predictions...")
        trading_decisions = []
        
        for ticker, articles in filtered_tickers.items():
            try:
                decision = await self._analyze_profit_opportunity(ticker, articles, market_regime)
                if decision and decision.action != 'HOLD':
                    trading_decisions.append(decision)
                    
            except Exception as e:
                log_error(f"Failed to analyze {ticker}: {e}")
                continue

        # Step 6: Position sizing and portfolio optimization
        if trading_decisions:
            log_debug("💰 Optimizing position sizes...")
            optimized_decisions = await self._optimize_portfolio_positions(trading_decisions)
            
            # Step 7: Log decisions and capture entry prices
            log_debug("📊 Logging trading decisions...")
            await self._log_trading_decisions(optimized_decisions)
            
            log_info(f"🎯 Generated {len(optimized_decisions)} optimized trading decisions")
        else:
            log_info("🎯 No profitable opportunities identified this cycle")

        # Step 8: Update tracking for existing positions
        log_debug("📈 Updating position tracking...")
        await self._update_position_tracking()

    async def _fetch_filtered_articles(self) -> List[Dict[str, Any]]:
        """Fetch and filter articles for processing"""
        try:
            # Fetch from all sources
            all_articles = []
            
            # Stock news
            stock_news = await self.news_fetcher.get_stock_news()
            all_articles.extend(stock_news)
            
            # Press releases
            press_releases = await self.news_fetcher.get_press_releases()
            all_articles.extend(press_releases)
            
            # Earnings events
            if self.earnings_manager:
                earnings_articles = await self.earnings_manager.get_earnings_events()
                all_articles.extend(earnings_articles)
            
            # Filter out already processed articles
            new_articles = []
            for article in all_articles:
                if not self.article_tracker.is_processed(article):
                    new_articles.append(article)
                    
            return new_articles
            
        except Exception as e:
            log_error(f"Error fetching articles: {e}")
            return []

    async def _apply_fundamental_filtering(self, ticker_groups: Dict[str, List]) -> Dict[str, List]:
        """Apply fundamental filtering to ticker groups"""
        if not self.ticker_filter:
            return ticker_groups
            
        try:
            tickers = list(ticker_groups.keys())
            filter_criteria = FilterCriteria()  # Use default criteria
            
            passed_tickers = await self.ticker_filter.filter_tickers(tickers, filter_criteria)
            
            # Return only the tickers that passed filtering
            return {ticker: articles for ticker, articles in ticker_groups.items() 
                   if ticker in passed_tickers}
                   
        except Exception as e:
            log_error(f"Fundamental filtering failed: {e}")
            return ticker_groups

    async def _detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime for adaptive strategy"""
        try:
            regime_data = await self.market_regime_detector.detect_current_regime()
            return regime_data
        except Exception as e:
            log_error(f"Market regime detection failed: {e}")
            return {
                'regime': 'unknown',
                'confidence': 0.5,
                'volatility': 'medium',
                'trend': 'sideways'
            }

    async def _analyze_profit_opportunity(self, ticker: str, articles: List[Dict], 
                                        market_regime: Dict[str, Any]) -> Optional[TradingDecision]:
        """Analyze profit opportunity for a single ticker"""
        try:
            # Step 1: Multi-horizon profit prediction
            profit_prediction = await self.multi_horizon_predictor.predict_profit(
                ticker=ticker,
                articles=articles,
                market_regime=market_regime
            )
            
            # Step 2: Uncertainty quantification  
            uncertainty_analysis = await self.uncertainty_estimator.analyze_uncertainty(
                ticker=ticker,
                prediction=profit_prediction,
                market_regime=market_regime
            )
            
            # Step 3: Risk assessment
            risk_metrics = await self.risk_assessment.assess_risk(
                ticker=ticker,
                profit_prediction=profit_prediction,
                uncertainty=uncertainty_analysis
            )
            
            # Step 4: Exit timing optimization
            exit_strategy = await self.exit_timing_optimizer.optimize_exit_timing(
                ticker=ticker,
                profit_prediction=profit_prediction,
                risk_metrics=risk_metrics
            )
            
            # Step 5: Make trading decision
            decision = await self.profit_engine.make_profit_decision(
                ticker=ticker,
                profit_prediction=profit_prediction,
                uncertainty=uncertainty_analysis,
                risk_metrics=risk_metrics,
                exit_strategy=exit_strategy,
                market_regime=market_regime,
                articles=articles
            )
            
            return decision
            
        except Exception as e:
            log_error(f"Error analyzing profit opportunity for {ticker}: {e}")
            return None

    async def _optimize_portfolio_positions(self, decisions: List[TradingDecision]) -> List[TradingDecision]:
        """Optimize position sizes for portfolio-level performance"""
        try:
            return await self.position_sizing.optimize_portfolio_positions(decisions)
        except Exception as e:
            log_error(f"Portfolio optimization failed: {e}")
            return decisions

    async def _log_trading_decisions(self, decisions: List[TradingDecision]) -> None:
        """Log trading decisions and capture entry prices"""
        try:
            # Capture current prices for all tickers
            tickers = [d.ticker for d in decisions]
            current_prices = await self.price_tracker.get_current_prices(tickers)
            
            # Update decisions with actual entry prices
            for decision in decisions:
                if decision.ticker in current_prices:
                    decision.recommended_entry_price = current_prices[decision.ticker]
            
            # Log to CSV
            await self.csv_logger.log_decisions(decisions)
            
            # Mark articles as processed
            for decision in decisions:
                # This would need to be implemented to track which articles led to decisions
                pass
                
        except Exception as e:
            log_error(f"Error logging trading decisions: {e}")

    async def _update_position_tracking(self) -> None:
        """Update tracking for existing positions"""
        try:
            await self.price_tracker.update_all_positions()
        except Exception as e:
            log_error(f"Error updating position tracking: {e}")

    async def _trigger_learning_update(self) -> None:
        """Trigger adaptive learning system update"""
        try:
            log_info("🎓 Triggering adaptive learning update...")
            success = await self.learning_system.run_profit_learning_update()
            
            if success:
                log_info("✅ Learning system updated successfully")
                # Reload models with updated weights
                await self._reload_updated_models()
            else:
                log_warning("⚠️ Learning update failed or insufficient data")
                
        except Exception as e:
            log_error(f"Learning update failed: {e}")

    async def _reload_updated_models(self) -> None:
        """Reload models after learning update"""
        try:
            await self.multi_horizon_predictor.reload_models()
            await self.market_regime_detector.reload_models()
            log_info("🔄 Models reloaded with updated weights")
        except Exception as e:
            log_error(f"Model reload failed: {e}")

    def _update_performance_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            # This would integrate with the learning system to track realized performance
            performance = self.learning_system.get_performance_summary()
            
            if performance:
                self.total_realized_profit = performance.get('total_profit', 0.0)
                self.successful_trades = performance.get('successful_trades', 0)
                self.failed_trades = performance.get('failed_trades', 0)
                
                if self.cycle_count % 10 == 0:  # Log every 10 cycles
                    log_info(f"📊 Performance Summary:")
                    log_info(f"   💰 Total realized profit: {self.total_realized_profit:+.2%}")
                    log_info(f"   ✅ Successful trades: {self.successful_trades}")
                    log_info(f"   ❌ Failed trades: {self.failed_trades}")
                    if self.successful_trades + self.failed_trades > 0:
                        win_rate = self.successful_trades / (self.successful_trades + self.failed_trades)
                        log_info(f"   🎯 Win rate: {win_rate:.1%}")
                        
        except Exception as e:
            log_debug(f"Performance metrics update failed: {e}")

    def cleanup(self) -> None:
        """Clean shutdown of all components"""
        log_info("🧹 Starting system cleanup...")
        
        try:
            # Save final learning state
            if hasattr(self, 'learning_system'):
                self.learning_system.save_state()
                
            # Stop background tasks
            # This would stop any running schedulers or background tasks
            
            log_info("✅ System cleanup completed")
            
        except Exception as e:
            log_error(f"Cleanup error: {e}")


def _log_startup_banner() -> None:
    """Log the startup banner"""
    log_info("=" * 90)
    log_info("🚀 PROFIT-MAXIMIZING TRADING SYSTEM v2.0")
    log_info("🎯 Profit-First Learning | Multi-Horizon Predictions | Dynamic Position Sizing")
    log_info("=" * 90)


async def main():
    """Main entry point"""
    system = None
    try:
        _log_startup_banner()
        system = ProfitMaximizingTradingSystem()

        # Register cleanup handlers
        import signal
        import atexit

        def signal_handler(signum, frame):
            log_info("🛑 Received shutdown signal, cleaning up...")
            if system:
                system.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(lambda: system.cleanup() if system else None)

        await system.run()
        
    except (KeyboardInterrupt, asyncio.CancelledError):
        log_info("🛑 System terminated by user")
        if system:
            system.cleanup()
    except Exception as e:
        log_error(f"💥 System failed: {e}")
        traceback.print_exc()
        if system:
            system.cleanup()
        sys.exit(1)
    finally:
        if system:
            system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())