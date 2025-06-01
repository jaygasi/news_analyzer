"""
Optimized comprehensive news catalyst trading system with enhanced logging
"""
import asyncio
import time
from datetime import datetime, timezone
import signal
import sys
from typing import Optional, List, Dict, Any
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning
from data_loaders.simple_fmp_loader import SimpleFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader


class ComprehensiveTradingSystem:
    """Optimized comprehensive trading system with enhanced error handling."""
    
    def __init__(self) -> None:
        """Initialize trading system with proper validation."""
        # Validate and extract API key
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        self.fmp_api_key: str = api_key
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.universe: List[str] = []
        self.running = True
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _initialize_components(self) -> None:
        """Initialize all system components with error handling."""
        try:
            self.fmp_loader = SimpleFMPLoader(self.fmp_api_key)
            self.news_analyzer = EnhancedNewsAnalyzer(self.fmp_loader)
            self.trader = EnhancedTrader(self.fmp_loader)
            log_info("All system components initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        log_info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize_universe(self) -> bool:
        """Initialize trading universe with comprehensive validation and logging."""
        log_info("Initializing comprehensive trading universe...")
        
        try:
            screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
            
            if screener_df is not None and not screener_df.empty:
                # FMP loader already did the filtering, so just extract symbols
                # No need for additional filtering here since FMP loader handles it properly
                self.universe = screener_df['symbol'].str.upper().tolist()
                
                log_info(f"Loaded {len(self.universe)} valid symbols in universe")
                
                # Debug: show first 20 symbols to verify they look correct
                sample_symbols = self.universe[:20]
                log_info(f"Sample universe symbols: {sample_symbols}")
                
                self._log_trading_universe()
                
                return len(self.universe) > 0
            else:
                log_error("Failed to load trading universe - no data returned")
                return False
                
        except Exception as e:
            log_error(f"Error initializing universe: {e}")
            import traceback
            log_error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _log_trading_universe(self) -> None:
        """Log the actual stocks in the trading universe."""
        if not self.universe:
            return
        
        log_info("=" * 60)
        log_info(f"TRADING UNIVERSE ({len(self.universe)} stocks):")
        log_info("=" * 60)
        
        # Categorize stocks for better readability
        categories = self._categorize_stocks()
        
        # Log each category
        self._log_stock_categories(categories)
        
        log_info("=" * 60)
        log_info("UNIVERSE CRITERIA: Market Cap >$100M, Price $2-$500, Volume >50K, NYSE/NASDAQ")
        log_info("=" * 60)
    
    def _categorize_stocks(self) -> Dict[str, List[str]]:
        """Categorize stocks by sector for better readability."""
        categories = {
            'TECHNOLOGY': [],
            'HEALTHCARE': [],
            'FINANCIALS': [],
            'ENERGY': [],
            'OTHER SECTORS': []
        }
        
        # Define sector mappings
        sector_mappings = self._get_sector_mappings()
        
        # Categorize symbols
        for symbol in self.universe:
            self._assign_symbol_to_category(symbol, sector_mappings, categories)
        
        return categories
    
    def _get_sector_mappings(self) -> Dict[str, set]:
        """Get predefined sector mappings."""
        return {
            'TECHNOLOGY': {
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 
                'CRM', 'ORCL', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO'
            },
            'HEALTHCARE': {
                'JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 
                'AMGN', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN'
            },
            'FINANCIALS': {
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'CB', 'SPGI'
            },
            'ENERGY': {
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OXY', 'VLO', 'MPC'
            }
        }
    
    def _assign_symbol_to_category(self, symbol: str, sector_mappings: Dict[str, set], 
                                 categories: Dict[str, List[str]]) -> None:
        """Assign a symbol to the appropriate category."""
        for category, symbols in sector_mappings.items():
            if symbol in symbols:
                categories[category].append(symbol)
                return
        
        # If not found in any specific sector, add to "OTHER SECTORS"
        categories['OTHER SECTORS'].append(symbol)
    
    def _log_stock_categories(self, categories: Dict[str, List[str]]) -> None:
        """Log stock categories with proper formatting."""
        for category, stocks in categories.items():
            if not stocks:
                continue
                
            if len(stocks) <= 50:
                log_info(f"{category} ({len(stocks)}): {', '.join(sorted(stocks))}")
            else:
                self._log_large_category(category, stocks)
    
    def _log_large_category(self, category: str, stocks: List[str]) -> None:
        """Log large categories in batches."""
        log_info(f"{category} ({len(stocks)} total):")
        sorted_stocks = sorted(stocks)
        
        # Print in batches of 20
        for i in range(0, min(50, len(sorted_stocks)), 20):
            batch = sorted_stocks[i:i+20]
            log_info(f"  {', '.join(batch)}")
        
        if len(stocks) > 50:
            log_info(f"  ... and {len(stocks) - 50} more")
    
    def process_comprehensive_news_cycle(self) -> None:
        """Optimized comprehensive news processing with error recovery."""
        try:
            # Get and filter news
            raw_news_df = self.fmp_loader.get_news_rss()
            if raw_news_df is None or raw_news_df.empty:
                log_warning("No news data received")
                return
            
            # Filter to universe symbols efficiently
            universe_set = set(self.universe)
            filtered_news_df = raw_news_df[
                raw_news_df['symbol'].str.upper().isin(universe_set)
            ].copy()
            
            if filtered_news_df.empty:
                log_info("No news for universe symbols")
                return
            
            log_info(f"Raw news: {len(raw_news_df)} articles, "
                    f"filtered to universe: {len(filtered_news_df)} articles")
            
            # Apply quality filters and process
            self._process_filtered_news(filtered_news_df)
                
        except Exception as e:
            log_error(f"Error in comprehensive news cycle: {e}")
    
    def _process_filtered_news(self, filtered_news_df) -> None:
        """Process filtered news through the analysis pipeline."""
        # Apply quality filters
        quality_filtered_df = self.trader.pre_filter_news(filtered_news_df)
        if quality_filtered_df is None or quality_filtered_df.empty:
            log_info("All news filtered out by quality filters")
            return
        
        log_info(f"Quality filtered news: {len(quality_filtered_df)} articles")
        
        # Get current prices for technical analysis
        relevant_symbols = quality_filtered_df['symbol'].unique().tolist()
        prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
        
        if prices_df is None or prices_df.empty:
            log_warning("Failed to get current prices for technical analysis")
            return
        
        # Enhanced news analysis
        analyses = self.news_analyzer.analyze_news_with_technical(
            quality_filtered_df, prices_df
        )
        
        if not analyses:
            log_info("No valid news analyses generated")
            return
        
        # Filter for high confidence and process signals
        self._process_analysis_results(analyses, prices_df)
    
    def _process_analysis_results(self, analyses, prices_df) -> None:
        """Process analysis results and create trades."""
        high_confidence = self.news_analyzer.filter_high_confidence(
            analyses, use_combined_confidence=True
        )
        
        if high_confidence:
            log_info(f"[SIGNALS] Found {len(high_confidence)} high-quality trading opportunities")
            self.trader.process_news_signals(high_confidence, prices_df)
        else:
            log_info("No high-confidence signals after filtering")
    
    def check_positions_cycle(self) -> None:
        """Check positions with enhanced error handling."""
        try:
            active_trades = self.trader.get_active_positions()
            if not active_trades:
                return
            
            symbols = [trade.symbol for trade in active_trades]
            prices_df = self.fmp_loader.get_real_time_prices(symbols)
            
            if prices_df is not None and not prices_df.empty:
                self.trader.check_exits(prices_df)
            else:
                log_warning("Failed to get prices for position monitoring")
                
        except Exception as e:
            log_error(f"Error checking positions: {e}")
    
    def print_comprehensive_status(self) -> None:
        """Print system status with enhanced metrics."""
        try:
            active_positions = len(self.trader.get_active_positions())
            daily_pnl = self.trader.get_daily_pnl()
            
            # Get filter performance analytics
            filter_perf = self.trader.get_filter_performance()
            
            log_info("[STATUS] System Status:")
            log_info(f"   Active Positions: {active_positions}")
            log_info(f"   Daily P&L: ${daily_pnl:.2f}")
            log_info(f"   Confidence Threshold: {CONFIG.min_confidence_score}")
            
            if filter_perf:
                self._log_performance_analytics(filter_perf)
                
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    def _log_performance_analytics(self, filter_perf: Dict[str, Any]) -> None:
        """Log detailed performance analytics."""
        log_info("[ANALYTICS] Filter Performance:")
        
        # Best performing regime
        self._log_best_regime(filter_perf)
        
        # Summary metrics
        self._log_summary_metrics(filter_perf)
    
    def _log_best_regime(self, filter_perf: Dict[str, Any]) -> None:
        """Log the best performing market regime."""
        if 'regime_performance' not in filter_perf:
            return
            
        regime_data = filter_perf['regime_performance']
        if 'sum' not in regime_data or not regime_data['sum']:
            return
            
        best_regime = max(
            regime_data['sum'].items(), 
            key=lambda x: x[1] if x[1] is not None else -float('inf')
        )
        log_info(f"   Best Market Regime: {best_regime[0]}")
    
    def _log_summary_metrics(self, filter_perf: Dict[str, Any]) -> None:
        """Log summary performance metrics."""
        metrics = ['total_trades', 'win_rate', 'avg_market_stress', 'avg_time_score']
        
        for metric in metrics:
            if metric not in filter_perf:
                continue
                
            value = filter_perf[metric]
            if metric == 'win_rate':
                log_info(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
            elif 'avg' in metric:
                log_info(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                log_info(f"   {metric.replace('_', ' ').title()}: {value}")
    
    async def run(self) -> None:
        """Main run loop with optimized timing and error recovery."""
        log_info("[START] Starting COMPREHENSIVE News Catalyst Trading System")
        
        # Log system features
        self._log_system_features()
        
        # Initialize universe
        if not self.initialize_universe():
            log_error("Failed to initialize trading universe")
            return
        
        # Run main loop
        await self._main_trading_loop()
        
        log_info("[STOP] Comprehensive trading system stopped")
    
    async def _main_trading_loop(self) -> None:
        """Main trading loop with error recovery."""
        # Initialize timing variables
        last_news_check = 0.0
        last_position_check = 0.0
        last_status_print = 0.0
        
        status_interval = 300  # 5 minutes
        error_count = 0
        max_errors = 10
        
        log_info("[TARGET] Comprehensive trading system running... Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # News processing cycle
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    self.process_comprehensive_news_cycle()
                    last_news_check = current_time
                    error_count = 0  # Reset on success
                
                # Position monitoring cycle
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    self.check_positions_cycle()
                    last_position_check = current_time
                
                # Status reporting cycle
                if current_time - last_status_print >= status_interval:
                    self.print_comprehensive_status()
                    last_status_print = current_time
                
                # Short sleep to prevent excessive CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                error_count += 1
                log_error(f"Error in main loop (count: {error_count}): {e}")
                
                if error_count >= max_errors:
                    log_error(f"Too many errors ({error_count}), shutting down")
                    self.running = False
                else:
                    # Exponential backoff for error recovery
                    sleep_time = min(2 ** error_count, 60)
                    log_info(f"Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)
    
    def _log_system_features(self) -> None:
        """Log system features without problematic characters."""
        features = [
            "[OK] FinBERT + Keyword Ensemble Analysis",
            "[OK] Technical Analysis (RSI, MA, Momentum, Liquidity)",
            "[OK] Market Condition Filters (Hours, VIX, Regime)",
            "[OK] News Quality Filters (Freshness, Deduplication)",
            "[OK] Price Action Filters (Gaps, Volatility, Extremes)",
            "[OK] Portfolio Risk Management (Concentration, Heat)",
            "[OK] Entry Timing Optimization",
            "[OK] Dynamic Position Sizing"
        ]
        
        log_info("[INFO] Features Enabled:")
        for feature in features:
            log_info(f"   {feature}")
        
        log_info(f"[MONEY] Configuration: ${CONFIG.position_size:.0f} base position, "
                f"{CONFIG.stop_loss_pct*100:.1f}% stop loss, "
                f"{CONFIG.take_profit_pct*100:.1f}% take profit")


async def main() -> int:
    """Main entry point with comprehensive error handling."""
    try:
        system = ComprehensiveTradingSystem()
        await system.run()
        return 0
        
    except KeyboardInterrupt:
        log_info("System interrupted by user")
        return 0
        
    except Exception as e:
        log_error(f"Fatal system error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start system: {e}")
        sys.exit(1)