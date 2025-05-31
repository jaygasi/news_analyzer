"""
Optimized comprehensive news catalyst trading system with enhanced logging
"""
import asyncio
import time
from datetime import datetime, timezone
import signal
import sys
from typing import Optional
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning
from data_loaders.simple_fmp_loader import SimpleFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader


class ComprehensiveTradingSystem:
    """Optimized comprehensive trading system with enhanced error handling."""
    
    def __init__(self) -> None:
        """Initialize trading system with proper validation."""
        # Validate and extract API key with proper type handling
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        # Store validated API key (now guaranteed to be str)
        self.fmp_api_key: str = api_key
        
        # Initialize components with error handling
        self._initialize_components()
        
        # System state
        self.universe: list = []
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
                # Validate and clean symbols
                valid_symbols = []
                for symbol in screener_df['symbol'].tolist():
                    if isinstance(symbol, str) and symbol.strip():
                        clean_symbol = symbol.strip().upper()
                        if len(clean_symbol) <= 5 and clean_symbol.isalpha():
                            valid_symbols.append(clean_symbol)
                
                self.universe = valid_symbols
                log_info(f"Loaded {len(self.universe)} valid symbols in universe")
                
                # LOG THE ACTUAL UNIVERSE
                self._log_trading_universe()
                
                return len(self.universe) > 0
            else:
                log_error("Failed to load trading universe - no data returned")
                return False
                
        except Exception as e:
            log_error(f"Error initializing universe: {e}")
            return False
    
    def _log_trading_universe(self) -> None:
        """Log the actual stocks in the trading universe."""
        if not self.universe:
            return
        
        log_info("=" * 60)
        log_info(f"TRADING UNIVERSE ({len(self.universe)} stocks):")
        log_info("=" * 60)
        
        # Group symbols by sectors for better readability
        tech_stocks = []
        healthcare_stocks = []
        finance_stocks = []
        energy_stocks = []
        other_stocks = []
        
        # Simple sector classification
        for symbol in self.universe:
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'CRM', 'ORCL', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO']:
                tech_stocks.append(symbol)
            elif symbol in ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB', 'VRTX', 'REGN', 'ILMN']:
                healthcare_stocks.append(symbol)
            elif symbol in ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW', 'CB', 'SPGI']:
                finance_stocks.append(symbol)
            elif symbol in ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OXY', 'VLO', 'MPC']:
                energy_stocks.append(symbol)
            else:
                other_stocks.append(symbol)
        
        # Log by sectors
        if tech_stocks:
            log_info(f"TECHNOLOGY ({len(tech_stocks)}): {', '.join(sorted(tech_stocks))}")
        
        if healthcare_stocks:
            log_info(f"HEALTHCARE ({len(healthcare_stocks)}): {', '.join(sorted(healthcare_stocks))}")
        
        if finance_stocks:
            log_info(f"FINANCIALS ({len(finance_stocks)}): {', '.join(sorted(finance_stocks))}")
        
        if energy_stocks:
            log_info(f"ENERGY ({len(energy_stocks)}): {', '.join(sorted(energy_stocks))}")
        
        # Show first 50 other stocks
        if other_stocks:
            if len(other_stocks) <= 50:
                log_info(f"OTHER SECTORS ({len(other_stocks)}): {', '.join(sorted(other_stocks))}")
            else:
                log_info(f"OTHER SECTORS ({len(other_stocks)} total):")
                # Print in batches of 20
                sorted_others = sorted(other_stocks)
                for i in range(0, min(50, len(sorted_others)), 20):
                    batch = sorted_others[i:i+20]
                    log_info(f"  {', '.join(batch)}")
                if len(other_stocks) > 50:
                    log_info(f"  ... and {len(other_stocks) - 50} more")
        
        log_info("=" * 60)
        log_info(f"UNIVERSE CRITERIA: Market Cap >$100M, Price $2-$500, Volume >50K, NYSE/NASDAQ")
        log_info("=" * 60)
    
    def process_comprehensive_news_cycle(self) -> None:
        """Optimized comprehensive news processing with error recovery."""
        try:
            # Get latest news with validation
            raw_news_df = self.fmp_loader.get_news_rss()
            if raw_news_df is None or raw_news_df.empty:
                log_warning("No news data received")
                return
            
            # Filter to universe symbols efficiently
            universe_set = set(self.universe)
            news_symbols = raw_news_df['symbol'].astype(str).str.upper()
            mask = news_symbols.isin(universe_set)
            filtered_news_df = raw_news_df[mask].copy()
            
            if filtered_news_df.empty:
                log_info("No news for universe symbols")
                return
            
            log_info(f"Raw news: {len(raw_news_df)} articles, "
                    f"filtered to universe: {len(filtered_news_df)} articles")
            
            # Apply news quality pre-filters
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
            
            # Enhanced news analysis with technical validation
            analyses = self.news_analyzer.analyze_news_with_technical(
                quality_filtered_df, prices_df
            )
            
            if not analyses:
                log_info("No valid news analyses generated")
                return
            
            # Filter for high combined confidence
            high_confidence = self.news_analyzer.filter_high_confidence(
                analyses, use_combined_confidence=True
            )
            
            if high_confidence:
                log_info(f"[SIGNALS] Found {len(high_confidence)} high-quality trading opportunities")
                self.trader.process_news_signals(high_confidence, prices_df)
            else:
                log_info("No high-confidence signals after filtering")
                
        except Exception as e:
            log_error(f"Error in comprehensive news cycle: {e}")
    
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
            
            log_info(f"[STATUS] System Status:")
            log_info(f"   Active Positions: {active_positions}")
            log_info(f"   Daily P&L: ${daily_pnl:.2f}")
            log_info(f"   Confidence Threshold: {CONFIG.min_confidence_score}")
            
            if filter_perf:
                log_info(f"[ANALYTICS] Filter Performance:")
                
                # Best performing regime
                if 'regime_performance' in filter_perf:
                    regime_data = filter_perf['regime_performance']
                    if 'sum' in regime_data and regime_data['sum']:
                        best_regime = max(regime_data['sum'].items(), 
                                        key=lambda x: x[1] if x[1] is not None else -float('inf'))
                        log_info(f"   Best Market Regime: {best_regime[0]}")
                
                # Average metrics
                avg_stress = filter_perf.get('avg_market_stress', 0)
                avg_time = filter_perf.get('avg_time_score', 0)
                win_rate = filter_perf.get('win_rate', 0)
                total_trades = filter_perf.get('total_trades', 0)
                
                log_info(f"   Total Trades: {total_trades}")
                log_info(f"   Win Rate: {win_rate:.1%}")
                log_info(f"   Avg Market Stress: {avg_stress:.2f}")
                log_info(f"   Avg Time Score: {avg_time:.2f}")
                
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    async def run(self) -> None:
        """Main run loop with optimized timing and error recovery."""
        log_info("[START] Starting COMPREHENSIVE News Catalyst Trading System")
        
        # Log system features (without emojis)
        self._log_system_features()
        
        # Initialize universe
        if not self.initialize_universe():
            log_error("Failed to initialize trading universe")
            return
        
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
                    error_count = 0  # Reset error count on success
                
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
        
        log_info("[STOP] Comprehensive trading system stopped")
    
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