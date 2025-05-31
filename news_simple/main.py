"""
Comprehensive Enhanced News Catalyst Trading System
"""
import asyncio
import time
from datetime import datetime
import signal
import sys
from config import CONFIG
from utils.simple_logger import log_info, log_error
from data_loaders.simple_fmp_loader import SimpleFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader

class ComprehensiveTradingSystem:
    """Comprehensive trading system with all quality filters"""
    
    def __init__(self):
        self.fmp_api_key = CONFIG.get_api_key('fmp')
        if not self.fmp_api_key:
            raise ValueError("FMP_API_KEY environment variable required")
        
        self.fmp_loader = SimpleFMPLoader(self.fmp_api_key)
        self.news_analyzer = EnhancedNewsAnalyzer(self.fmp_loader)
        self.trader = EnhancedTrader(self.fmp_loader)  # Pass fmp_loader for filters
        self.universe = []
        self.running = True
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        log_info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def initialize_universe(self):
        """Initialize trading universe"""
        log_info("Initializing comprehensive trading universe...")
        
        screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
        if screener_df is not None and not screener_df.empty:
            self.universe = screener_df['symbol'].tolist()
            log_info(f"Loaded {len(self.universe)} symbols in universe")
        else:
            log_error("Failed to load trading universe")
            return False
        
        return True
    
    def process_comprehensive_news_cycle(self):
        """Comprehensive news processing with all quality filters"""
        try:
            # Get latest news
            raw_news_df = self.fmp_loader.get_news_rss()
            if raw_news_df is None or raw_news_df.empty:
                return
            
            # Filter to universe symbols
            raw_news_df = raw_news_df[raw_news_df['symbol'].isin(self.universe)]
            if raw_news_df.empty:
                return
            
            log_info(f"Raw news: {len(raw_news_df)} articles")
            
            # Apply news quality pre-filters
            filtered_news_df = self.trader.pre_filter_news(raw_news_df)
            if filtered_news_df is None or filtered_news_df.empty:
                log_info("All news filtered out by quality filters")
                return
            
            log_info(f"Quality filtered news: {len(filtered_news_df)} articles")
            
            # Get current prices for technical analysis
            relevant_symbols = filtered_news_df['symbol'].unique().tolist()
            prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
            
            if prices_df is None:
                log_error("Failed to get current prices")
                return
            
            # Enhanced news analysis with technical validation
            analyses = self.news_analyzer.analyze_news_with_technical(filtered_news_df, prices_df)
            if not analyses:
                log_info("No valid news analyses generated")
                return
            
            # Filter for high combined confidence
            high_confidence = self.news_analyzer.filter_high_confidence(
                analyses, use_combined_confidence=True
            )
            
            if high_confidence:
                log_info(f"🎯 Found {len(high_confidence)} high-quality trading opportunities")
                
                # Process with comprehensive filtering
                self.trader.process_news_signals(high_confidence, prices_df)
            else:
                log_info("No high-confidence signals after filtering")
        
        except Exception as e:
            log_error(f"Error in comprehensive news cycle: {e}")
    
    def check_positions_cycle(self):
        """Check positions with enhanced exit logic"""
        try:
            active_trades = self.trader.get_active_positions()
            if not active_trades:
                return
            
            symbols = [trade.symbol for trade in active_trades]
            prices_df = self.fmp_loader.get_real_time_prices(symbols)
            
            if prices_df is not None:
                self.trader.check_exits(prices_df)
        
        except Exception as e:
            log_error(f"Error checking positions: {e}")
    
    def print_comprehensive_status(self):
        """Print comprehensive system status"""
        active_positions = len(self.trader.get_active_positions())
        daily_pnl = self.trader.get_daily_pnl()
        
        # Get filter performance analytics
        filter_perf = self.trader.get_filter_performance()
        
        log_info(f"📊 SYSTEM STATUS:")
        log_info(f"   Active Positions: {active_positions}")
        log_info(f"   Daily P&L: ${daily_pnl:.2f}")
        
        if filter_perf:
            log_info(f"📈 FILTER PERFORMANCE:")
            if 'regime_performance' in filter_perf:
                regime_data = filter_perf['regime_performance']
                log_info(f"   Best Market Regime: {max(regime_data['sum'].items(), key=lambda x: x[1])[0] if regime_data['sum'] else 'N/A'}")
            
            avg_stress = filter_perf.get('avg_market_stress', 0)
            avg_time = filter_perf.get('avg_time_score', 0)
            log_info(f"   Avg Market Stress: {avg_stress:.2f}")
            log_info(f"   Avg Time Score: {avg_time:.2f}")
    
    async def run(self):
        """Main run loop with comprehensive filtering"""
        log_info("🚀 Starting COMPREHENSIVE News Catalyst Trading System")
        log_info("📋 Features Enabled:")
        log_info("   ✅ FinBERT + Keyword Ensemble Analysis")
        log_info("   ✅ Technical Analysis (RSI, MA, Momentum, Liquidity)")
        log_info("   ✅ Market Condition Filters (Hours, VIX, Regime)")
        log_info("   ✅ News Quality Filters (Freshness, Deduplication)")
        log_info("   ✅ Price Action Filters (Gaps, Volatility, Extremes)")
        log_info("   ✅ Portfolio Risk Management (Concentration, Heat)")
        log_info("   ✅ Entry Timing Optimization")
        log_info("   ✅ Dynamic Position Sizing")
        
        log_info(f"💰 Configuration: ${CONFIG.position_size:.0f} base position, "
                f"{CONFIG.stop_loss_pct*100:.1f}% stop loss, "
                f"{CONFIG.take_profit_pct*100:.1f}% take profit")
        
        if not self.initialize_universe():
            return
        
        last_news_check = 0
        last_position_check = 0
        last_status_print = 0
        
        log_info("🎯 Comprehensive trading system running... Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Comprehensive news processing
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    self.process_comprehensive_news_cycle()
                    last_news_check = current_time
                
                # Enhanced position monitoring
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    self.check_positions_cycle()
                    last_position_check = current_time
                
                # Comprehensive status update
                if current_time - last_status_print >= 300:  # Every 5 minutes
                    self.print_comprehensive_status()
                    last_status_print = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                log_error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
        
        log_info("🏁 Comprehensive trading system stopped")

async def main():
    try:
        system = ComprehensiveTradingSystem()
        await system.run()
    except KeyboardInterrupt:
        log_info("Interrupted by user")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)