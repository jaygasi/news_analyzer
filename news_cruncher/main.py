"""
Main application for simplified financial news analysis system
Python 3.13.3 compatible
"""
import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List
from config import Config
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from database.article_tracker import ArticleTracker
from data_loaders.news_fetcher import NewsFetcher
from core.ticker_aggregator import TickerAggregator
from analysis.multi_llm_analyzer import MultiLLMAnalyzer
from analysis.technical_analyzer_simple import TechnicalAnalyzer
from core.decision_engine import DecisionEngine
from output.csv_logger import CSVLogger


class FinancialNewsAnalyzer:
    """Main application class for financial news analysis"""
    
    def __init__(self) -> None:
        """Initialize the financial news analyzer"""
        self.running = True
        self.cycle_count = 0
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        log_info("Financial News Analyzer initialized successfully")
    
    def _validate_configuration(self) -> None:
        """Validate required configuration"""
        api_keys = Config.validate_api_keys()
        
        if not api_keys['fmp']:
            raise ValueError("FMP_API_KEY is required")
        
        # Log available services
        available_services = [key for key, available in api_keys.items() if available]
        log_info(f"Available API services: {', '.join(available_services)}")
        
        if len(available_services) < 2:
            log_warning("Limited API services available - consider adding more for redundancy")
    
    def _initialize_components(self) -> None:
        """Initialize all system components"""
        try:
            # Database
            self.article_tracker = ArticleTracker()
            
            # Data fetching
            self.news_fetcher = NewsFetcher(Config.FMP_API_KEY)
            
            # Core processing
            self.ticker_aggregator = TickerAggregator()
            
            # Analysis components
            self.llm_analyzer = MultiLLMAnalyzer()
            self.technical_analyzer = TechnicalAnalyzer(self.news_fetcher)
            
            # Decision making
            self.decision_engine = DecisionEngine()
            
            # Output
            self.csv_logger = CSVLogger()
            
            log_info("All components initialized successfully")
            
        except Exception as e:
            log_error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum: int, frame) -> None:
            log_info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> None:
        """Main application loop"""
        log_info("🚀 Starting Financial News Analysis System")
        self._print_startup_info()
        
        try:
            while self.running:
                await self._process_cycle()
                
                if self.running:
                    log_info("Cycle completed, waiting 5 minutes before next run...")
                    await asyncio.sleep(300)  # Wait 5 minutes between cycles
                    
        except KeyboardInterrupt:
            log_info("Interrupted by user")
        except Exception as e:
            log_error(f"Fatal error in main loop: {e}")
            raise
        finally:
            await self._shutdown()
    
    def _print_startup_info(self) -> None:
        """Print startup information"""
        log_info("=" * 60)
        log_info("📊 FINANCIAL NEWS ANALYSIS SYSTEM")
        log_info("=" * 60)
        log_info(f"🔑 Configuration:")
        log_info(f"   Min confidence threshold: {Config.MIN_CONFIDENCE_THRESHOLD}")
        log_info(f"   Max news age: {Config.MAX_NEWS_AGE_HOURS} hours")
        log_info(f"   Output file: {Config.CSV_OUTPUT_PATH}")
        log_info(f"   Database: {Config.SQLITE_DB_PATH}")
        
        # Service status
        service_status = self.llm_analyzer.get_service_status()
        available_services = [name for name, status in service_status.items() if status['available']]
        log_info(f"🤖 Available AI services: {', '.join(available_services)}")
        
        log_info("=" * 60)
    
    async def _process_cycle(self) -> None:
        """Process one complete analysis cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        log_info(f"🔄 Starting analysis cycle #{self.cycle_count}")
        
        try:
            # Step 1: Fetch latest news
            log_info("📰 Step 1: Fetching latest financial news...")
            all_articles = self.news_fetcher.fetch_all_news()
            
            if not all_articles:
                log_info("No news articles found, ending cycle")
                return
            
            # Step 2: Filter unprocessed articles
            log_info("🔍 Step 2: Filtering unprocessed articles...")
            unprocessed_articles = self.article_tracker.filter_unprocessed_articles(all_articles)
            
            if not unprocessed_articles:
                log_info("No new articles to process, ending cycle")
                return
            
            # Step 3: Aggregate by ticker
            log_info("📊 Step 3: Aggregating articles by ticker...")
            ticker_buckets = self.ticker_aggregator.aggregate_by_ticker(unprocessed_articles)
            
            if not ticker_buckets:
                log_info("No valid ticker buckets created, ending cycle")
                self._mark_articles_processed(unprocessed_articles)
                return
            
            # Step 4: Prioritize tickers
            log_info("🎯 Step 4: Prioritizing tickers for analysis...")
            prioritized_tickers = self.ticker_aggregator.prioritize_tickers(ticker_buckets)
            
            # Step 5: Analyze each ticker
            log_info(f"🧠 Step 5: Analyzing {len(prioritized_tickers)} tickers...")
            ticker_analyses = await self._analyze_tickers(prioritized_tickers, ticker_buckets)
            
            # Step 6: Make trading decisions
            log_info("⚖️ Step 6: Making trading decisions...")
            decisions = self.decision_engine.batch_process_decisions(ticker_analyses)
            
            # Step 7: Log decisions
            if decisions:
                log_info(f"📝 Step 7: Logging {len(decisions)} trading decisions...")
                logged_count = self.csv_logger.log_decisions_batch(decisions)
                log_info(f"✅ Successfully logged {logged_count} decisions")
                
                # Print summary of decisions
                self._print_decisions_summary(decisions)
            else:
                log_info("📝 Step 7: No high-confidence decisions to log")
            
            # Step 8: Mark articles as processed
            log_info("✅ Step 8: Marking articles as processed...")
            self._mark_articles_processed(all_articles, decisions)
            
            # Cycle summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            log_info(f"🏁 Cycle #{self.cycle_count} completed in {cycle_duration:.1f} seconds")
            
            # Cleanup old data periodically
            if self.cycle_count % 10 == 0:
                self._periodic_cleanup()
                
        except Exception as e:
            log_error(f"Error in processing cycle: {e}")
            # Still mark articles as processed to avoid reprocessing
            if 'unprocessed_articles' in locals():
                self._mark_articles_processed(unprocessed_articles)
    
    async def _analyze_tickers(self, prioritized_tickers: List[str], 
                              ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Analyze each ticker with news and technical analysis"""
        ticker_analyses = {}
        
        for i, ticker in enumerate(prioritized_tickers[:50], 1):  # Limit to top 50 tickers
            try:
                log_debug(f"Analyzing {ticker} ({i}/{min(len(prioritized_tickers), 50)})")
                
                articles = ticker_buckets[ticker]
                
                # News analysis
                news_prediction = self.llm_analyzer.analyze_news_direction(ticker, articles)
                
                # Technical analysis
                technical_signal = self.technical_analyzer.analyze_ticker(ticker)
                
                ticker_analyses[ticker] = {
                    'news_prediction': news_prediction,
                    'technical_signal': technical_signal,
                    'article_count': len(articles)
                }
                
                # Small delay to be respectful to APIs
                await asyncio.sleep(0.5)
                
            except Exception as e:
                log_error(f"Error analyzing {ticker}: {e}")
                continue
        
        return ticker_analyses
    
    def _mark_articles_processed(self, articles: List[Dict[str, Any]], 
                                decisions: List = None) -> None:
        """Mark articles as processed in database"""
        try:
            decisions_by_ticker = {}
            if decisions:
                for decision in decisions:
                    decisions_by_ticker[decision.ticker] = decision
            
            for article in articles:
                ticker = article.get('symbol', '')
                decision = decisions_by_ticker.get(ticker)
                
                decision_str = decision.decision if decision else ''
                confidence = decision.confidence if decision else 0.0
                
                self.article_tracker.mark_article_processed(
                    article, decision_str, confidence
                )
                
        except Exception as e:
            log_error(f"Error marking articles as processed: {e}")
    
    def _print_decisions_summary(self, decisions: List) -> None:
        """Print summary of trading decisions"""
        if not decisions:
            return
        
        stats = self.decision_engine.get_decision_statistics(decisions)
        
        log_info("📈 TRADING DECISIONS SUMMARY:")
        log_info(f"   Total decisions: {stats['total_decisions']}")
        log_info(f"   LONG positions: {stats['long_decisions']}")
        log_info(f"   SHORT positions: {stats['short_decisions']}")
        log_info(f"   Average confidence: {stats['avg_confidence']:.3f}")
        
        # Show top decisions
        top_decisions = sorted(decisions, key=lambda x: x.confidence, reverse=True)[:5]
        log_info("   Top decisions:")
        for decision in top_decisions:
            log_info(f"     {decision.ticker}: {decision.decision} (conf: {decision.confidence:.3f})")
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup tasks"""
        try:
            log_info("🧹 Performing periodic cleanup...")
            
            # Cleanup old articles
            deleted_count = self.article_tracker.cleanup_old_articles(days=30)
            if deleted_count > 0:
                log_info(f"   Cleaned up {deleted_count} old articles")
            
            # Get and log statistics
            db_stats = self.article_tracker.get_statistics()
            csv_stats = self.csv_logger.get_csv_statistics()
            
            log_info("📊 System Statistics:")
            log_info(f"   Database: {db_stats.get('total_articles', 0)} articles, {db_stats.get('unique_tickers', 0)} tickers")
            log_info(f"   CSV: {csv_stats.get('total_decisions', 0)} decisions logged")
            
        except Exception as e:
            log_error(f"Error in periodic cleanup: {e}")
    
    async def _shutdown(self) -> None:
        """Graceful shutdown"""
        log_info("🛑 Initiating graceful shutdown...")
        
        try:
            # Print final statistics
            db_stats = self.article_tracker.get_statistics()
            csv_stats = self.csv_logger.get_csv_statistics()
            service_status = self.llm_analyzer.get_service_status()
            
            log_info("📊 FINAL STATISTICS:")
            log_info(f"   Cycles completed: {self.cycle_count}")
            log_info(f"   Articles processed: {db_stats.get('total_articles', 0)}")
            log_info(f"   Decisions logged: {csv_stats.get('total_decisions', 0)}")
            log_info(f"   Unique tickers analyzed: {db_stats.get('unique_tickers', 0)}")
            
            # Service usage
            for service_name, status in service_status.items():
                if status['available']:
                    requests = status.get('requests_today', 0)
                    limit = status.get('daily_limit', 0)
                    log_info(f"   {service_name}: {requests}/{limit} requests used")
            
            log_info("✅ Shutdown completed successfully")
            
        except Exception as e:
            log_error(f"Error during shutdown: {e}")


async def main() -> int:
    """Main entry point"""
    try:
        # Create and run the analyzer
        analyzer = FinancialNewsAnalyzer()
        await analyzer.run()
        return 0
        
    except KeyboardInterrupt:
        log_info("Application interrupted by user")
        return 0
        
    except Exception as e:
        log_error(f"Fatal application error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)