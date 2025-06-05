"""
Main application for simplified financial news analysis system with dynamic time tracking
Python 3.13.3 compatible
"""
import asyncio
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from config import Config
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from database.article_tracker import ArticleTracker
from database.run_tracker import RunTracker
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
            # Database components
            self.article_tracker = ArticleTracker()
            self.run_tracker = RunTracker()  # New component for dynamic time tracking
            
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
        """Main application loop with dynamic time tracking"""
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
        # Add debugging
        self.run_tracker.debug_run_history()
        
        # Get last run info for dynamic time display
        last_run_time = self.run_tracker.get_last_successful_run_time()
        time_since_last = datetime.now(timezone.utc) - last_run_time
        
        log_info("=" * 60)
        log_info("📊 FINANCIAL NEWS ANALYSIS SYSTEM")
        log_info("=" * 60)
        log_info(f"🔑 Configuration:")
        log_info(f"   Min confidence threshold: {Config.MIN_CONFIDENCE_THRESHOLD}")
        log_info(f"   Last successful run: {last_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Show actual lookback that will be used
        cutoff_time = self._get_dynamic_news_cutoff_time()
        actual_lookback = (datetime.now(timezone.utc) - cutoff_time).total_seconds() / 3600
        log_info(f"   Actual lookback: {actual_lookback:.1f} hours")
        
        log_info(f"   Max articles per cycle: {Config.MAX_NEWS_ARTICLES}")
        log_info(f"   Output file: {Config.CSV_OUTPUT_PATH}")
        log_info(f"   Database: {Config.SQLITE_DB_PATH}")
        
        # Service status
        service_status = self.llm_analyzer.get_service_status()
        available_services = [name for name, status in service_status.items() if status['available']]
        log_info(f"🤖 Available AI services: {', '.join(available_services)}")
        
        # Run statistics
        run_stats = self.run_tracker.get_run_statistics(days=7)
        if run_stats:
            log_info(f"📈 Last 7 days: {run_stats['successful_runs']}/{run_stats['total_runs']} successful runs")
        
        log_info("=" * 60)
    
    def _get_dynamic_news_cutoff_time(self) -> datetime:
        """Get dynamic cutoff time based on last successful run"""
        # Ensure we're always working in UTC
        now = datetime.now(timezone.utc)
        last_run_time = self.run_tracker.get_last_successful_run_time()
        
        # Convert local time to UTC for logging clarity
        local_time = datetime.now()  # Your EST time
        log_debug(f"Local time (EST): {local_time}")
        log_debug(f"Current UTC time: {now}")
        log_debug(f"Last run time (UTC): {last_run_time}")
        
        # Calculate time since last run
        time_since_last = now - last_run_time
        log_debug(f"Time since last run: {time_since_last.total_seconds() / 3600:.1f} hours")
        
        # Apply constraints with more reasonable minimums
        if time_since_last.total_seconds() < Config.MIN_NEWS_AGE_MINUTES * 60:
            # Too soon since last run, use minimum gap
            cutoff_time = now - timedelta(minutes=Config.MIN_NEWS_AGE_MINUTES)
            log_debug(f"Using minimum gap: {Config.MIN_NEWS_AGE_MINUTES} minutes")
        elif time_since_last.total_seconds() > Config.MAX_NEWS_AGE_HOURS * 3600:
            # Too long since last run, cap at maximum
            cutoff_time = now - timedelta(hours=Config.MAX_NEWS_AGE_HOURS)
            log_debug(f"Capping at maximum lookback: {Config.MAX_NEWS_AGE_HOURS} hours")
        else:
            # Use actual time since last run, but enforce minimum 30-minute lookback
            if time_since_last.total_seconds() < 1800:  # Less than 30 minutes
                cutoff_time = now - timedelta(minutes=30)
                log_debug(f"Enforcing 30-minute minimum lookback instead of {time_since_last.total_seconds()/60:.1f} minutes")
            else:
                cutoff_time = last_run_time
                log_debug(f"Using dynamic lookback: {time_since_last.total_seconds() / 3600:.1f} hours")
    
        # Final sanity check
        if cutoff_time > now:
            log_error(f"⚠️ Cutoff time {cutoff_time} is in the future! Using 1 hour ago instead.")
            cutoff_time = now - timedelta(hours=1)
        
        # Convert to EST for user-friendly logging  
        try:
            cutoff_est = cutoff_time.astimezone(timezone(timedelta(hours=-5)))  # EST is UTC-5
            final_lookback_hours = (now - cutoff_time).total_seconds() / 3600
            log_info(f"Final cutoff time: {cutoff_time} UTC ({cutoff_est.strftime('%Y-%m-%d %H:%M:%S EST')}) - {final_lookback_hours:.1f}h lookback")
        except Exception as e:
            log_debug(f"Error with timezone conversion: {e}")
            final_lookback_hours = (now - cutoff_time).total_seconds() / 3600
            log_info(f"Final cutoff time: {cutoff_time} UTC - {final_lookback_hours:.1f}h lookback")
        
        return cutoff_time
    
    async def _process_cycle(self) -> None:
        """Process one complete analysis cycle with run tracking"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        # Start run tracking
        run_id = self.run_tracker.start_run()
        
        log_info(f"🔄 Starting analysis cycle #{self.cycle_count} (run_id: {run_id})")
        
        articles_fetched = 0
        articles_processed = 0
        decisions_made = 0
        
        try:
            # Get dynamic cutoff time
            cutoff_time = self._get_dynamic_news_cutoff_time()
            
            # Step 1: Fetch latest news with dynamic time filter
            log_info(f"📰 Step 1: Fetching news since {cutoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            all_articles = self.news_fetcher.fetch_all_news_since(cutoff_time)
            articles_fetched = len(all_articles)
            
            if not all_articles:
                log_info("No news articles found since last run, ending cycle")
                self.run_tracker.complete_run(run_id, articles_fetched, 0, 0)
                return
            
            # Step 2: Filter unprocessed articles
            log_info("🔍 Step 2: Filtering unprocessed articles...")
            unprocessed_articles = self.article_tracker.filter_unprocessed_articles(all_articles)
            
            if not unprocessed_articles:
                log_info("No new articles to process, ending cycle")
                self.run_tracker.complete_run(run_id, articles_fetched, 0, 0)
                return
            
            # Step 2.5: Apply article limit AFTER filtering processed articles
            original_count = len(unprocessed_articles)
            unprocessed_articles = unprocessed_articles[:Config.MAX_NEWS_ARTICLES]
            articles_processed = len(unprocessed_articles)
            
            if original_count > Config.MAX_NEWS_ARTICLES:
                log_info(f"Applied {Config.MAX_NEWS_ARTICLES} article limit: {original_count} -> {articles_processed} articles")
            
            # Step 3: Aggregate by ticker
            log_info("📊 Step 3: Aggregating articles by ticker...")
            ticker_buckets = self.ticker_aggregator.aggregate_by_ticker(unprocessed_articles)
            
            if not ticker_buckets:
                log_info("No valid ticker buckets created, ending cycle")
                self._mark_articles_processed(unprocessed_articles)
                self.run_tracker.complete_run(run_id, articles_fetched, articles_processed, 0)
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
            decisions_made = len(decisions)
            
            # Step 7: Log decisions
            if decisions:
                log_info(f"📝 Step 7: Logging {decisions_made} trading decisions...")
                logged_count = self.csv_logger.log_decisions_batch(decisions)
                log_info(f"✅ Successfully logged {logged_count} decisions")
                
                # Print summary of decisions
                self._print_decisions_summary(decisions)
            else:
                log_info("📝 Step 7: No high-confidence decisions to log")
            
            # Step 8: Mark articles as processed
            log_info("✅ Step 8: Marking articles as processed...")
            self._mark_articles_processed(all_articles, decisions)
            
            # Complete successful run
            self.run_tracker.complete_run(run_id, articles_fetched, articles_processed, decisions_made)
            
            # Cycle summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            log_info(f"🏁 Cycle #{self.cycle_count} completed in {cycle_duration:.1f} seconds")
            
            # Cleanup old data periodically
            if self.cycle_count % 10 == 0:
                self._periodic_cleanup()
                
        except Exception as e:
            log_error(f"Error in processing cycle: {e}")
            self.run_tracker.fail_run(run_id, str(e))
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
            deleted_articles = self.article_tracker.cleanup_old_articles(days=30)
            if deleted_articles > 0:
                log_info(f"   Cleaned up {deleted_articles} old articles")
            
            # Cleanup old run history
            deleted_runs = self.run_tracker.cleanup_old_runs(days=30)
            if deleted_runs > 0:
                log_info(f"   Cleaned up {deleted_runs} old run records")
            
            # Get and log statistics
            db_stats = self.article_tracker.get_statistics()
            csv_stats = self.csv_logger.get_csv_statistics()
            run_stats = self.run_tracker.get_run_statistics(days=7)
            
            log_info("📊 System Statistics:")
            log_info(f"   Database: {db_stats.get('total_articles', 0)} articles, {db_stats.get('unique_tickers', 0)} tickers")
            log_info(f"   CSV: {csv_stats.get('total_decisions', 0)} decisions logged")
            log_info(f"   Run success rate: {run_stats.get('success_rate', 0):.1f}% (last 7 days)")
            
        except Exception as e:
            log_error(f"Error in periodic cleanup: {e}")
    
    async def _shutdown(self) -> None:
        """Graceful shutdown"""
        log_info("🛑 Initiating graceful shutdown...")
        
        try:
            # Print final statistics
            db_stats = self.article_tracker.get_statistics()
            csv_stats = self.csv_logger.get_csv_statistics()
            run_stats = self.run_tracker.get_run_statistics(days=7)
            service_status = self.llm_analyzer.get_service_status()
            
            log_info("📊 FINAL STATISTICS:")
            log_info(f"   Cycles completed: {self.cycle_count}")
            log_info(f"   Articles processed: {db_stats.get('total_articles', 0)}")
            log_info(f"   Decisions logged: {csv_stats.get('total_decisions', 0)}")
            log_info(f"   Unique tickers analyzed: {db_stats.get('unique_tickers', 0)}")
            log_info(f"   Success rate (7 days): {run_stats.get('success_rate', 0):.1f}%")
            
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