"""
Enhanced Financial News Analyzer - Main Application
Integrates neural networks, earnings transcripts, and fundamental filtering for superior trading decisions
Python 3.13.3 compatible - FIXED VERSION
"""
import asyncio
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from config import Config
from database.article_tracker import ArticleTracker
from data_loaders.news_fetcher import NewsFetcher
from core.ticker_filter import TickerFilterEngine, FilterCriteria
from earnings_event_system import EarningsIntegrationManager
from analysis.multi_llm_analyzer import MultiLLMAnalyzer
from enhanced_decision_engine import EnhancedDecisionEngine
from analysis.price_tracker import PriceTracker, TrackingScheduler
from output.csv_logger import CSVLogger
from core.ticker_aggregator import TickerAggregator
from utils.simple_logger import log_info, log_error, log_debug, log_warning


class EnhancedFinancialNewsAnalyzer:
    """Enhanced main application class with neural networks and earnings analysis"""
    
    def __init__(self) -> None:
        """Initialize enhanced analyzer"""
        self.cycle_count = 0
        self.last_successful_run = None
        
        # Initialize components
        self._initialize_components()
        
        # Start background price tracking
        self._start_background_scheduler()
        
        # Log enhanced startup summary
        self._log_enhanced_startup_summary()
    
    def _initialize_components(self) -> None:
        """Initialize all system components including enhanced features"""
        log_info("🔧 Initializing enhanced system components...")
        
        # Core data components
        self.article_tracker = ArticleTracker()
        self.news_fetcher = NewsFetcher(Config.FMP_API_KEY)  # Pass API key for earnings transcripts
        self.ticker_aggregator = TickerAggregator()
        
         # Earnings integration
        if Config.ENABLE_EARNINGS_EVENTS:
            self.earnings_manager = EarningsIntegrationManager(self.news_fetcher)
            log_info("✅ Earnings event analysis initialized")
        else:
            self.earnings_manager = None
            log_info("❌ Earnings event analysis disabled")
            
        # Fundamental filtering (if enabled)
        if Config.ENABLE_FUNDAMENTAL_FILTERING:
            filter_criteria = FilterCriteria(
                min_price=Config.MIN_STOCK_PRICE,
                max_price=Config.MAX_STOCK_PRICE,
                min_avg_volume=Config.MIN_AVG_VOLUME,
                min_dollar_volume=Config.MIN_DOLLAR_VOLUME,
                min_market_cap=Config.MIN_MARKET_CAP,
                max_volatility_beta=Config.MAX_VOLATILITY_BETA,
                require_options=Config.REQUIRE_OPTIONS,
                allowed_exchanges=Config.ALLOWED_EXCHANGES
            )
            self.ticker_filter = TickerFilterEngine(self.news_fetcher, filter_criteria)
            self.ticker_filter.set_debug_mode(True)  # Enable detailed debugging
            log_info("✅ Fundamental filtering initialized")
        else:
            self.ticker_filter = None
            log_info("❌ Fundamental filtering disabled")
        
        # Enhanced analysis components
        log_info("🧠 Initializing AI analysis components...")
        self.llm_analyzer = MultiLLMAnalyzer()
        self.decision_engine = EnhancedDecisionEngine()
        
        # Enhanced price tracking components
        log_info("📈 Initializing price tracking components...")
        self.price_tracker = PriceTracker(self.news_fetcher)  # Reuse news_fetcher for API access
        self.csv_logger = CSVLogger()
        self.tracking_scheduler = TrackingScheduler(self.price_tracker, self.csv_logger)
        
        # Log enhanced component summary
        self._log_component_summary()
    
    def _log_component_summary(self) -> None:
        """Log summary of initialized enhanced components"""
        log_info("📋 Component initialization summary:")
        
        # Enhanced neural component status
        service_status = self.llm_analyzer.get_service_status()
        if 'enhanced_neural' in service_status:
            neural_info = service_status['enhanced_neural']
            if neural_info.get('available'):
                log_info(f"   🚀 Enhanced Neural: {neural_info.get('accuracy', 'N/A')} on {neural_info.get('device', 'CPU')}")
                if 'architecture' in neural_info:
                    log_info(f"      Architecture: {neural_info['architecture']}")
            else:
                log_info(f"   ❌ Enhanced Neural: Not available")
        
        # Traditional services count
        traditional_count = sum(1 for name, status in service_status.items() 
                              if name != 'enhanced_neural' and status.get('available'))
        traditional_services = [name for name, status in service_status.items() 
                              if name != 'enhanced_neural' and status.get('available')]
        log_info(f"   🤖 Traditional services: {traditional_count} available ({', '.join(traditional_services)})")
        
        # Earnings transcript capability
        if Config.ENABLE_EARNINGS_EVENTS:
            log_info(f"   🎙️ Earnings transcripts: Up to {Config.MAX_EARNINGS_EVENTS_PER_CYCLE} per cycle")
        else:
            log_info(f"   🎙️ Earnings transcripts: Disabled")
        
        # Fundamental filtering status
        if Config.ENABLE_FUNDAMENTAL_FILTERING:
            exchange_count = len(Config.ALLOWED_EXCHANGES)
            log_info(f"   🔍 Fundamental filtering: Active with {exchange_count} exchanges")
        else:
            log_info(f"   🔍 Fundamental filtering: Disabled")
        
        # Enhanced price tracking intervals
        checkpoint_info = Config.get_checkpoint_info()
        intervals_str = ", ".join([f"{cp['short_label']}" for cp in checkpoint_info])
        log_info(f"   📊 Price tracking: {intervals_str}")
        
        log_info("✅ All enhanced components initialized successfully")
    
    def _start_background_scheduler(self) -> None:
        """Start background price tracking scheduler"""
        if not hasattr(self, 'tracking_scheduler'):
            log_warning("Price tracking scheduler not initialized")
            return
            
        # Start scheduler in background
        asyncio.create_task(self.tracking_scheduler.run_scheduler())
        log_debug("📈 Background price tracking scheduler started")
    
    def _calculate_news_cutoff_time(self) -> datetime:
        """Calculate intelligent news cutoff time based on last successful run"""
        now = datetime.now(timezone.utc)
        
        # Use last successful run if available, otherwise use default lookback
        if self.last_successful_run:
            # Calculate hours since last run
            hours_since_last = (now - self.last_successful_run).total_seconds() / 3600
            
            if hours_since_last <= Config.DEFAULT_NEWS_LOOKBACK_HOURS:
                # Normal operation - use last run time
                cutoff_time = self.last_successful_run
                log_info(f"Using last successful run time: {cutoff_time}")
            elif hours_since_last <= Config.MAX_NEWS_AGE_HOURS:
                # Extended downtime but reasonable - use actual time since last run
                cutoff_time = self.last_successful_run
                log_info(f"Extended downtime ({hours_since_last:.1f}h). Using actual last run time.")
            else:
                # Been down too long - cap at maximum age
                cutoff_time = now - timedelta(hours=Config.MAX_NEWS_AGE_HOURS)
                log_warning(f"Long downtime detected ({hours_since_last:.1f}h). Capping lookback at {Config.MAX_NEWS_AGE_HOURS}h.")
        else:
            # First run - use default lookback
            cutoff_time = now - timedelta(hours=Config.DEFAULT_NEWS_LOOKBACK_HOURS)
            log_info(f"First run detected. Using default {Config.DEFAULT_NEWS_LOOKBACK_HOURS}h lookback.")
            
        # Additional safety check
        if cutoff_time > now:
            log_warning("Calculated cutoff time is in the future! Using 1 hour ago instead.")
            cutoff_time = now - timedelta(hours=1)
        
        final_lookback_hours = (now - cutoff_time).total_seconds() / 3600
        log_info(f"Final cutoff time: {cutoff_time} UTC - {final_lookback_hours:.1f}h lookback")
        
        return cutoff_time
    
    async def _process_cycle(self) -> None:
        """Process one complete enhanced analysis cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        log_info(f"🔄 Starting enhanced analysis cycle #{self.cycle_count}")
        
        articles_fetched = 0
        articles_processed = 0
        tickers_before_filtering = 0
        tickers_after_filtering = 0
        decisions_made = 0
        earnings_transcripts_processed = 0
        
        try:
            # Step 1: Fetch latest news (including earnings transcripts)
            log_info(f"📰 Step 1: Fetching enhanced news sources")
            all_articles = self.news_fetcher.fetch_all_news()
            # Add these 3 lines after: all_articles = self.news_fetcher.fetch_all_news()
            sources = {}
            for a in all_articles: sources[a.get('source', 'unknown')] = sources.get(a.get('source', 'unknown'), 0) + 1
            log_info(f"🐛 NEWS SOURCES: {sources}")
            
            articles_fetched = len(all_articles)
            
            # Count earnings transcript articles separately
            earnings_articles = [a for a in all_articles if 'transcript' in a.get('source', '')]
            earnings_transcripts_processed = len(earnings_articles)
            
            if not all_articles:
                log_info("No news articles found, ending cycle")
                return
            
            log_info(f"📊 Fetched {articles_fetched} total articles ({earnings_transcripts_processed} from earnings transcripts)")
            
            # Step 2: Filter unprocessed articles
            log_info("🔍 Step 2: Filtering unprocessed articles...")
            unprocessed_articles = self.article_tracker.filter_unprocessed_articles(all_articles)
            
            if not unprocessed_articles:
                log_info("No new articles to process, ending cycle")
                return
            
            # Step 2.5: Apply article limit AFTER filtering processed articles
            original_count = len(unprocessed_articles)
            unprocessed_articles = unprocessed_articles[:Config.MAX_NEWS_ARTICLES]
            articles_processed = len(unprocessed_articles)
            
            if original_count > Config.MAX_NEWS_ARTICLES:
                log_info(f"Article limit applied: processing {articles_processed}/{original_count} articles")
            
            # Step 3: Aggregate articles by ticker
            log_info("📊 Step 3: Aggregating articles by ticker...")
            ticker_buckets = self.ticker_aggregator.aggregate_by_ticker(unprocessed_articles)
            tickers_before_filtering = len(ticker_buckets)
            
            if not ticker_buckets:
                log_info("No valid tickers found in articles")
                return
            
            # Step 3.5: Apply fundamental filtering if enabled
            if Config.ENABLE_FUNDAMENTAL_FILTERING and self.ticker_filter:
                log_info("🔍 Step 3.5: Applying fundamental filtering...")
                ticker_buckets = self.ticker_filter.filter_ticker_buckets(ticker_buckets)
                tickers_after_filtering = len(ticker_buckets)
                
                if not ticker_buckets:
                    log_info("No tickers passed fundamental filtering")
                    return
            else:
                tickers_after_filtering = tickers_before_filtering
                log_info("🔍 Step 3.5: Fundamental filtering disabled")
            
            # Step 3.75: Analyze earnings events if enabled
            earnings_analyses = {}
            if Config.ENABLE_EARNINGS_EVENTS and self.earnings_manager:
                log_info("📅 Step 3.75: Analyzing earnings events...")
                earnings_analyses = self.earnings_manager.analyze_tickers_for_earnings(ticker_buckets)
            
            # Step 4: Prioritize tickers for analysis
            log_info("🎯 Step 4: Prioritizing tickers for enhanced analysis...")
            prioritized_tickers = self.ticker_aggregator.prioritize_tickers(ticker_buckets)
            
            # Step 5: Enhanced neural analysis of each ticker
            log_info(f"🧠 Step 5: Enhanced neural analysis of {len(prioritized_tickers)} tickers...")
            ticker_analyses = await self._analyze_tickers_enhanced(prioritized_tickers, ticker_buckets)
            
            # Step 6: Make enhanced trading decisions
            log_info("⚖️ Step 6: Making enhanced trading decisions with earnings integration...")
            decisions = self.decision_engine.batch_process_enhanced_decisions(ticker_analyses, earnings_analyses)
            decisions_made = len(decisions)
            
            # Step 6.5: Add entry prices before CSV logging - FIXED: Remove await
            if decisions:
                log_info("💰 Step 6.5: Adding entry prices to trading decisions...")
                self._add_entry_prices_to_decisions(decisions)
            
            # Step 7: Log decisions with enhanced metrics
            if decisions:
                log_info(f"📝 Step 7: Logging {decisions_made} enhanced trading decisions...")
                logged_count = self.csv_logger.log_decisions_batch(decisions)
                
                # Add price tracking for LONG/SHORT decisions
                tracking_added = 0
                for decision in decisions:
                    if decision.decision in ['LONG', 'SHORT']:
                        if hasattr(decision, 'recommendation_price') and decision.recommendation_price:
                            self.tracking_scheduler.add_tracking(decision)
                            tracking_added += 1
                        else:
                            log_warning(f"Skipping price tracking for {decision.ticker} - no entry price available")
                
                log_info(f"✅ Successfully logged {logged_count} decisions with enhanced analysis")
                log_info(f"📊 Added price tracking for {tracking_added} LONG/SHORT positions")
                
                # Print enhanced decision summary
                self._print_enhanced_decisions_summary(decisions)
            else:
                log_info("📝 Step 7: No high-confidence decisions to log")
                
                # Debug: Show what decisions were made (even low confidence ones)
                if decisions_made > 0:
                    log_debug(f"Made {decisions_made} total decisions but none met confidence threshold")
            
            # Step 8: Mark articles as processed
            log_info("✅ Step 8: Marking articles as processed...")
            self._mark_articles_processed(unprocessed_articles, decisions)
            
            # Record successful completion
            self.last_successful_run = datetime.now(timezone.utc)
            
            # Log enhanced cycle summary
            self._log_enhanced_cycle_summary(
                cycle_start, articles_fetched, earnings_transcripts_processed, articles_processed, 
                tickers_before_filtering, tickers_after_filtering, decisions_made
            )
            
        except Exception as e:
            log_error(f"Error in enhanced processing cycle: {e}")
            traceback.print_exc()
            raise
    
    async def _analyze_tickers_enhanced(self, prioritized_tickers: List[str], 
                                      ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze tickers using enhanced neural networks and traditional methods"""
        ticker_analyses = {}
        
        for i, ticker in enumerate(prioritized_tickers[:Config.MAX_TICKERS_TO_ANALYZE]):
            try:
                articles = ticker_buckets[ticker]
                log_debug(f"🧠 Analyzing {ticker} ({i+1}/{min(len(prioritized_tickers), Config.MAX_TICKERS_TO_ANALYZE)}) with {len(articles)} articles")
                
                # Enhanced multi-source analysis
                news_prediction = self.llm_analyzer.analyze_ticker_multi_source(ticker, articles)
                
                # Check if ticker has earnings data
                has_earnings_data = any('earnings' in article.get('source', '') for article in articles)
                earnings_articles = [a for a in articles if 'earnings' in a.get('source', '')]
                
                # Store analysis results
                ticker_analyses[ticker] = {
                    'ticker': ticker,
                    'articles': articles,
                    'news_prediction': news_prediction,
                    'has_earnings_data': has_earnings_data,
                    'earnings_articles': earnings_articles,
                    'article_count': len(articles)
                }
                
                # Enhanced logging for high-value analysis
                if news_prediction and hasattr(news_prediction, 'source'):
                    if 'enhanced_neural' in news_prediction.source:
                        log_debug(f"✨ {ticker}: Enhanced neural analysis with {news_prediction.confidence:.3f} confidence")
                    if has_earnings_data:
                        log_debug(f"🎙️ {ticker}: Includes {len(earnings_articles)} earnings transcript articles")
                
                # Small delay to be respectful to APIs
                await asyncio.sleep(0.3)  # Reduced delay since enhanced neural is local
                
            except Exception as e:
                log_error(f"Error in enhanced analysis for {ticker}: {e}")
                continue
        
        return ticker_analyses
    
    def _add_entry_prices_to_decisions(self, decisions: List) -> None:
        """Add current market prices to all trading decisions as entry prices - FIXED: Made synchronous"""
        prices_added = 0
        prices_failed = 0
        
        for decision in decisions:
            try:
                # FIXED: Remove await since get_current_price is synchronous
                current_price = self.price_tracker.get_current_price(decision.ticker)
                if current_price:
                    decision.recommendation_price = current_price
                    decision.recommendation_timestamp = datetime.now()
                    prices_added += 1
                    log_debug(f"💰 Added entry price for {decision.ticker}: ${current_price:.2f}")
                else:
                    log_warning(f"Could not get entry price for {decision.ticker}")
                    prices_failed += 1
            except Exception as e:
                log_error(f"Error getting entry price for {decision.ticker}: {e}")
                prices_failed += 1
                
        log_info(f"💰 Entry price capture: {prices_added} successful, {prices_failed} failed")
        
        if prices_failed > 0:
            log_warning(f"⚠️ {prices_failed} decisions will be logged without entry prices")
    
    def _mark_articles_processed(self, articles: List[Dict[str, Any]], 
                                decisions: List = None) -> None:
        """Mark articles as processed in database"""
        try:
            decisions_by_ticker = {}
            if decisions:
                for decision in decisions:
                    decisions_by_ticker[decision.ticker] = decision
            
            successful_marks = 0
            failed_marks = 0
            
            for article in articles:
                try:
                    # Create unique article identifier
                    article_id = self.article_tracker._create_article_id(article)
                    
                    # Determine decision info
                    ticker = article.get('symbol', 'UNKNOWN')
                    decision_info = decisions_by_ticker.get(ticker)
                    
                    decision_type = decision_info.decision if decision_info else 'NONE'
                    confidence = decision_info.confidence if decision_info else 0.0
                    
                    # Mark as processed
                    self.article_tracker.mark_article_processed(
                        article_id=article_id,
                        ticker=ticker,
                        decision=decision_type,
                        confidence=confidence,
                        source=article.get('source', 'unknown')
                    )
                    successful_marks += 1
                    
                except Exception as e:
                    log_error(f"Failed to mark article as processed: {e}")
                    failed_marks += 1
            
            log_debug(f"Article processing: {successful_marks} marked, {failed_marks} failed")
            
        except Exception as e:
            log_error(f"Error in batch article marking: {e}")
    
    def _print_enhanced_decisions_summary(self, decisions: List) -> None:
        """Print enhanced summary of trading decisions"""
        if not decisions:
            return
        
        # Count decisions by type and source
        decision_counts = {'LONG': 0, 'SHORT': 0, 'NONE': 0}
        source_counts = {}
        confidence_sum = 0
        enhanced_neural_count = 0
        earnings_boost_count = 0
        
        for decision in decisions:
            decision_counts[decision.decision] += 1
            confidence_sum += decision.confidence
            
            # Track analysis source
            if hasattr(decision, 'analysis_method'):
                source = decision.analysis_method
                source_counts[source] = source_counts.get(source, 0) + 1
                
                if 'enhanced_neural' in source:
                    enhanced_neural_count += 1
            
            # Track earnings boost
            if hasattr(decision, 'has_earnings_data') and decision.has_earnings_data:
                earnings_boost_count += 1
        
        avg_confidence = confidence_sum / len(decisions)
        
        log_info("📊 Enhanced decision summary:")
        log_info(f"   📈 LONG: {decision_counts['LONG']}, 📉 SHORT: {decision_counts['SHORT']}, ⚖️ NONE: {decision_counts['NONE']}")
        log_info(f"   🎯 Average confidence: {avg_confidence:.3f}")
        log_info(f"   🚀 Enhanced neural decisions: {enhanced_neural_count}")
        log_info(f"   🎙️ With earnings data: {earnings_boost_count}")
        
        # Show top performing decisions
        high_confidence = [d for d in decisions if d.confidence >= 0.8]
        if high_confidence:
            log_info(f"   ⭐ High confidence (≥0.8): {len(high_confidence)} decisions")
            for decision in sorted(high_confidence, key=lambda x: x.confidence, reverse=True)[:3]:
                log_info(f"      {decision.ticker}: {decision.decision} ({decision.confidence:.3f})")
    
    def _log_enhanced_cycle_summary(self, cycle_start: datetime, articles_fetched: int,
                                  earnings_transcripts_processed: int, articles_processed: int,
                                  tickers_before_filtering: int, tickers_after_filtering: int, 
                                  decisions_made: int) -> None:
        """Log enhanced cycle completion summary"""
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        log_info("🏁 Enhanced Cycle Summary:")
        log_info(f"   📰 Articles fetched: {articles_fetched}")
        log_info(f"   🎙️ Earnings transcripts: {earnings_transcripts_processed}")
        log_info(f"   📄 Articles processed: {articles_processed}")
        log_info(f"   📊 Tickers before filtering: {tickers_before_filtering}")
        log_info(f"   🔍 Tickers after filtering: {tickers_after_filtering}")
        log_info(f"   ⚖️ Trading decisions made: {decisions_made}")
        log_info(f"   ⏱️ Cycle duration: {cycle_duration:.1f} seconds")
        
        # Show filtering and enhancement efficiency
        if Config.ENABLE_FUNDAMENTAL_FILTERING and tickers_before_filtering > 0:
            filter_efficiency = (tickers_before_filtering - tickers_after_filtering) / tickers_before_filtering * 100
            log_info(f"   🔍 Filter efficiency: {filter_efficiency:.1f}% of tickers filtered out")
        
        if earnings_transcripts_processed > 0:
            transcript_ratio = (earnings_transcripts_processed / articles_fetched) * 100
            log_info(f"   🎙️ Transcript coverage: {transcript_ratio:.1f}% of articles from earnings calls")
        
        # Performance metrics
        if articles_processed > 0:
            processing_speed = articles_processed / cycle_duration
            log_info(f"   ⚡ Processing speed: {processing_speed:.1f} articles/second")
    
    async def _shutdown(self) -> None:
        """Graceful shutdown with enhanced cleanup"""
        log_info("🔄 Shutting down enhanced analyzer...")
        
        # Enhanced cleanup for neural models
        if hasattr(self, 'llm_analyzer') and self.llm_analyzer.enhanced_neural:
            log_info("🧠 Cleaning up enhanced neural analyzer...")
            try:
                # Clear GPU memory if using CUDA
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log_debug("✅ CUDA memory cache cleared")
            except:
                pass  # Fail silently if torch not available
        
        # Standard cleanup
        if hasattr(self, 'article_tracker'):
            pass  # Article tracker uses context managers, no explicit cleanup needed
        
        log_info("✅ Enhanced shutdown complete")
    
    async def run(self) -> None:
        """Main application loop with enhanced features"""
        log_info("🚀 Starting Enhanced Financial News Analyzer...")
        
        try:
            # Load last successful run time if available
            if hasattr(self.article_tracker, 'get_last_successful_run'):
                self.last_successful_run = self.article_tracker.get_last_successful_run()
        except Exception as e:
            log_debug(f"Could not load last successful run time: {e}")
            # Keep the initialization value from __init__
            
            while True:
                try:
                    await self._process_cycle()
                    log_info(f"✅ Enhanced cycle completed, waiting {Config.CYCLE_INTERVAL_MINUTES} minutes before next run...")
                    await asyncio.sleep(Config.CYCLE_INTERVAL_MINUTES * 60)
                    
                except KeyboardInterrupt:
                    log_info("🛑 Enhanced analyzer interrupted by user")
                    break
                except Exception as e:
                    log_error(f"💥 Enhanced cycle failed: {e}")
                    traceback.print_exc()
                    log_info(f"⏳ Waiting {Config.CYCLE_INTERVAL_MINUTES} minutes before retry...")
                    await asyncio.sleep(Config.CYCLE_INTERVAL_MINUTES * 60)
                        
    def _log_enhanced_startup_summary(self) -> None:
        """Log enhanced startup configuration summary"""
        log_info("🚀 Enhanced Financial News Analyzer with neural networks and earnings analysis initialized successfully  ")
        
        
def _log_enhanced_startup_banner() -> None:
    """Log enhanced startup banner with configuration details"""
    log_info("🚀 Starting Enhanced Financial News Analyzer...")
    log_info("=" * 90)
    log_info("🚀 ENHANCED FINANCIAL NEWS ANALYSIS SYSTEM")
    log_info("   RoBERTa+LSTM/CNN Neural Networks + Earnings Transcript Analysis")
    log_info("=" * 90)
    
    # Enhanced configuration section
    log_info("🔑 Enhanced Configuration:")
    log_info(f"   Min confidence threshold: {Config.MIN_CONFIDENCE_THRESHOLD}")
    log_info(f"   Max tickers to analyze: {Config.MAX_TICKERS_TO_ANALYZE}")
    
    # Calculate and show last run info
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=Config.DEFAULT_NEWS_LOOKBACK_HOURS)
    
    # Try to get actual last run from database
    try:
        from database.article_tracker import ArticleTracker
        tracker = ArticleTracker()
        last_run = tracker.get_last_successful_run()
        if last_run:
            log_info(f"   Last successful run: {last_run} UTC")
            cutoff_time = last_run
        else:
            log_info(f"   Last successful run: First run detected")
    except:
        log_info(f"   Last successful run: Could not determine")
    
    lookback_hours = (datetime.now(timezone.utc) - cutoff_time).total_seconds() / 3600
    log_info(f"Final cutoff time: {cutoff_time} UTC - {lookback_hours:.1f}h lookback")
    log_info(f"   Actual lookback: {lookback_hours:.1f} hours")
    log_info(f"   Max articles per cycle: {Config.MAX_NEWS_ARTICLES}")
    log_info(f"   Output file: {Config.CSV_OUTPUT_PATH}")
    log_info(f"   Database: {Config.SQLITE_DB_PATH}")
    
    # Enhanced AI Analysis section
    log_info("🧠 Enhanced AI Analysis:")
    
    # Import here to avoid circular imports and check neural status
    try:
        from analysis.multi_llm_analyzer import MultiLLMAnalyzer
        analyzer = MultiLLMAnalyzer()
        service_status = analyzer.get_service_status()
        
        # Enhanced neural status
        if 'enhanced_neural' in service_status:
            neural_info = service_status['enhanced_neural']
            if neural_info.get('available'):
                log_info(f"   🚀 Enhanced Neural: {neural_info.get('accuracy', 'N/A')} on {neural_info.get('device', 'CPU')}")
                log_info(f"      Parameters: {neural_info.get('parameters', 'N/A'):,}")
            else:
                log_info(f"   ❌ Enhanced Neural: Not available")
        
        # Traditional services
        traditional_services = [name for name, status in service_status.items() 
                              if name != 'enhanced_neural' and status.get('available')]
        log_info(f"   🤖 Traditional services: {', '.join(traditional_services)}")
        
        # Earnings transcript status
        if Config.ENABLE_EARNINGS_EVENTS:
            log_info(f"📅 Earnings Event Analysis:")
            log_info(f"   Event detection window: {Config.EARNINGS_LOOKBACK_DAYS} days back, {Config.EARNINGS_LOOKAHEAD_DAYS} days ahead")
            log_info(f"   Min confidence threshold: {Config.MIN_EARNINGS_CONFIDENCE}")
            log_info(f"   3-way scoring weights: news={Config.NEWS_WEIGHT_3WAY}, earnings={Config.EARNINGS_WEIGHT_3WAY}, tech={Config.TECHNICAL_WEIGHT_3WAY}")
        else:
            log_info(f"📅 Earnings Event Analysis: DISABLED")
        
        # Fundamental filtering status
        if Config.ENABLE_FUNDAMENTAL_FILTERING:
            log_info(f"🔍 Fundamental Filtering: ENABLED")
            log_info(f"   Price range: ${Config.MIN_STOCK_PRICE:.2f} - ${Config.MAX_STOCK_PRICE:.2f}")
            log_info(f"   Min market cap: ${Config.MIN_MARKET_CAP:,}")
            log_info(f"   Allowed exchanges: {', '.join(Config.ALLOWED_EXCHANGES)}")
        else:
            log_info(f"🔍 Fundamental Filtering: DISABLED")
        
        # Enhanced price tracking intervals
        checkpoint_info = Config.get_checkpoint_info()
        intervals_str = ", ".join([cp['short_label'] for cp in checkpoint_info])
        log_info(f"📈 Enhanced Price Tracking: {intervals_str}")
        
        log_info("=" * 90)
        
    except ImportError as e:
        log_error(f"Could not import analysis components for startup summary: {e}")
    except Exception as e:
        log_error(f"Error generating startup summary: {e}")


async def main():
    """Enhanced main entry point"""
    try:
        _log_enhanced_startup_banner()
        analyzer = EnhancedFinancialNewsAnalyzer()
        await analyzer.run()
    except KeyboardInterrupt:
        log_info("🛑 Enhanced application terminated by user")
    except Exception as e:
        log_error(f"💥 Enhanced application failed: {e}")
        traceback.print_exc()  # ADD THIS LINE
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
