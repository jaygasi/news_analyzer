"""
Enhanced financial news analysis system with RoBERTa+LSTM/CNN hybrid neural analyzer and earnings transcript analysis
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
from data_loaders.news_fetcher import NewsFetcher
from core.ticker_aggregator import TickerAggregator
from core.ticker_filter import TickerFilterEngine, FilterCriteria
from analysis.multi_llm_analyzer import MultiLLMAnalyzer
from analysis.technical_analyzer_simple import TechnicalAnalyzer
from analysis.price_tracker import PriceTracker, TrackingScheduler
from core.decision_engine import DecisionEngine
from output.csv_logger import CSVLogger
import traceback
from earnings_event_system import EarningsIntegrationManager
from enhanced_decision_engine import EnhancedDecisionEngine, EnhancedTradingDecision


class EnhancedFinancialNewsAnalyzer:
    """Enhanced financial news analyzer with neural networks and earnings transcript analysis"""
    
    def __init__(self) -> None:
        """Initialize the enhanced financial news analyzer"""
        self.running = True
        self.cycle_count = 0
        
        # Initialize last run tracking (simple in-memory for now)
        self.last_successful_run = datetime.now(timezone.utc) - timedelta(hours=Config.DEFAULT_NEWS_AGE_HOURS)
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        log_info("🚀 Enhanced Financial News Analyzer with neural networks and earnings analysis initialized successfully")
    
    def _validate_configuration(self) -> None:
        """Validate required configuration with enhanced feature status"""
        api_keys = Config.validate_api_keys()
        
        if not api_keys['fmp']:
            raise ValueError("FMP_API_KEY is required")
        
        # Log available services
        available_services = [key for key, available in api_keys.items() if available]
        log_info(f"Available API services: {', '.join(available_services)}")
        
        # Show detailed service toggle status including enhanced features
        log_info("🔧 Service toggle status:")
        toggles = {
            'enhanced_neural': Config.ENABLE_ENHANCED_NEURAL,
            'finbert': Config.ENABLE_FINBERT,
            'gemini': Config.ENABLE_GEMINI,
            'openai': Config.ENABLE_OPENAI,
            'claude': Config.ENABLE_CLAUDE,
            'alpha_vantage': Config.ENABLE_ALPHA_VANTAGE,
            'polygon': Config.ENABLE_POLYGON,
            'tiingo': Config.ENABLE_TIINGO,
            'keyword_sentiment': Config.ENABLE_KEYWORD_SENTIMENT,
            'fundamental_filtering': Config.ENABLE_FUNDAMENTAL_FILTERING,
            'earnings_events': Config.ENABLE_EARNINGS_EVENTS
        }
        
        for service, enabled in toggles.items():
            status = "✅ ENABLED" if enabled else "❌ DISABLED"
            log_info(f"   {service}: {status}")
        
        # Validate enhanced neural dependencies
        if Config.ENABLE_ENHANCED_NEURAL:
            if Config.has_enhanced_neural_dependencies():
                log_info("✅ Enhanced Neural dependencies satisfied")
            else:
                log_warning("⚠️ Enhanced Neural enabled but dependencies missing (torch, transformers, numpy)")
                log_warning("   Install with: pip install torch transformers numpy")
        
        # Validate earnings transcript configuration
        if Config.ENABLE_EARNINGS_EVENTS:
            log_info(f"📅 Earnings event settings:")
            log_info(f"   Lookback days: {Config.EARNINGS_LOOKBACK_DAYS}")
            log_info(f"   Lookahead days: {Config.EARNINGS_LOOKAHEAD_DAYS}")
            log_info(f"   Min confidence: {Config.MIN_EARNINGS_CONFIDENCE}")
            log_info(f"   Cache hours: {Config.EARNINGS_CACHE_HOURS}")
        else:
            log_info(f"📅 Earnings event analysis: DISABLED")
                
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
        
        # Multi-LLM analyzer (now includes enhanced neural)
        self.llm_analyzer = MultiLLMAnalyzer()
        
        # Technical analyzer
        self.technical_analyzer = TechnicalAnalyzer(self.news_fetcher)
        
        # Decision engine
        #self.decision_engine = DecisionEngine()
        self.decision_engine = EnhancedDecisionEngine()
        
        # Price tracking components
        log_info("📈 Initializing price tracking components...")
        self.price_tracker = PriceTracker(self.news_fetcher)
        
        # Output component
        self.csv_logger = CSVLogger()
        
        self.tracking_scheduler = TrackingScheduler(self.price_tracker, self.csv_logger)
        
        
        
        # Log component initialization summary
        self._log_initialization_summary()
        
        log_info("✅ All enhanced components initialized successfully")
    
    def _log_initialization_summary(self) -> None:
        """Log summary of initialized components and their capabilities"""
        
        # Get service status from multi-LLM analyzer
        service_status = self.llm_analyzer.get_service_status()
        
        log_info("📋 Component initialization summary:")
        
        # Neural analysis capabilities
        if 'enhanced_neural' in service_status:
            neural_status = service_status['enhanced_neural']
            if neural_status.get('available'):
                log_info(f"   🚀 Enhanced Neural: {neural_status.get('accuracy', 'N/A')} accuracy on {neural_status.get('device', 'CPU')}")
                log_info(f"      Architecture: {neural_status.get('type', 'RoBERTa+LSTM+CNN')}")
            else:
                log_info("   ❌ Enhanced Neural: Not available")
        
        # Traditional analysis services
        traditional_services = ['finbert', 'gemini', 'openai', 'claude', 'alpha_vantage', 'polygon', 'tiingo', 'keyword']
        available_traditional = [s for s in traditional_services if service_status.get(s, {}).get('available', False)]
        log_info(f"   🤖 Traditional services: {len(available_traditional)} available ({', '.join(available_traditional)})")
        
        # Earnings transcript capability
        if Config.ENABLE_EARNINGS_EVENTS:
            log_info(f"   🎙️ Earnings transcripts: Up to per cycle")
        else:
            log_info("   ❌ Earnings transcripts: Disabled")
        
        # Fundamental filtering
        if self.ticker_filter:
            log_info(f"   🔍 Fundamental filtering: Active with {len(Config.ALLOWED_EXCHANGES)} exchanges")
        else:
            log_info("   ❌ Fundamental filtering: Disabled")
        
        # Price tracking
        checkpoint_info = Config.get_checkpoint_info()
        intervals = ", ".join([cp['short_label'] for cp in checkpoint_info])
        log_info(f"   📊 Price tracking: {intervals}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            log_info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _get_dynamic_news_cutoff_time(self) -> datetime:
        """Calculate dynamic cutoff time based on last successful run with intelligent gap handling"""
        now = datetime.now(timezone.utc)
        
        # Calculate time since last run
        time_since_last = now - self.last_successful_run
        hours_since_last = time_since_last.total_seconds() / 3600
        
        log_debug(f"Time since last successful run: {hours_since_last:.1f} hours")
        
        # Apply intelligent logic
        if hours_since_last < (Config.MIN_NEWS_AGE_MINUTES / 60):
            # Too recent - use minimum gap
            cutoff_time = now - timedelta(minutes=Config.MIN_NEWS_AGE_MINUTES)
            log_debug(f"Very recent run detected. Using minimum gap of {Config.MIN_NEWS_AGE_MINUTES} minutes.")
        elif hours_since_last <= Config.MAX_NEWS_AGE_HOURS:
            # Normal case - use actual last run time
            cutoff_time = self.last_successful_run
            log_debug(f"Normal case: Using last successful run time as cutoff.")
        else:
            # Been down too long - cap at maximum age
            cutoff_time = now - timedelta(hours=Config.MAX_NEWS_AGE_HOURS)
            log_warning(f"Long downtime detected ({hours_since_last:.1f}h). Capping lookback at {Config.MAX_NEWS_AGE_HOURS}h.")
            
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
        EARNINGS_EVENTS_processed = 0
        
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
            EARNINGS_EVENTS_processed = len(earnings_articles)
            
            if not all_articles:
                log_info("No news articles found, ending cycle")
                return
            
            log_info(f"📊 Fetched {articles_fetched} total articles ({EARNINGS_EVENTS_processed} from earnings transcripts)")
            
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
                log_info(f"Applied {Config.MAX_NEWS_ARTICLES} article limit: {original_count} -> {articles_processed} articles")
            
            # Step 3: Aggregate by ticker
            log_info("📊 Step 3: Aggregating articles by ticker...")
            ticker_buckets = self.ticker_aggregator.aggregate_by_ticker(unprocessed_articles)
            
            if not ticker_buckets:
                log_info("No valid ticker buckets created, ending cycle")
                self._mark_articles_processed(unprocessed_articles)
                return
            
            tickers_before_filtering = len(ticker_buckets)
            
            # Step 3.5: Apply fundamental filtering (if enabled)
            if Config.ENABLE_FUNDAMENTAL_FILTERING and self.ticker_filter:
                log_info("🔍 Step 3.5: Applying fundamental filtering...")
                ticker_buckets = self.ticker_filter.filter_ticker_buckets(ticker_buckets)
                
                if not ticker_buckets:
                    log_warning("⚠️ No tickers passed fundamental filtering, ending cycle")
                    self._mark_articles_processed(unprocessed_articles)
                    return
            else:
                log_info("⏭️ Step 3.5: Fundamental filtering disabled, skipping...")
            
            tickers_after_filtering = len(ticker_buckets)
            
             # NEW: Step 3.75: Analyze earnings events
            log_info("📅 Step 3.75: Analyzing earnings events...")
            earnings_analyses = {}
            earnings_analyses_count = 0
            
            if Config.ENABLE_EARNINGS_EVENTS and self.earnings_manager:
                earnings_analyses = self.earnings_manager.analyze_tickers_for_earnings(ticker_buckets)
                earnings_analyses_count = len(earnings_analyses)
                
                if earnings_analyses_count > 0:
                    log_info(f"📅 Found {earnings_analyses_count} tickers with earnings events")
                    for ticker, analysis in earnings_analyses.items():
                        log_info(f"  📊 {ticker}: {analysis.direction} (score: {analysis.overall_score:+.2f}, "
                                f"conf: {analysis.confidence:.2f}) - {analysis.get_reasoning()}")
                else:
                    log_info("📅 No tickers with upcoming/recent earnings events found")
            else:
                log_info("📅 Earnings event analysis disabled")
            
            # Step 4: Prioritize tickers
            log_info("🎯 Step 4: Prioritizing tickers for enhanced analysis...")
            prioritized_tickers = self.ticker_aggregator.prioritize_tickers(ticker_buckets)
            
            # Step 5: Enhanced neural analysis of each ticker
            log_info(f"🧠 Step 5: Enhanced neural analysis of {len(prioritized_tickers)} tickers...")
            ticker_analyses = await self._analyze_tickers_enhanced(prioritized_tickers, ticker_buckets)
            
            # Step 6: Make enhanced trading decisions
            log_info("⚖️ Step 6: Making enhanced trading decisions with earnings integration...")
            decisions = self.decision_engine.batch_process_enhanced_decisions(ticker_analyses, earnings_analyses)
            decisions_made = len(decisions)
            
            # Step 6.5: Add entry prices before CSV logging
            if decisions:
                log_info("💰 Step 6.5: Adding entry prices to trading decisions...")
                await self._add_entry_prices_to_decisions(decisions)
            
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
                    log_debug(f"Made {decisions_made} total decisions, but none met confidence threshold of {Config.MIN_CONFIDENCE_THRESHOLD}")
            
            # Step 8: Mark articles as processed
            log_info("✅ Step 8: Marking articles as processed...")
            self._mark_articles_processed(unprocessed_articles, decisions)
            
            # Update last successful run time
            self.last_successful_run = datetime.now(timezone.utc)
            
            # Enhanced cycle summary
            self._print_enhanced_cycle_summary(
                cycle_start, articles_fetched, articles_processed, 
                tickers_before_filtering, tickers_after_filtering, 
                decisions_made, EARNINGS_EVENTS_processed
            )
            
        except Exception as e:
            log_error(f"Error in enhanced analysis cycle: {e}")
            raise
    
    async def _analyze_tickers_enhanced(self, prioritized_tickers: List[str], 
                                       ticker_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Enhanced ticker analysis using neural networks and comprehensive data"""
        ticker_analyses = {}
        
        # Use configurable limit
        max_tickers = Config.MAX_TICKERS_TO_ANALYZE
        tickers_to_analyze = prioritized_tickers[:max_tickers]
        
        for i, ticker in enumerate(tickers_to_analyze, 1):
            try:
                log_debug(f"🧠 Enhanced analysis {ticker} ({i}/{len(tickers_to_analyze)})")
                
                articles = ticker_buckets[ticker]
                
                # Enhanced news analysis (includes neural networks)
                news_prediction = self.llm_analyzer.analyze_news_direction(ticker, articles)
                
                # Technical analysis
                technical_signal = self.technical_analyzer.analyze_ticker(ticker)
                
                # Check if this ticker has earnings transcript data
                earnings_articles = [a for a in articles if 'transcript' in a.get('source', '')]
                has_earnings_data = len(earnings_articles) > 0
                
                ticker_analyses[ticker] = {
                    'news_prediction': news_prediction,
                    'technical_signal': technical_signal,
                    'article_count': len(articles),
                    'has_earnings_data': has_earnings_data,
                    'earnings_article_count': len(earnings_articles)
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
    
    async def _add_entry_prices_to_decisions(self, decisions: List) -> None:
        """Add current market prices to all trading decisions as entry prices"""
        prices_added = 0
        prices_failed = 0
        
        for decision in decisions:
            try:
                current_price = await self.price_tracker.get_current_price(decision.ticker)
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
                    ticker = article.get('symbol', '')
                    decision = decisions_by_ticker.get(ticker)
                    
                    decision_str = decision.decision if decision else ''
                    confidence = decision.confidence if decision else 0.0
                    
                    # CORRECT METHOD CALL: mark_article_processed(article, decision, confidence)
                    success = self.article_tracker.mark_article_processed(
                        article=article,  # Pass the full article dict
                        decision=decision_str,
                        confidence=confidence
                    )
                    
                    if success:
                        successful_marks += 1
                    else:
                        failed_marks += 1
                        
                except Exception as e:
                    failed_marks += 1
                    log_debug(f"Error marking individual article for {ticker}: {e}")
            
            log_debug(f"Marked articles as processed: {successful_marks} successful, {failed_marks} failed")
            
        except Exception as e:
            log_error(f"Error marking articles as processed: {e}")
    
    def _print_enhanced_decisions_summary(self, decisions: List) -> None:
        """Print enhanced summary of decisions including neural analysis insights"""
        if not decisions:
            return
        
        # Filter for high-confidence decisions
        high_conf_decisions = [d for d in decisions if d.confidence >= Config.MIN_CONFIDENCE_THRESHOLD]
        
        if not high_conf_decisions:
            return
        
        log_info("📊 Enhanced trading decisions summary:")
        
        # Analyze decision sources
        neural_decisions = []
        traditional_decisions = []
        
        for decision in high_conf_decisions:
            if hasattr(decision, 'news_prediction') and decision.news_prediction:
                if 'enhanced_neural' in str(decision.news_prediction.source):
                    neural_decisions.append(decision)
                else:
                    traditional_decisions.append(decision)
        
        # Log enhanced neural performance
        if neural_decisions:
            avg_neural_confidence = sum(d.confidence for d in neural_decisions) / len(neural_decisions)
            log_info(f"   🚀 Enhanced neural decisions: {len(neural_decisions)} (avg confidence: {avg_neural_confidence:.3f})")
        
        if traditional_decisions:
            avg_traditional_confidence = sum(d.confidence for d in traditional_decisions) / len(traditional_decisions)
            log_info(f"   🤖 Traditional model decisions: {len(traditional_decisions)} (avg confidence: {avg_traditional_confidence:.3f})")
        
        # Group by decision type
        long_decisions = [d for d in high_conf_decisions if d.decision == 'LONG']
        short_decisions = [d for d in high_conf_decisions if d.decision == 'SHORT']
        
        if long_decisions:
            log_info(f"   📈 LONG positions ({len(long_decisions)}):")
            for decision in sorted(long_decisions, key=lambda x: x.confidence, reverse=True)[:5]:
                price_info = f" @ ${decision.recommendation_price:.2f}" if hasattr(decision, 'recommendation_price') and decision.recommendation_price else ""
                neural_flag = "🚀" if decision in neural_decisions else "🤖"
                log_info(f"      {neural_flag} {decision.ticker}: {decision.confidence:.3f}{price_info}")
        
        if short_decisions:
            log_info(f"   📉 SHORT positions ({len(short_decisions)}):")
            for decision in sorted(short_decisions, key=lambda x: x.confidence, reverse=True)[:5]:
                price_info = f" @ ${decision.recommendation_price:.2f}" if hasattr(decision, 'recommendation_price') and decision.recommendation_price else ""
                neural_flag = "🚀" if decision in neural_decisions else "🤖"
                log_info(f"      {neural_flag} {decision.ticker}: {decision.confidence:.3f}{price_info}")
        
        # Show price tracking info with enhanced features
        tracking_decisions = [d for d in high_conf_decisions if d.decision in ['LONG', 'SHORT']]
        if tracking_decisions:
            checkpoint_info = Config.get_checkpoint_info()
            intervals_str = ", ".join([cp['short_label'] for cp in checkpoint_info])
            log_info(f"   📊 Enhanced price tracking: {len(tracking_decisions)} positions at {intervals_str}")
    
    def _print_enhanced_cycle_summary(self, cycle_start: datetime, articles_fetched: int, 
                                    articles_processed: int, tickers_before_filtering: int, 
                                    tickers_after_filtering: int, decisions_made: int, 
                                    EARNINGS_EVENTS_processed: int) -> None:
        """Print comprehensive cycle summary with enhanced metrics"""
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        log_info("🏁 Enhanced Cycle Summary:")
        log_info(f"   📰 Articles fetched: {articles_fetched}")
        log_info(f"   🎙️ Earnings transcripts: {EARNINGS_EVENTS_processed}")
        log_info(f"   📄 Articles processed: {articles_processed}")
        log_info(f"   📊 Tickers before filtering: {tickers_before_filtering}")
        log_info(f"   🔍 Tickers after filtering: {tickers_after_filtering}")
        log_info(f"   ⚖️ Trading decisions made: {decisions_made}")
        log_info(f"   ⏱️ Cycle duration: {cycle_duration:.1f} seconds")
        
        # Show filtering and enhancement efficiency
        if Config.ENABLE_FUNDAMENTAL_FILTERING and tickers_before_filtering > 0:
            filter_efficiency = (tickers_before_filtering - tickers_after_filtering) / tickers_before_filtering * 100
            log_info(f"   🔍 Filter efficiency: {filter_efficiency:.1f}% of tickers filtered out")
        
        if EARNINGS_EVENTS_processed > 0:
            transcript_ratio = (EARNINGS_EVENTS_processed / articles_fetched) * 100
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
        
        # Print enhanced startup information
        self._print_enhanced_startup_info()
        
        # Start price tracking scheduler as background task
        tracking_task = asyncio.create_task(self.tracking_scheduler.run_scheduler())
        
        try:
            while self.running:
                await self._process_cycle()
                
                if self.running:
                    log_info("✅ Enhanced cycle completed, waiting 5 minutes before next run...")
                    await asyncio.sleep(300)  # Wait 5 minutes between cycles
                    
        except KeyboardInterrupt:
            log_info("⚠️ Interrupted by user")
        except Exception as e:
            log_error(f"💥 Fatal error in enhanced main loop: {e}")
            raise
        finally:
            # Stop price tracking scheduler
            self.tracking_scheduler.running = False
            tracking_task.cancel()
            
            await self._shutdown()
    
    def _print_enhanced_startup_info(self) -> None:
        """Print enhanced startup information with neural analysis and earnings transcript details"""
        # Get last run info for dynamic time display
        time_since_last = datetime.now(timezone.utc) - self.last_successful_run
        
        log_info("=" * 90)
        log_info("🚀 ENHANCED FINANCIAL NEWS ANALYSIS SYSTEM")
        log_info("   RoBERTa+LSTM/CNN Neural Networks + Earnings Transcript Analysis")
        log_info("=" * 90)
        
        log_info(f"🔑 Enhanced Configuration:")
        log_info(f"   Min confidence threshold: {Config.MIN_CONFIDENCE_THRESHOLD}")
        log_info(f"   Max tickers to analyze: {Config.MAX_TICKERS_TO_ANALYZE}")
        log_info(f"   Last successful run: {self.last_successful_run.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Show actual lookback that will be used
        cutoff_time = self._get_dynamic_news_cutoff_time()
        actual_lookback = (datetime.now(timezone.utc) - cutoff_time).total_seconds() / 3600
        log_info(f"   Actual lookback: {actual_lookback:.1f} hours")
        
        log_info(f"   Max articles per cycle: {Config.MAX_NEWS_ARTICLES}")
        log_info(f"   Output file: {Config.CSV_OUTPUT_PATH}")
        log_info(f"   Database: {Config.SQLITE_DB_PATH}")
        
        # Enhanced AI service status
        service_status = self.llm_analyzer.get_service_status()
        log_info(f"🧠 Enhanced AI Analysis:")
        
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


async def main():
    """Enhanced main entry point"""
    try:
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
