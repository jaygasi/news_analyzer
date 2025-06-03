"""
Optimized main trading system with enhanced performance monitoring and error recovery
"""
import asyncio
import time
import hashlib
import json
from datetime import datetime, timezone, timedelta
import signal
import sys
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from data_loaders.simple_fmp_loader import SimpleFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader
from trading.trade_models import ProcessedArticle
import pandas as pd


class OptimizedNewsProcessor:
    """High-performance news processor with intelligent deduplication and caching."""
    
    # Class-level constants for performance
    MAX_CACHE_SIZE = 3000
    DEDUP_SIMILARITY_THRESHOLD = 0.65  # Reduced from 0.75 to allow more variation
    DEFAULT_MAX_AGE_HOURS = 6  # Reduced from 12 to get fresher content
    CACHE_CLEANUP_INTERVAL = 1800  # 30 minutes
    
    def __init__(self):
        """Initialize with optimized configuration."""
        self.processed_articles_file = CONFIG.cache_dir / "processed_articles.json"
        self.processed_hashes: Set[str] = set()
        self.processed_articles: Dict[str, ProcessedArticle] = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'duplicates_filtered': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Auto-configuration based on mode
        self.max_age_hours = 24 if CONFIG.testing_mode else self.DEFAULT_MAX_AGE_HOURS
        self.last_cleanup = time.time()
        
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize cache with error recovery."""
        try:
            self._load_processed_articles()
            self._cleanup_old_articles()
            log_info(f"News processor initialized with {len(self.processed_articles)} cached articles")
        except Exception as e:
            log_error(f"Error initializing news processor cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def _generate_article_hash(self, row: pd.Series) -> str:
        """Generate optimized article hash for deduplication."""
        try:
            symbol = str(row.get('symbol', '')).strip().upper()
            title = str(row.get('title', '')).strip().lower()
            # Use only date, not hour, to allow for updates throughout the day
            date_only = str(row.get('publishedDate', ''))[:10]  # YYYY-MM-DD only
            
            # Normalize title for better deduplication
            normalized_title = self._normalize_title_advanced(title)
            
            # Create hash key - removed hour to allow for updated articles
            hash_input = f"{symbol}-{normalized_title}-{date_only}"
            return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            
        except Exception as e:
            log_debug(f"Error generating article hash: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    def _normalize_title_advanced(self, title: str) -> str:
        """Advanced title normalization for improved deduplication."""
        import re
        
        # Remove common noise patterns
        noise_patterns = [
            r'^\s*(breaking|news|update|alert):\s*',
            r'\s*-\s*(reuters|bloomberg|cnbc|marketwatch|yahoo|benzinga).*$',
            r'\b(today|yesterday|this morning|this afternoon|tonight|just now|moments ago)\b',
            r'\b\d{1,2}:\d{2}\s*(AM|PM|EST|EDT|PST|PDT)\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b(update|updated|revision|revised)\b'
        ]
        
        cleaned_title = title
        for pattern in noise_patterns:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE)
        
        # Normalize whitespace and length
        cleaned_title = ' '.join(cleaned_title.split())
        return cleaned_title[:80]  # Reduced from 100 for better matching
    
    def _load_processed_articles(self) -> None:
        """Load processed articles with robust error handling."""
        if not self.processed_articles_file.exists():
            return
        
        try:
            with open(self.processed_articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_count = 0
            for article_hash, article_data in data.items():
                try:
                    processed_article = ProcessedArticle.from_dict(article_data)
                    
                    # Validate article age
                    if self._is_article_recent(processed_article):
                        self.processed_articles[article_hash] = processed_article
                        self.processed_hashes.add(article_hash)
                        loaded_count += 1
                        
                except Exception as e:
                    log_debug(f"Skipping invalid cached article: {e}")
                    continue
            
            log_info(f"Loaded {loaded_count} valid articles from cache")
            
        except Exception as e:
            log_warning(f"Error loading article cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def _is_article_recent(self, article: ProcessedArticle) -> bool:
        """Check if article is within age limits."""
        try:
            article_time = article.processed_datetime
            current_time = datetime.now(timezone.utc)
            age_hours = (current_time - article_time).total_seconds() / 3600
            
            return age_hours <= self.max_age_hours
            
        except Exception:
            return False
    
    def _save_processed_articles(self) -> None:
        """Save articles with atomic write and compression."""
        try:
            # Prepare data for serialization
            data = {
                article_hash: article.to_dict() 
                for article_hash, article in self.processed_articles.items()
            }
            
            # Atomic write using temporary file
            temp_file = self.processed_articles_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            
            # Atomic replace
            temp_file.replace(self.processed_articles_file)
            
            log_debug(f"Saved {len(data)} articles to cache")
            
        except Exception as e:
            log_error(f"Error saving article cache: {e}")
    
    def filter_new_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized article filtering with performance tracking."""
        if news_df is None or news_df.empty:
            return news_df
        
        start_time = time.time()
        initial_count = len(news_df)
        
        try:
            # Periodic cleanup
            if time.time() - self.last_cleanup > self.CACHE_CLEANUP_INTERVAL:
                self._cleanup_old_articles()
                self.last_cleanup = time.time()
            
            new_articles = []
            duplicate_count = 0
            
            # Process articles efficiently
            for idx, row in news_df.iterrows():
                article_hash = self._generate_article_hash(row)
                
                if article_hash in self.processed_hashes:
                    duplicate_count += 1
                    self.processing_stats['cache_hits'] += 1
                else:
                    # Additional similarity check for current batch
                    if not self._is_similar_to_recent(row, new_articles[-10:]):  # Reduced from 20 to 10
                        new_articles.append(row)
                        self.processing_stats['cache_misses'] += 1
                    else:
                        duplicate_count += 1
                        self.processing_stats['duplicates_filtered'] += 1
            
            # Create result DataFrame
            result_df = pd.DataFrame(new_articles).reset_index(drop=True) if new_articles else pd.DataFrame()
            
            # Update stats
            self.processing_stats['duplicates_filtered'] += duplicate_count
            
            # Log performance
            processing_time = time.time() - start_time
            log_info(f"Article filtering: {initial_count} → {len(result_df)} articles "
                    f"(removed {duplicate_count} duplicates) in {processing_time:.2f}s")
            
            return result_df
            
        except Exception as e:
            log_error(f"Error filtering articles: {e}")
            return news_df
    
    def _is_similar_to_recent(self, current_row: pd.Series, recent_articles: List) -> bool:
        """Check similarity to recently processed articles in current batch."""
        if not recent_articles:
            return False
        
        try:
            current_symbol = str(current_row.get('symbol', '')).upper()
            current_title = self._normalize_title_advanced(str(current_row.get('title', '')).lower())
            current_words = set(word for word in current_title.split() if len(word) > 3)
            
            for existing_row in recent_articles:
                existing_symbol = str(existing_row.get('symbol', '')).upper()
                existing_title = self._normalize_title_advanced(str(existing_row.get('title', '')).lower())
                existing_words = set(word for word in existing_title.split() if len(word) > 3)
                
                # Check symbol and content similarity
                if (current_symbol == existing_symbol and 
                    current_words and existing_words and
                    len(current_words & existing_words) / len(current_words | existing_words) > self.DEDUP_SIMILARITY_THRESHOLD):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def mark_articles_processed(self, analyses: List, original_news_df: pd.DataFrame) -> None:
        """Mark articles as processed with optimized batch operations."""
        if original_news_df.empty:
            return
        
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            new_processed = 0
            
            # Create analysis lookup for efficiency
            analysis_lookup = {
                analysis.symbol: analysis for analysis in analyses
            } if analyses else {}
            
            # Batch process articles
            for _, row in original_news_df.iterrows():
                article_hash = self._generate_article_hash(row)
                
                if article_hash in self.processed_hashes:
                    continue
                
                symbol = str(row.get('symbol', '')).strip().upper()
                analysis = analysis_lookup.get(symbol)
                
                # Create processed article record
                processed_article = ProcessedArticle(
                    article_hash=article_hash,
                    symbol=symbol,
                    title_hash=self._normalize_title_advanced(str(row.get('title', '')).lower()),
                    processed_time=current_time,
                    sentiment_score=analysis.sentiment_score if analysis else 0.0,
                    combined_confidence=analysis.combined_confidence if analysis else 0.0,
                    was_traded=1 if (analysis and analysis.combined_confidence >= CONFIG.min_confidence_score) else 0,
                    trade_side=self._determine_trade_side(analysis.sentiment_score) if analysis else ""
                )
                
                self.processed_articles[article_hash] = processed_article
                self.processed_hashes.add(article_hash)
                new_processed += 1
            
            # Update stats and save
            self.processing_stats['total_processed'] += new_processed
            
            if new_processed > 0:
                log_info(f"Marked {new_processed} articles as processed")
                self._save_processed_articles()
                
        except Exception as e:
            log_error(f"Error marking articles as processed: {e}")
    
    def _determine_trade_side(self, sentiment_score: float) -> str:
        """Determine trade side from sentiment score."""
        if sentiment_score > 0.20:
            return "long"
        elif sentiment_score < -0.20:
            return "short"
        return ""
    
    def _cleanup_old_articles(self) -> None:
        """Optimized cleanup of old articles."""
        if not self.processed_articles:
            return
        
        try:
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time - timedelta(hours=self.max_age_hours)
            
            # Find old articles
            old_hashes = [
                article_hash for article_hash, article in self.processed_articles.items()
                if not self._is_article_recent(article)
            ]
            
            # Remove old articles
            for article_hash in old_hashes:
                self.processed_articles.pop(article_hash, None)
                self.processed_hashes.discard(article_hash)
            
            # Limit total size
            if len(self.processed_articles) > self.MAX_CACHE_SIZE:
                self._trim_cache_to_size()
            
            if old_hashes:
                log_info(f"Cleaned up {len(old_hashes)} old articles, "
                        f"cache now has {len(self.processed_articles)} articles")
                self._save_processed_articles()
                
        except Exception as e:
            log_error(f"Error cleaning up articles: {e}")
    
    def _trim_cache_to_size(self) -> None:
        """Trim cache to maximum size keeping most valuable articles."""
        if len(self.processed_articles) <= self.MAX_CACHE_SIZE:
            return
        
        try:
            # Sort by value (recent + high confidence)
            sorted_articles = sorted(
                self.processed_articles.items(),
                key=lambda x: (x[1].processed_datetime.timestamp(), x[1].combined_confidence),
                reverse=True
            )
            
            # Keep top articles
            keep_count = int(self.MAX_CACHE_SIZE * 0.9)
            articles_to_keep = dict(sorted_articles[:keep_count])
            
            # Update structures
            removed_count = len(self.processed_articles) - len(articles_to_keep)
            self.processed_articles = articles_to_keep
            self.processed_hashes = set(self.processed_articles.keys())
            
            log_info(f"Trimmed cache: removed {removed_count} articles")
            
        except Exception as e:
            log_error(f"Error trimming cache: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        current_time = datetime.now(timezone.utc)
        
        # Calculate additional metrics
        high_confidence_count = sum(
            1 for article in self.processed_articles.values()
            if article.combined_confidence >= CONFIG.min_confidence_score
        )
        
        trades_count = sum(
            1 for article in self.processed_articles.values()
            if article.was_traded == 1
        )
        
        symbols_covered = len(set(article.symbol for article in self.processed_articles.values()))
        
        # Average confidence
        confidences = [article.combined_confidence for article in self.processed_articles.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            **self.processing_stats,
            'cached_articles': len(self.processed_articles),
            'high_confidence_signals': high_confidence_count,
            'actual_trades': trades_count,
            'symbols_covered': symbols_covered,
            'avg_confidence': avg_confidence,
            'max_age_hours': self.max_age_hours,
            'cache_efficiency': self.processing_stats['cache_hits'] / max(
                self.processing_stats['cache_hits'] + self.processing_stats['cache_misses'], 1
            )
        }


class EnhancedTradingSystem:
    """High-performance trading system with comprehensive monitoring and error recovery."""
    
    # System-level constants
    MAX_CONSECUTIVE_ERRORS = 10
    ERROR_BACKOFF_BASE = 2
    MAX_BACKOFF_SECONDS = 300  # 5 minutes
    UNIVERSE_REFRESH_HOURS = 4  # Increased from 2 hours
    STATUS_INTERVAL_MINUTES = 5  # Increased from 3 minutes
    
    def __init__(self) -> None:
        """Initialize trading system with robust error handling."""
        # Validate API key
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        self.fmp_api_key = api_key
        self.running = True
        
        # System state
        self.universe: List[str] = []
        self.error_count = 0
        self.last_universe_refresh = 0.0
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'cycles_completed': 0,
            'total_articles_processed': 0,
            'total_trades_created': 0,
            'uptime_start': time.time(),
            'errors_recovered': 0
        }
        
        # Initialize components
        self._initialize_system_components()
        self._setup_signal_handlers()
        
        log_info("Enhanced trading system initialized successfully")
    
    def _initialize_system_components(self) -> None:
        """Initialize all system components with error recovery."""
        try:
            self.fmp_loader = SimpleFMPLoader(self.fmp_api_key)
            self.news_analyzer = EnhancedNewsAnalyzer(self.fmp_loader)
            self.trader = EnhancedTrader(self.fmp_loader)
            self.news_processor = OptimizedNewsProcessor()
            
            log_info("All system components initialized")
            
        except Exception as e:
            log_error(f"Failed to initialize system components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum: int, frame) -> None:
            log_info(f"Received signal {signum}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_universe(self) -> bool:
        """Initialize trading universe with enhanced error recovery."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                log_info(f"Initializing trading universe (attempt {attempt + 1}/{max_attempts})")
                
                screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
                
                if screener_df is not None and not screener_df.empty:
                    # Enhanced universe filtering
                    self.universe = self._create_optimized_universe(screener_df)
                    
                    if len(self.universe) > 0:
                        self.last_universe_refresh = time.time()
                        log_info(f"Universe initialized with {len(self.universe)} symbols")
                        log_info(f"Sample symbols: {self.universe[:15]}")
                        return True
                
            except Exception as e:
                log_error(f"Universe initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        log_error("Failed to initialize trading universe after all attempts")
        return False
    
    def _create_optimized_universe(self, screener_df: pd.DataFrame) -> List[str]:
        """Create optimized universe with news-focused filtering."""
        try:
            # Extract and clean symbols
            symbols = screener_df['symbol'].str.upper().tolist()
            
            # Enhanced filtering for news-active stocks
            filtered_symbols = []
            for symbol in symbols:
                if self._is_suitable_for_news_trading(symbol):
                    filtered_symbols.append(symbol)
            
            # Sort by market cap if available for better news coverage
            if 'marketCap' in screener_df.columns:
                df_with_filtered = screener_df[screener_df['symbol'].isin(filtered_symbols)]
                df_sorted = df_with_filtered.sort_values('marketCap', ascending=False)
                return df_sorted['symbol'].str.upper().tolist()
            
            return filtered_symbols
            
        except Exception as e:
            log_error(f"Error creating optimized universe: {e}")
            return screener_df['symbol'].str.upper().tolist()[:CONFIG.max_symbols]
    
    def _is_suitable_for_news_trading(self, symbol: str) -> bool:
        """Check if symbol is suitable for news-based trading."""
        if not symbol or len(symbol) > 5:
            return False
        
        # Filter out complex symbols
        if any(char in symbol for char in ['.', '-', '/']):
            return False
        
        # Must be alphabetic
        if not symbol.replace('-', '').replace('.', '').isalpha():
            return False
        
        return True
    
    def _refresh_universe_if_needed(self) -> None:
        """Refresh universe periodically for optimal coverage."""
        current_time = time.time()
        hours_since_refresh = (current_time - self.last_universe_refresh) / 3600
        
        if hours_since_refresh >= self.UNIVERSE_REFRESH_HOURS:
            log_info("Refreshing trading universe...")
            try:
                if self.initialize_universe():
                    log_info("Universe refresh completed successfully")
                else:
                    log_warning("Universe refresh failed, using existing universe")
            except Exception as e:
                log_error(f"Error refreshing universe: {e}")
    
    async def run(self) -> None:
        """Main system run loop with enhanced error recovery."""
        log_info("🚀 Starting Enhanced News Catalyst Trading System")
        log_info(f"Configuration: {CONFIG.news_page_limit} pages, "
                f"{CONFIG.news_per_page_limit} per page, "
                f"confidence threshold: {CONFIG.min_confidence_score}")
        log_info(f"Universe size: {CONFIG.max_symbols}, Cache age limit: {self.news_processor.max_age_hours}h")
        
        # Initialize universe
        if not self.initialize_universe():
            log_error("❌ Failed to initialize trading universe")
            return
        
        # Main trading loop
        await self._execute_main_loop()
        
        # Shutdown
        self._perform_graceful_shutdown()
        log_info("🏁 Trading system shutdown complete")
    
    async def _execute_main_loop(self) -> None:
        """Execute main trading loop with comprehensive error handling."""
        last_news_check = 0.0
        last_position_check = 0.0
        last_status_print = 0.0
        
        log_info("📊 Enhanced trading system active - Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # News processing cycle
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    await self._process_news_cycle()
                    last_news_check = current_time
                    self.error_count = 0  # Reset error count on success
                
                # Position monitoring cycle
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    await self._process_position_cycle()
                    last_position_check = current_time
                
                # Status reporting cycle
                status_interval = self.STATUS_INTERVAL_MINUTES * 60
                if current_time - last_status_print >= status_interval:
                    self._print_comprehensive_status()
                    last_status_print = current_time
                
                # Universe refresh check (less frequent)
                if self.cycle_count % 100 == 0:  # Every 100 cycles instead of 50
                    self._refresh_universe_if_needed()
                
                await asyncio.sleep(2.0)  # Increased from 1.0 to reduce load
                
            except Exception as e:
                await self._handle_main_loop_error(e)
    
    async def _process_news_cycle(self) -> None:
        """Process news cycle with comprehensive error handling."""
        cycle_start = time.time()
        self.cycle_count += 1
        
        try:
            log_debug(f"Starting news cycle #{self.cycle_count}")
            
            # Get raw news
            raw_news_df = self.fmp_loader.get_comprehensive_news()
            
            if raw_news_df is None or raw_news_df.empty:
                log_debug("No news data received")
                return
            
            log_info(f"Retrieved {len(raw_news_df)} raw articles")
            
            # Filter to universe
            universe_news_df = self._filter_news_to_universe(raw_news_df)
            
            if universe_news_df.empty:
                log_debug("No universe news found")
                return
            
            # Process through pipeline
            await self._process_news_pipeline(universe_news_df)
            
            # Update performance metrics
            cycle_time = time.time() - cycle_start
            self.performance_metrics['cycles_completed'] += 1
            
            log_debug(f"News cycle #{self.cycle_count} completed in {cycle_time:.2f}s")
            
        except Exception as e:
            log_error(f"Error in news cycle #{self.cycle_count}: {e}")
            log_debug(f"News cycle error traceback: {traceback.format_exc()}")
    
    def _filter_news_to_universe(self, raw_news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter news to universe symbols efficiently."""
        try:
            universe_set = set(self.universe)
            
            # Efficient filtering using pandas operations
            universe_mask = raw_news_df['symbol'].str.upper().isin(universe_set)
            filtered_df = raw_news_df[universe_mask].copy()
            
            if not filtered_df.empty:
                symbol_count = filtered_df['symbol'].nunique()
                log_info(f"Universe filter: {len(filtered_df)} articles across {symbol_count} symbols")
            
            return filtered_df
            
        except Exception as e:
            log_error(f"Error filtering news to universe: {e}")
            return pd.DataFrame()
    
    async def _process_news_pipeline(self, universe_news_df: pd.DataFrame) -> None:
        """Process news through the analysis pipeline."""
        try:
            # Filter new articles
            new_articles_df = self.news_processor.filter_new_articles(universe_news_df)
            
            if new_articles_df.empty:
                log_debug("No new articles to process")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            log_info(f"Processing {len(new_articles_df)} new articles")
            
            # Quality filtering
            quality_filtered_df = self.trader.pre_filter_news(new_articles_df)
            
            if quality_filtered_df is None or quality_filtered_df.empty:
                log_info("All articles filtered by quality checks")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            # Get prices for relevant symbols
            relevant_symbols = quality_filtered_df['symbol'].unique().tolist()
            prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
            
            if prices_df is None or prices_df.empty:
                log_warning(f"Failed to get prices for {len(relevant_symbols)} symbols")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            # Perform analysis
            analyses = self.news_analyzer.analyze_news_with_technical(
                quality_filtered_df, prices_df
            )
            
            # Mark articles as processed
            self.news_processor.mark_articles_processed(analyses, universe_news_df)
            
            if analyses:
                # Process trading signals
                self.trader.process_news_signals(analyses, prices_df)
                self.performance_metrics['total_articles_processed'] += len(analyses)
            else:
                log_info("No valid analyses generated")
                
        except Exception as e:
            log_error(f"Error in news pipeline: {e}")
    
    async def _process_position_cycle(self) -> None:
        """Process position monitoring cycle."""
        try:
            active_positions = self.trader.get_active_positions()
            
            if not active_positions:
                return
            
            # Get current prices for active positions
            symbols = list(set(trade.symbol for trade in active_positions))
            prices_df = self.fmp_loader.get_real_time_prices(symbols)
            
            if prices_df is not None and not prices_df.empty:
                self.trader.check_exits(prices_df)
            else:
                log_debug(f"Failed to get prices for {len(symbols)} active positions")
                
        except Exception as e:
            log_error(f"Error in position cycle: {e}")
    
    async def _handle_main_loop_error(self, error: Exception) -> None:
        """Handle main loop errors with exponential backoff."""
        self.error_count += 1
        self.performance_metrics['errors_recovered'] += 1
        
        log_error(f"Main loop error #{self.error_count}: {error}")
        
        if self.error_count >= self.MAX_CONSECUTIVE_ERRORS:
            log_error("Too many consecutive errors, shutting down")
            self.running = False
            return
        
        # Exponential backoff
        backoff_time = min(
            self.ERROR_BACKOFF_BASE ** self.error_count,
            self.MAX_BACKOFF_SECONDS
        )
        
        log_info(f"Sleeping {backoff_time}s before retry (error #{self.error_count})")
        await asyncio.sleep(backoff_time)
    
    def _print_comprehensive_status(self) -> None:
        """Print comprehensive system status."""
        try:
            uptime_hours = (time.time() - self.performance_metrics['uptime_start']) / 3600
            
            # Trading metrics
            active_positions = len(self.trader.get_active_positions())
            daily_pnl = self.trader.get_daily_pnl()
            
            # Performance metrics
            trader_stats = self.trader.get_performance_summary()
            news_stats = self.news_processor.get_processing_stats()
            analyzer_stats = self.news_analyzer.get_performance_stats()
            
            log_info("=" * 80)
            log_info("📊 ENHANCED TRADING SYSTEM STATUS")
            log_info("=" * 80)
            
            # System health
            log_info(f"🎯 System Health:")
            log_info(f"   Uptime: {uptime_hours:.1f} hours")
            log_info(f"   Cycles Completed: {self.performance_metrics['cycles_completed']}")
            log_info(f"   Universe Size: {len(self.universe)}")
            log_info(f"   Error Count: {self.error_count}")
            log_info(f"   Errors Recovered: {self.performance_metrics['errors_recovered']}")
            
            # Trading performance
            log_info(f"💰 Trading Performance:")
            log_info(f"   Active Positions: {active_positions}")
            log_info(f"   Daily P&L: ${daily_pnl:.2f}")
            log_info(f"   Total Trades Created: {trader_stats.get('trades_created', 0)}")
            log_info(f"   Recommendations Logged: {trader_stats.get('recommendations_logged', 0)}")
            
            # News processing
            log_info(f"📰 News Processing:")
            log_info(f"   Articles Cached: {news_stats.get('cached_articles', 0)}")
            log_info(f"   Cache Efficiency: {news_stats.get('cache_efficiency', 0):.1%}")
            log_info(f"   High Confidence Signals: {news_stats.get('high_confidence_signals', 0)}")
            log_info(f"   Symbols Covered: {news_stats.get('symbols_covered', 0)}")
            
            # Analysis performance
            log_info(f"🔍 Analysis Performance:")
            log_info(f"   Total Analyses: {analyzer_stats.get('total_analyses', 0)}")
            log_info(f"   FinBERT Enabled: {analyzer_stats.get('finbert_enabled', False)}")
            log_info(f"   Gemini Enabled: {analyzer_stats.get('gemini_enabled', False)}")
            log_info(f"   Processing Device: {analyzer_stats.get('device', 'unknown')}")
            
            log_info("=" * 80)
            
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    def _perform_graceful_shutdown(self) -> None:
        """Perform graceful system shutdown."""
        try:
            log_info("Performing graceful shutdown...")
            
            # Save final state
            final_stats = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'uptime_hours': (time.time() - self.performance_metrics['uptime_start']) / 3600,
                'final_performance': self.performance_metrics,
                'final_universe_size': len(self.universe),
                'active_positions': len(self.trader.get_active_positions())
            }
            
            log_info(f"Final stats: {final_stats}")
            
        except Exception as e:
            log_error(f"Error during graceful shutdown: {e}")


async def main() -> int:
    """Main entry point with comprehensive error handling."""
    try:
        # Create and run trading system
        system = EnhancedTradingSystem()
        await system.run()
        return 0
        
    except KeyboardInterrupt:
        log_info("System interrupted by user")
        return 0
        
    except Exception as e:
        log_error(f"Fatal system error: {e}")
        log_error(f"Fatal error traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start system: {e}")
        sys.exit(1)