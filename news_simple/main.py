"""
Optimized main trading system with fixed caching and news processing
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
from data_loaders.enhanced_fmp_loader import EnhancedFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader
from trading.trade_models import ProcessedArticle
import pandas as pd

class OptimizedNewsProcessor:
    """Fixed news processor with proper caching for 25-second scans."""
    
    # Optimized for 25-second scanning
    MAX_CACHE_SIZE = 500
    DEDUP_SIMILARITY_THRESHOLD = 0.85
    CACHE_RETENTION_HOURS = 1  # Much shorter retention
    CACHE_CLEANUP_INTERVAL = 300  # 5 minutes
    FRESH_BYPASS_MINUTES = 5  # More realistic bypass window
    
    def __init__(self):
        """Initialize with configuration optimized for frequent scanning."""
        self.processed_articles_file = CONFIG.cache_dir / "processed_articles.json"
        self.processed_hashes: Set[str] = set()
        self.processed_articles: Dict[str, ProcessedArticle] = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'duplicates_filtered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fresh_articles_processed': 0,
            'new_articles_found': 0
        }
        
        # Adaptive settings based on scan frequency
        self.scan_frequency = CONFIG.news_check_interval
        self.last_cleanup = time.time()
        
        self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize cache with appropriate settings for scan frequency."""
        try:
            self._load_processed_articles()
            self._cleanup_old_articles()
            log_info(f"News processor initialized: {len(self.processed_articles)} cached articles "
                    f"(retention: {self.CACHE_RETENTION_HOURS}h, scan: {self.scan_frequency}s)")
        except Exception as e:
            log_error(f"Error initializing cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def _generate_article_hash(self, row: pd.Series) -> str:
        """Generate improved hash for better duplicate detection."""
        try:
            symbol = str(row.get('symbol', '')).strip().upper()
            title = str(row.get('title', '')).strip().lower()
            
            # Use more specific hash - include hour for rapid updates
            pub_date = str(row.get('publishedDate', ''))
            date_hour = pub_date[:13] if len(pub_date) >= 13 else pub_date
            
            # Normalize title more aggressively
            normalized_title = self._normalize_title_for_hash(title)
            
            # Create hash
            hash_input = f"{symbol}|{normalized_title}|{date_hour}"
            return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            
        except Exception as e:
            log_debug(f"Error generating hash: {e}")
            return hashlib.md5(f"{time.time()}".encode()).hexdigest()
    
    def _normalize_title_for_hash(self, title: str) -> str:
        """Normalize title for better hash consistency."""
        import re
        
        # Remove common variations that shouldn't affect hash
        title = re.sub(r'\b\d{1,2}:\d{2}\s*(AM|PM)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*-\s*(reuters|bloomberg|ap|marketwatch).*$', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^\s*(breaking|news alert|update):\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title[:100]  # Limit length
    
    def _load_processed_articles(self) -> None:
        """Load processed articles with strict age filtering."""
        if not self.processed_articles_file.exists():
            return
        
        try:
            with open(self.processed_articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_time = datetime.now(timezone.utc)
            loaded_count = 0
            
            for article_hash, article_data in data.items():
                try:
                    processed_article = ProcessedArticle.from_dict(article_data)
                    
                    # Strict age check for frequent scanning
                    age_hours = (current_time - processed_article.processed_datetime).total_seconds() / 3600
                    
                    if age_hours <= self.CACHE_RETENTION_HOURS:
                        self.processed_articles[article_hash] = processed_article
                        self.processed_hashes.add(article_hash)
                        loaded_count += 1
                        
                except Exception:
                    continue
            
            log_info(f"Loaded {loaded_count} recent articles (max age: {self.CACHE_RETENTION_HOURS}h)")
            
        except Exception as e:
            log_warning(f"Error loading cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def filter_new_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Improved article filtering with fixed logic."""
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
            similar_count = 0
            fresh_count = 0
            
            for idx, row in news_df.iterrows():
                article_hash = self._generate_article_hash(row)
                
                # Check if we've seen this exact article recently
                if article_hash in self.processed_hashes:
                    cached_article = self.processed_articles.get(article_hash)
                    if cached_article:
                        # For 25s scans, only reprocess if very old or low confidence
                        age_minutes = (datetime.now(timezone.utc) - 
                                     cached_article.processed_datetime).total_seconds() / 60
                        
                        should_reprocess = (
                            age_minutes > 30 or  # Older than 30 minutes
                            (age_minutes > 10 and cached_article.combined_confidence < 0.4)  # Low confidence after 10 min
                        )
                        
                        if should_reprocess:
                            new_articles.append(row)
                            fresh_count += 1
                        else:
                            duplicate_count += 1
                            self.processing_stats['cache_hits'] += 1
                    else:
                        # Hash exists but no article data - process it
                        new_articles.append(row)
                        self.processing_stats['cache_misses'] += 1
                else:
                    # New article hash - check for similarity to recent articles
                    if not self._is_too_similar_to_recent(row, new_articles[-5:]):  # Only check last 5
                        new_articles.append(row)
                        self.processing_stats['cache_misses'] += 1
                        self.processing_stats['new_articles_found'] += 1
                    else:
                        similar_count += 1
                        self.processing_stats['duplicates_filtered'] += 1
            
            # Create result
            result_df = pd.DataFrame(new_articles).reset_index(drop=True) if new_articles else pd.DataFrame()
            
            processing_time = time.time() - start_time
            log_info(f"Article filter: {initial_count} → {len(result_df)} "
                    f"(dups: {duplicate_count}, similar: {similar_count}, fresh: {fresh_count}) "
                    f"in {processing_time:.2f}s")
            
            return result_df
            
        except Exception as e:
            log_error(f"Error filtering articles: {e}")
            return news_df
    
    def _is_too_similar_to_recent(self, current_row: pd.Series, recent_articles: List) -> bool:
        """Check similarity to recent articles with high threshold."""
        if not recent_articles:
            return False
        
        try:
            current_symbol = str(current_row.get('symbol', '')).upper()
            current_title = self._normalize_title_for_hash(str(current_row.get('title', '')).lower())
            current_words = set(word for word in current_title.split() if len(word) > 3)
            
            if not current_words:
                return False
            
            # Only check same symbol
            for existing_row in recent_articles:
                existing_symbol = str(existing_row.get('symbol', '')).upper()
                
                if current_symbol != existing_symbol:
                    continue
                
                existing_title = self._normalize_title_for_hash(str(existing_row.get('title', '')).lower())
                existing_words = set(word for word in existing_title.split() if len(word) > 3)
                
                if existing_words:
                    similarity = len(current_words & existing_words) / len(current_words | existing_words)
                    if similarity > self.DEDUP_SIMILARITY_THRESHOLD:
                        return True
            
            return False
            
        except:
            return False
    
    def mark_articles_processed(self, analyses: List, original_news_df: pd.DataFrame) -> None:
        """Mark articles as processed with optimized logic."""
        if original_news_df.empty:
            return
        
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            new_processed = 0
            
            # Create analysis lookup
            analysis_lookup = {analysis.symbol: analysis for analysis in analyses} if analyses else {}
            
            for _, row in original_news_df.iterrows():
                article_hash = self._generate_article_hash(row)
                symbol = str(row.get('symbol', '')).strip().upper()
                analysis = analysis_lookup.get(symbol)
                
                # Create processed record
                processed_article = ProcessedArticle(
                    article_hash=article_hash,
                    symbol=symbol,
                    title_hash=self._normalize_title_for_hash(str(row.get('title', '')).lower()),
                    processed_time=current_time,
                    sentiment_score=analysis.sentiment_score if analysis else 0.0,
                    combined_confidence=analysis.combined_confidence if analysis else 0.0,
                    was_traded=1 if (analysis and analysis.combined_confidence >= CONFIG.min_confidence_score) else 0,
                    trade_side=self._determine_trade_side(analysis.sentiment_score) if analysis else ""
                )
                
                self.processed_articles[article_hash] = processed_article
                self.processed_hashes.add(article_hash)
                new_processed += 1
            
            self.processing_stats['total_processed'] += new_processed
            
            if new_processed > 0:
                log_info(f"Marked {new_processed} articles as processed")
                self._save_processed_articles()
                
        except Exception as e:
            log_error(f"Error marking articles: {e}")
    
    def _determine_trade_side(self, sentiment_score: float) -> str:
        """Determine trade side from sentiment."""
        if sentiment_score > 0.20:
            return "long"
        elif sentiment_score < -0.20:
            return "short"
        return ""
    
    def _cleanup_old_articles(self) -> None:
        """Clean up old articles with aggressive pruning."""
        if not self.processed_articles:
            return
        
        try:
            initial_count = len(self.processed_articles)
            current_time = datetime.now(timezone.utc)
            
            # Remove old articles
            old_hashes = []
            for article_hash, article in self.processed_articles.items():
                age_hours = (current_time - article.processed_datetime).total_seconds() / 3600
                
                if age_hours > self.CACHE_RETENTION_HOURS:
                    old_hashes.append(article_hash)
            
            # Remove old entries
            for article_hash in old_hashes:
                self.processed_articles.pop(article_hash, None)
                self.processed_hashes.discard(article_hash)
            
            # Size limit
            if len(self.processed_articles) > self.MAX_CACHE_SIZE:
                self._trim_cache_to_size()
            
            if old_hashes:
                log_info(f"Cleaned {len(old_hashes)} old articles ({initial_count} → {len(self.processed_articles)})")
                self._save_processed_articles()
                
        except Exception as e:
            log_error(f"Error cleaning cache: {e}")
    
    def _trim_cache_to_size(self) -> None:
        """Trim cache to size keeping most valuable articles."""
        if len(self.processed_articles) <= self.MAX_CACHE_SIZE:
            return
        
        try:
            # Sort by value (recent + high confidence + traded)
            sorted_articles = sorted(
                self.processed_articles.items(),
                key=lambda x: (
                    x[1].processed_datetime.timestamp(),
                    x[1].combined_confidence,
                    x[1].was_traded
                ),
                reverse=True
            )
            
            # Keep top 75%
            keep_count = int(self.MAX_CACHE_SIZE * 0.75)
            articles_to_keep = dict(sorted_articles[:keep_count])
            
            removed_count = len(self.processed_articles) - len(articles_to_keep)
            self.processed_articles = articles_to_keep
            self.processed_hashes = set(self.processed_articles.keys())
            
            log_info(f"Trimmed cache: removed {removed_count} articles")
            
        except Exception as e:
            log_error(f"Error trimming cache: {e}")
    
    def _save_processed_articles(self) -> None:
        """Save articles atomically."""
        try:
            data = {
                article_hash: article.to_dict() 
                for article_hash, article in self.processed_articles.items()
            }
            
            temp_file = self.processed_articles_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            
            temp_file.replace(self.processed_articles_file)
            log_debug(f"Saved {len(data)} articles to cache")
            
        except Exception as e:
            log_error(f"Error saving cache: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        current_time = datetime.now(timezone.utc)
        
        high_confidence_count = sum(
            1 for article in self.processed_articles.values()
            if article.combined_confidence >= CONFIG.min_confidence_score
        )
        
        trades_count = sum(
            1 for article in self.processed_articles.values()
            if article.was_traded == 1
        )
        
        symbols_covered = len(set(article.symbol for article in self.processed_articles.values()))
        
        confidences = [article.combined_confidence for article in self.processed_articles.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            **self.processing_stats,
            'cached_articles': len(self.processed_articles),
            'high_confidence_signals': high_confidence_count,
            'actual_trades': trades_count,
            'symbols_covered': symbols_covered,
            'avg_confidence': avg_confidence,
            'cache_retention_hours': self.CACHE_RETENTION_HOURS,
            'cache_efficiency': self.processing_stats['cache_hits'] / max(
                self.processing_stats['cache_hits'] + self.processing_stats['cache_misses'], 1
            ),
            'similarity_threshold': self.DEDUP_SIMILARITY_THRESHOLD,
            'scan_frequency_seconds': self.scan_frequency
        }


class EnhancedTradingSystem:
    """Trading system with fixed news processing."""
    
    MAX_CONSECUTIVE_ERRORS = 10
    ERROR_BACKOFF_BASE = 2
    MAX_BACKOFF_SECONDS = 300
    UNIVERSE_REFRESH_HOURS = 8  # Less frequent refresh
    STATUS_INTERVAL_MINUTES = 2
    
    def __init__(self) -> None:
        """Initialize trading system."""
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        self.fmp_api_key = api_key
        self.running = True
        
        self.universe: List[str] = []
        self.error_count = 0
        self.last_universe_refresh = 0.0
        self.cycle_count = 0
        
        self.performance_metrics = {
            'cycles_completed': 0,
            'total_articles_processed': 0,
            'total_trades_created': 0,
            'uptime_start': time.time(),
            'errors_recovered': 0,
            'fresh_articles_found': 0,
            'universe_refreshes': 0
        }
        
        self._initialize_system_components()
        self._setup_signal_handlers()
        
        log_info("Enhanced trading system initialized with fixed news processing")
    
    def _initialize_system_components(self) -> None:
        """Initialize system components."""
        try:
            self.fmp_loader = EnhancedFMPLoader(self.fmp_api_key)
            self.news_analyzer = EnhancedNewsAnalyzer(self.fmp_loader)
            self.trader = EnhancedTrader(self.fmp_loader)
            self.news_processor = OptimizedNewsProcessor()
            
            log_info("All system components initialized")
            
        except Exception as e:
            log_error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers."""
        def signal_handler(signum: int, frame) -> None:
            log_info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_universe(self) -> bool:
        """Initialize trading universe."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                log_info(f"Initializing universe (attempt {attempt + 1}/{max_attempts})")
                
                screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
                
                if screener_df is not None and not screener_df.empty:
                    self.universe = self._create_optimized_universe(screener_df)
                    
                    if len(self.universe) > 0:
                        self.last_universe_refresh = time.time()
                        self.performance_metrics['universe_refreshes'] += 1
                        
                        # Set universe cache for news loader
                        self.fmp_loader.set_universe_cache(self.universe)
                        
                        log_info(f"Universe initialized: {len(self.universe)} symbols")
                        log_info(f"Sample: {self.universe[:10]}")
                        return True
                
            except Exception as e:
                log_error(f"Universe init attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
        
        log_error("Failed to initialize universe")
        return False
    
    def _create_optimized_universe(self, screener_df: pd.DataFrame) -> List[str]:
        """Create optimized universe."""
        try:
            symbols = screener_df['symbol'].str.upper().tolist()
            
            filtered_symbols = []
            for symbol in symbols:
                if self._is_suitable_for_news_trading(symbol):
                    filtered_symbols.append(symbol)
            
            # Sort by market cap for better news coverage
            if 'marketCap' in screener_df.columns:
                df_filtered = screener_df[screener_df['symbol'].isin(filtered_symbols)]
                df_sorted = df_filtered.sort_values('marketCap', ascending=False)
                return df_sorted['symbol'].str.upper().tolist()
            
            return filtered_symbols
            
        except Exception as e:
            log_error(f"Error creating universe: {e}")
            return screener_df['symbol'].str.upper().tolist()[:CONFIG.max_symbols]
    
    def _is_suitable_for_news_trading(self, symbol: str) -> bool:
        """Check if symbol is suitable for news trading."""
        if not symbol or len(symbol) > 5:
            return False
        
        if any(char in symbol for char in ['.', '-', '/', '$']):
            return False
        
        if not symbol.replace('-', '').replace('.', '').isalpha():
            return False
        
        return True
    
    def _refresh_universe_if_needed(self) -> None:
        """Refresh universe periodically."""
        current_time = time.time()
        hours_since_refresh = (current_time - self.last_universe_refresh) / 3600
        
        if hours_since_refresh >= self.UNIVERSE_REFRESH_HOURS:
            log_info("Refreshing universe...")
            try:
                if self.initialize_universe():
                    log_info("Universe refresh completed")
                else:
                    log_warning("Universe refresh failed")
            except Exception as e:
                log_error(f"Error refreshing universe: {e}")
    
    async def run(self) -> None:
        """Main system run loop."""
        log_info("🚀 Starting Enhanced Trading System with Fixed News Processing")
        log_info(f"Config: {CONFIG.news_page_limit} pages, scan every {CONFIG.news_check_interval}s")
        log_info(f"Cache retention: {self.news_processor.CACHE_RETENTION_HOURS}h")
        
        if not self.initialize_universe():
            log_error("❌ Failed to initialize universe")
            return
        
        await self._execute_main_loop()
        self._perform_graceful_shutdown()
        log_info("🏁 System shutdown complete")
    
    async def _execute_main_loop(self) -> None:
        """Execute main trading loop."""
        last_news_check = 0.0
        last_position_check = 0.0
        last_status_print = 0.0
        
        log_info("📊 Trading system active - Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # News processing
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    await self._process_news_cycle()
                    last_news_check = current_time
                    self.error_count = 0
                
                # Position monitoring
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    await self._process_position_cycle()
                    last_position_check = current_time
                
                # Status reporting
                if current_time - last_status_print >= (self.STATUS_INTERVAL_MINUTES * 60):
                    self._print_system_status()
                    last_status_print = current_time
                
                # Universe refresh
                if self.cycle_count % 100 == 0:
                    self._refresh_universe_if_needed()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                await self._handle_main_loop_error(e)
    
    async def _process_news_cycle(self) -> None:
        """Process news cycle."""
        cycle_start = time.time()
        self.cycle_count += 1
        
        try:
            log_debug(f"News cycle #{self.cycle_count}")
            
            # Get fresh news
            raw_news_df = self.fmp_loader.get_comprehensive_news()
            
            if raw_news_df is None or raw_news_df.empty:
                log_debug("No fresh news")
                return
            
            # Filter to universe
            universe_news_df = self._filter_to_universe(raw_news_df)
            
            if universe_news_df.empty:
                log_debug("No universe news")
                return
            
            log_info(f"Got {len(universe_news_df)} universe articles from {universe_news_df['source'].nunique()} sources")
            
            # Process pipeline
            await self._process_news_pipeline(universe_news_df)
            
            cycle_time = time.time() - cycle_start
            self.performance_metrics['cycles_completed'] += 1
            log_debug(f"Cycle #{self.cycle_count} completed in {cycle_time:.2f}s")
            
        except Exception as e:
            log_error(f"Error in news cycle: {e}")
    
    def _filter_to_universe(self, raw_news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter news to universe symbols."""
        try:
            universe_set = set(self.universe)
            universe_mask = raw_news_df['symbol'].str.upper().isin(universe_set)
            filtered_df = raw_news_df[universe_mask].copy()
            
            if not filtered_df.empty:
                symbol_count = filtered_df['symbol'].nunique()
                log_info(f"Universe filter: {len(filtered_df)} articles, {symbol_count} symbols")
            
            return filtered_df
            
        except Exception as e:
            log_error(f"Error filtering to universe: {e}")
            return pd.DataFrame()
    
    async def _process_news_pipeline(self, universe_news_df: pd.DataFrame) -> None:
        """Process news through pipeline."""
        try:
            # Filter new articles
            new_articles_df = self.news_processor.filter_new_articles(universe_news_df)
            
            if new_articles_df.empty:
                log_info("No new articles after filtering")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            log_info(f"Processing {len(new_articles_df)} new articles")
            
            # Quality filtering
            quality_filtered_df = self.trader.pre_filter_news(new_articles_df)
            
            if quality_filtered_df is None or quality_filtered_df.empty:
                log_info("All articles filtered by quality")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            # Get prices
            relevant_symbols = quality_filtered_df['symbol'].unique().tolist()
            prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
            
            if prices_df is None or prices_df.empty:
                log_warning(f"No prices for {len(relevant_symbols)} symbols")
                self.news_processor.mark_articles_processed([], universe_news_df)
                return
            
            # Analyze
            analyses = self.news_analyzer.analyze_news_with_technical(
                quality_filtered_df, prices_df
            )
            
            # Mark as processed
            self.news_processor.mark_articles_processed(analyses, universe_news_df)
            
            if analyses:
                # Process signals
                self.trader.process_news_signals(analyses, prices_df)
                self.performance_metrics['total_articles_processed'] += len(analyses)
                self.performance_metrics['fresh_articles_found'] += len(new_articles_df)
            else:
                log_info("No analyses generated")
                
        except Exception as e:
            log_error(f"Error in news pipeline: {e}")
    
    async def _process_position_cycle(self) -> None:
        """Process position monitoring."""
        try:
            active_positions = self.trader.get_active_positions()
            
            if not active_positions:
                return
            
            symbols = list(set(trade.symbol for trade in active_positions))
            prices_df = self.fmp_loader.get_real_time_prices(symbols)
            
            if prices_df is not None and not prices_df.empty:
                self.trader.check_exits(prices_df)
            
        except Exception as e:
            log_error(f"Error in position cycle: {e}")
    
    async def _handle_main_loop_error(self, error: Exception) -> None:
        """Handle main loop errors."""
        self.error_count += 1
        self.performance_metrics['errors_recovered'] += 1
        
        log_error(f"Main loop error #{self.error_count}: {error}")
        
        if self.error_count >= self.MAX_CONSECUTIVE_ERRORS:
            log_error("Too many errors, shutting down")
            self.running = False
            return
        
        backoff_time = min(
            self.ERROR_BACKOFF_BASE ** self.error_count,
            self.MAX_BACKOFF_SECONDS
        )
        
        log_info(f"Backing off {backoff_time}s")
        await asyncio.sleep(backoff_time)
    
    def _print_system_status(self) -> None:
        """Print system status."""
        try:
            uptime_hours = (time.time() - self.performance_metrics['uptime_start']) / 3600
            active_positions = len(self.trader.get_active_positions())
            daily_pnl = self.trader.get_daily_pnl()
            
            trader_stats = self.trader.get_performance_summary()
            news_stats = self.news_processor.get_processing_stats()
            
            log_info("=" * 60)
            log_info("📊 TRADING SYSTEM STATUS")
            log_info("=" * 60)
            
            log_info(f"🎯 System: Uptime {uptime_hours:.1f}h, Cycles {self.performance_metrics['cycles_completed']}")
            log_info(f"💰 Trading: {active_positions} positions, P&L ${daily_pnl:.2f}")
            log_info(f"📰 News: {news_stats.get('cached_articles', 0)} cached, "
                    f"{news_stats.get('new_articles_found', 0)} new found")
            log_info(f"🔍 Analysis: {trader_stats.get('trades_created', 0)} trades, "
                    f"{trader_stats.get('recommendations_logged', 0)} recommendations")
            log_info("=" * 60)
            
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    def _perform_graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        try:
            log_info("Performing graceful shutdown...")
            
            final_stats = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'uptime_hours': (time.time() - self.performance_metrics['uptime_start']) / 3600,
                'performance': self.performance_metrics,
                'universe_size': len(self.universe),
                'active_positions': len(self.trader.get_active_positions())
            }
            
            log_info(f"Final stats: {final_stats}")
            
        except Exception as e:
            log_error(f"Shutdown error: {e}")


async def main() -> int:
    """Main entry point."""
    try:
        system = EnhancedTradingSystem()
        await system.run()
        return 0
        
    except KeyboardInterrupt:
        log_info("Interrupted by user")
        return 0
        
    except Exception as e:
        log_error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)