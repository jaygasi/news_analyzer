"""
Main trading system using modular components with enhanced news processing
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
from config import CONFIG
from utils.simple_logger import log_info, log_error, log_warning, log_debug
from data_loaders.simple_fmp_loader import SimpleFMPLoader
from analysis.enhanced_news_analyzer import EnhancedNewsAnalyzer
from trading.enhanced_trader import EnhancedTrader
from trading.trade_models import ProcessedArticle
import pandas as pd


class NewsProcessor:
    """Process and deduplicate news articles with performance optimizations"""
    
    def __init__(self):
        self.processed_articles_file = CONFIG.cache_dir / "processed_articles.json"
        self.processed_hashes: Set[str] = set()
        self.processed_articles: Dict[str, ProcessedArticle] = {}
        
        # Enhanced configuration for better news coverage
        self.max_age_hours = 8  # Increased from 6 for better coverage
        self.max_articles = 2000  # Increased capacity
        self.dedup_window_hours = 2  # Time window for deduplication
        
        self._load_processed_articles()
        self._cleanup_old_articles()
    
    def _generate_article_hash(self, row: pd.Series) -> str:
        """Generate hash for article deduplication with improved normalization."""
        title = str(row.get('title', '')).strip().lower()
        symbol = str(row.get('symbol', '')).strip().upper()
        
        title_cleaned = self._normalize_title(title)
        date_str = str(row.get('publishedDate', ''))[:10]
        
        # Include hour for better deduplication of updated articles
        hour_str = str(row.get('publishedDate', ''))[:13]  # YYYY-MM-DDTHH
        
        content_key = f"{symbol}-{title_cleaned}-{hour_str}"
        return hashlib.md5(content_key.encode('utf-8')).hexdigest()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication with enhanced cleaning."""
        import re
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^\s*(breaking|news|update|alert):\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*-\s*(reuters|bloomberg|cnbc|marketwatch|yahoo|benzinga|seeking alpha).*$', '', title, flags=re.IGNORECASE)
        
        # Remove time references that change frequently
        title = re.sub(r'\b(today|yesterday|this morning|this afternoon|tonight|just now|moments ago)\b', '', title, flags=re.IGNORECASE)
        
        # Remove specific times and dates
        title = re.sub(r'\b\d{1,2}:\d{2}\s*(AM|PM|EST|EDT|PST|PDT)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', title, flags=re.IGNORECASE)
        
        # Remove update indicators
        title = re.sub(r'\b(update|updated|revision|revised)\b', '', title, flags=re.IGNORECASE)
        
        # Normalize whitespace and truncate
        title = ' '.join(title.split())
        return title[:80]  # Increased from 60 for better accuracy
    
    def _load_processed_articles(self) -> None:
        """Load processed articles from cache with error handling."""
        if not self.processed_articles_file.exists():
            log_debug("No processed articles cache found, starting fresh")
            return
        
        try:
            with open(self.processed_articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_count = 0
            for article_hash, article_data in data.items():
                try:
                    processed_article = ProcessedArticle.from_dict(article_data)
                    self.processed_articles[article_hash] = processed_article
                    self.processed_hashes.add(article_hash)
                    loaded_count += 1
                    
                except Exception as e:
                    log_warning(f"Skipping invalid cached article: {e}")
                    continue
            
            log_info(f"Loaded {loaded_count} previously processed articles from cache")
            
        except Exception as e:
            log_warning(f"Error loading processed articles cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def _save_processed_articles(self) -> None:
        """Save processed articles to cache with atomic write."""
        try:
            data = {}
            for article_hash, article in self.processed_articles.items():
                data[article_hash] = article.to_dict()
            
            temp_file = self.processed_articles_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_file.replace(self.processed_articles_file)
            log_debug(f"Saved {len(data)} processed articles to cache")
            
        except Exception as e:
            log_error(f"Error saving processed articles cache: {e}")
    
    def _cleanup_old_articles(self) -> None:
        """Clean up old articles from cache with improved logic."""
        if not self.processed_articles:
            return
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=self.max_age_hours)
        
        # Remove old articles
        old_hashes = []
        for article_hash, article in self.processed_articles.items():
            try:
                article_time = article.processed_datetime
                if article_time < cutoff_time:
                    old_hashes.append(article_hash)
            except Exception:
                old_hashes.append(article_hash)
        
        for article_hash in old_hashes:
            self.processed_articles.pop(article_hash, None)
            self.processed_hashes.discard(article_hash)
        
        # Limit total articles by keeping most recent and highest confidence
        if len(self.processed_articles) > self.max_articles:
            # Sort by combination of time and confidence
            sorted_articles = sorted(
                self.processed_articles.items(),
                key=lambda x: (x[1].processed_datetime.timestamp(), x[1].combined_confidence),
                reverse=True
            )
            
            articles_to_keep = dict(sorted_articles[:self.max_articles])
            articles_to_remove = set(self.processed_articles.keys()) - set(articles_to_keep.keys())
            
            for article_hash in articles_to_remove:
                self.processed_articles.pop(article_hash, None)
                self.processed_hashes.discard(article_hash)
        
        if old_hashes:
            self.processed_hashes = set(self.processed_articles.keys())
            log_info(f"Cleaned up {len(old_hashes)} old articles, "
                    f"now tracking {len(self.processed_articles)} articles")
            self._save_processed_articles()
    
    def filter_new_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out previously processed articles with enhanced deduplication."""
        if news_df is None or news_df.empty:
            return news_df
        
        new_articles = []
        skipped_count = 0
        duplicate_details = {}
        
        for idx, row in news_df.iterrows():
            article_hash = self._generate_article_hash(row)
            symbol = str(row.get('symbol', 'UNKNOWN'))
            
            if article_hash not in self.processed_hashes:
                # Additional check for very recent duplicates within the current batch
                if not self._is_recent_duplicate(row, new_articles):
                    new_articles.append(row)
                else:
                    skipped_count += 1
                    duplicate_details[symbol] = duplicate_details.get(symbol, 0) + 1
            else:
                skipped_count += 1
                duplicate_details[symbol] = duplicate_details.get(symbol, 0) + 1
                log_debug(f"Skipping cached duplicate: {symbol} - {str(row.get('title', ''))[:50]}...")
        
        if new_articles:
            new_df = pd.DataFrame(new_articles).reset_index(drop=True)
            log_info(f"Article filtering: {len(news_df)} total -> {len(new_df)} new articles "
                    f"(skipped {skipped_count} duplicates)")
            
            if duplicate_details:
                top_duplicates = sorted(duplicate_details.items(), key=lambda x: x[1], reverse=True)[:5]
                log_debug(f"Top duplicate symbols: {dict(top_duplicates)}")
            
            return new_df
        else:
            log_info(f"No new articles found ({skipped_count} duplicates)")
            return pd.DataFrame()
    
    def _is_recent_duplicate(self, current_row: pd.Series, existing_articles: List) -> bool:
        """Check if article is duplicate of recently processed articles in current batch."""
        if not existing_articles:
            return False
        
        current_symbol = str(current_row.get('symbol', '')).upper()
        current_title = self._normalize_title(str(current_row.get('title', '')).lower())
        
        # Only check last 10 articles for performance
        recent_articles = existing_articles[-10:] if len(existing_articles) > 10 else existing_articles
        
        for existing_row in recent_articles:
            existing_symbol = str(existing_row.get('symbol', '')).upper()
            existing_title = self._normalize_title(str(existing_row.get('title', '')).lower())
            
            if (current_symbol == existing_symbol and 
                self._titles_similar(current_title, existing_title)):
                return True
        
        return False
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar using word overlap."""
        words1 = set(word for word in title1.split() if len(word) > 2)
        words2 = set(word for word in title2.split() if len(word) > 2)
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return (intersection / union) > threshold if union > 0 else False
    
    def mark_articles_processed(self, analyses: List, original_news_df: pd.DataFrame) -> None:
        """Mark articles as processed with enhanced metadata."""
        current_time = datetime.now(timezone.utc).isoformat()
        new_processed_count = 0
        
        # Create analysis lookup with better handling
        analysis_map = {}
        for analysis in analyses:
            key = analysis.symbol
            if key not in analysis_map or analysis.combined_confidence > analysis_map[key].combined_confidence:
                analysis_map[key] = analysis
        
        for idx, row in original_news_df.iterrows():
            article_hash = self._generate_article_hash(row)
            
            if article_hash in self.processed_hashes:
                continue
            
            symbol = str(row.get('symbol', '')).strip().upper()
            title = str(row.get('title', ''))
            analysis = analysis_map.get(symbol)
            
            processed_article = ProcessedArticle(
                article_hash=article_hash,
                symbol=symbol,
                title_hash=self._normalize_title(title.lower()),
                processed_time=current_time,
                sentiment_score=analysis.sentiment_score if analysis else 0.0,
                combined_confidence=analysis.combined_confidence if analysis else 0.0,
                was_traded=1 if (analysis and analysis.combined_confidence >= CONFIG.min_confidence_score) else 0,
                trade_side=self._determine_trade_side(analysis.sentiment_score) if analysis else ""
            )
            
            self.processed_articles[article_hash] = processed_article
            self.processed_hashes.add(article_hash)
            new_processed_count += 1
        
        if new_processed_count > 0:
            log_info(f"Marked {new_processed_count} new articles as processed")
            self._save_processed_articles()
    
    def _determine_trade_side(self, sentiment_score: float) -> str:
        """Determine trade side from sentiment."""
        if sentiment_score > 0.20:
            return "long"
        elif sentiment_score < -0.20:
            return "short"
        return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with enhanced metrics."""
        if not self.processed_articles:
            return {
                'total_processed': 0,
                'high_confidence_signals': 0,
                'actual_trades': 0,
                'cache_age_hours': 0,
                'symbols_covered': 0,
                'avg_confidence': 0.0
            }
        
        current_time = datetime.now(timezone.utc)
        
        high_confidence_count = sum(
            1 for article in self.processed_articles.values()
            if article.combined_confidence >= CONFIG.min_confidence_score
        )
        
        trades_count = sum(
            1 for article in self.processed_articles.values()
            if article.was_traded == 1
        )
        
        # Unique symbols covered
        symbols_covered = len(set(article.symbol for article in self.processed_articles.values()))
        
        # Average confidence
        confidences = [article.combined_confidence for article in self.processed_articles.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Find oldest article
        oldest_time = current_time
        for article in self.processed_articles.values():
            try:
                article_time = article.processed_datetime
                if article_time < oldest_time:
                    oldest_time = article_time
            except Exception:
                continue
        
        cache_age_hours = (current_time - oldest_time).total_seconds() / 3600 if oldest_time < current_time else 0
        
        return {
            'total_processed': len(self.processed_articles),
            'high_confidence_signals': high_confidence_count,
            'actual_trades': trades_count,
            'cache_age_hours': cache_age_hours,
            'symbols_covered': symbols_covered,
            'avg_confidence': avg_confidence
        }


class TradingSystem:
    """Main trading system using modular components with enhanced news processing"""
    
    def __init__(self) -> None:
        """Initialize trading system."""
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        self.fmp_api_key: str = api_key
        self._initialize_components()
        
        self.news_processor = NewsProcessor()
        self.universe: List[str] = []
        self.running = True
        self._setup_signal_handlers()
        
        # Performance tracking
        self.cycle_count = 0
        self.last_universe_refresh = 0.0
        self.universe_refresh_interval = 3600  # 1 hour
    
    def _initialize_components(self) -> None:
        """Initialize system components."""
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
        """Initialize trading universe with caching."""
        log_info("Initializing trading universe...")
        
        try:
            screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
            
            if screener_df is not None and not screener_df.empty:
                # Store universe with additional filtering
                universe_candidates = screener_df['symbol'].str.upper().tolist()
                
                # Additional filtering for news-heavy symbols
                self.universe = [symbol for symbol in universe_candidates 
                               if len(symbol) <= 5 and symbol.isalpha()]
                
                log_info(f"Loaded {len(self.universe)} valid symbols in universe")
                
                sample_symbols = self.universe[:20]
                log_info(f"Sample universe symbols: {sample_symbols}")
                
                self.last_universe_refresh = time.time()
                return len(self.universe) > 0
            else:
                log_error("Failed to load trading universe - no data returned")
                return False
                
        except Exception as e:
            log_error(f"Error initializing universe: {e}")
            return False
    
    def _refresh_universe_if_needed(self) -> None:
        """Refresh universe periodically for better coverage."""
        if time.time() - self.last_universe_refresh > self.universe_refresh_interval:
            log_info("Refreshing trading universe...")
            self.initialize_universe()
    
    def process_news_cycle(self) -> None:
        """Process news cycle with enhanced performance and logging."""
        cycle_start = time.time()
        self.cycle_count += 1
        
        try:
            # Refresh universe periodically
            if self.cycle_count % 10 == 0:  # Every 10th cycle
                self._refresh_universe_if_needed()
            
            # Get raw news with enhanced fetching
            log_debug(f"Starting news cycle #{self.cycle_count}")
            raw_news_df = self.fmp_loader.get_news_rss()
            
            if raw_news_df is None or raw_news_df.empty:
                log_warning("No news data received from FMP API")
                return
            
            log_info(f"Retrieved {len(raw_news_df)} raw news articles from FMP")
            
            # Filter to universe with performance optimization
            universe_set = set(self.universe)
            filtered_news_df = raw_news_df[
                raw_news_df['symbol'].str.upper().isin(universe_set)
            ].copy()
            
            if filtered_news_df.empty:
                log_info("No news articles match universe symbols")
                return
            
            universe_coverage = len(filtered_news_df['symbol'].unique())
            log_info(f"News coverage: {len(filtered_news_df)} articles across {universe_coverage} symbols")
            
            # Filter new articles
            new_articles_df = self.news_processor.filter_new_articles(filtered_news_df)
            
            if new_articles_df.empty:
                log_info("No new articles to process (all were duplicates)")
                return
            
            log_info(f"Processing {len(new_articles_df)} new articles")
            
            # Process new articles
            self._process_filtered_news(new_articles_df)
            
            cycle_time = time.time() - cycle_start
            log_debug(f"News cycle #{self.cycle_count} completed in {cycle_time:.2f}s")
                
        except Exception as e:
            log_error(f"Error in news cycle #{self.cycle_count}: {e}")
    
    def _process_filtered_news(self, filtered_news_df: pd.DataFrame) -> None:
        """Process filtered news through the analysis pipeline with enhanced logging."""
        processing_start = time.time()
        
        # Apply quality filters
        quality_filtered_df = self.trader.pre_filter_news(filtered_news_df)
        if quality_filtered_df is None or quality_filtered_df.empty:
            log_info("All new articles filtered out by quality filters")
            self.news_processor.mark_articles_processed([], filtered_news_df)
            return
        
        quality_filter_time = time.time() - processing_start
        log_debug(f"Quality filtering completed in {quality_filter_time:.2f}s: {len(quality_filtered_df)} articles")
        
        # Get current prices for relevant symbols
        relevant_symbols = quality_filtered_df['symbol'].unique().tolist()
        prices_start = time.time()
        prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
        prices_time = time.time() - prices_start
        
        if prices_df is None or prices_df.empty:
            log_warning(f"Failed to get current prices for {len(relevant_symbols)} symbols")
            self.news_processor.mark_articles_processed([], filtered_news_df)
            return
        
        log_debug(f"Price fetching completed in {prices_time:.2f}s for {len(prices_df)} symbols")
        
        # Enhanced news analysis
        analysis_start = time.time()
        analyses = self.news_analyzer.analyze_news_with_technical(
            quality_filtered_df, prices_df
        )
        analysis_time = time.time() - analysis_start
        
        # Mark articles as processed
        self.news_processor.mark_articles_processed(analyses, filtered_news_df)
        
        if not analyses:
            log_info("No valid news analyses generated")
            return
        
        log_debug(f"News analysis completed in {analysis_time:.2f}s for {len(analyses)} articles")
        
        # Process analysis results
        self._process_analysis_results(analyses, prices_df)
        
        total_time = time.time() - processing_start
        log_debug(f"Total news processing time: {total_time:.2f}s")
    
    def _process_analysis_results(self, analyses, prices_df) -> None:
        """Process analysis results with enhanced metrics."""
        # Get confidence distribution
        confidences = [a.combined_confidence for a in analyses]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        
        high_confidence = self.news_analyzer.filter_high_confidence(
            analyses, use_combined_confidence=True
        )
        
        log_info(f"Analysis results: {len(analyses)} total, avg_conf={avg_confidence:.3f}, "
                f"max_conf={max_confidence:.3f}, high_conf={len(high_confidence)}")
        
        if high_confidence:
            # Log top signals
            top_signals = sorted(high_confidence, key=lambda x: x.combined_confidence, reverse=True)[:3]
            for i, signal in enumerate(top_signals, 1):
                log_info(f"Top signal #{i}: {signal.symbol} conf={signal.combined_confidence:.3f} "
                        f"sentiment={signal.sentiment_score:.3f} topic={signal.topic}")
            
            self.trader.process_news_signals(high_confidence, prices_df)
        else:
            log_info("No high-confidence signals after filtering")
    
    def check_positions_cycle(self) -> None:
        """Check positions cycle with enhanced monitoring."""
        try:
            active_trades = self.trader.get_active_positions()
            if not active_trades:
                return
            
            symbols = [trade.symbol for trade in active_trades]
            prices_df = self.fmp_loader.get_real_time_prices(symbols)
            
            if prices_df is not None and not prices_df.empty:
                self.trader.check_exits(prices_df)
            else:
                log_warning(f"Failed to get prices for {len(symbols)} active positions")
                
        except Exception as e:
            log_error(f"Error checking positions: {e}")
    
    def print_system_status(self) -> None:
        """Print comprehensive system status with enhanced metrics."""
        try:
            active_positions = len(self.trader.get_active_positions())
            daily_pnl = self.trader.get_daily_pnl()
            
            # Get processing stats
            processing_stats = self.news_processor.get_processing_stats()
            
            log_info("[STATUS] System Status:")
            log_info(f"   Cycle Count: {self.cycle_count}")
            log_info(f"   Universe Size: {len(self.universe)}")
            log_info(f"   Active Positions: {active_positions}")
            log_info(f"   Daily P&L: ${daily_pnl:.2f}")
            log_info(f"   Confidence Threshold: {CONFIG.min_confidence_score}")
            
            log_info("[NEWS] Article Processing Statistics:")
            log_info(f"   Total Articles Processed: {processing_stats['total_processed']}")
            log_info(f"   Symbols Covered: {processing_stats['symbols_covered']}")
            log_info(f"   Average Confidence: {processing_stats['avg_confidence']:.3f}")
            log_info(f"   High Confidence Signals: {processing_stats['high_confidence_signals']}")
            log_info(f"   Actual Trades: {processing_stats['actual_trades']}")
            log_info(f"   Cache Age: {processing_stats['cache_age_hours']:.1f} hours")
            
            # News processing efficiency
            if processing_stats['total_processed'] > 0:
                trade_rate = processing_stats['actual_trades'] / processing_stats['total_processed']
                signal_rate = processing_stats['high_confidence_signals'] / processing_stats['total_processed']
                log_info(f"   Signal Rate: {signal_rate:.1%}")
                log_info(f"   Trade Rate: {trade_rate:.1%}")
                
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    async def run(self) -> None:
        """Main run loop."""
        log_info("[START] Starting Enhanced News Catalyst Trading System")
        log_info(f"[CONFIG] News limits: {CONFIG.news_page_limit} pages, "
                f"{CONFIG.news_per_page_limit} per page, "
                f"{CONFIG.max_total_news_articles} max total")
        
        if not self.initialize_universe():
            log_error("Failed to initialize trading universe")
            return
        
        await self._main_trading_loop()
        log_info("[STOP] Trading system stopped")
    
    async def _main_trading_loop(self) -> None:
        """Main trading loop with enhanced performance monitoring."""
        last_news_check = 0.0
        last_position_check = 0.0
        last_status_print = 0.0
        status_interval = 180  # 3 minutes - more frequent for better monitoring
        error_count = 0
        max_errors = 8
        
        log_info("[TARGET] Enhanced trading system running... Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # News processing cycle
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    self.process_news_cycle()
                    last_news_check = current_time
                    error_count = 0
                
                # Position monitoring cycle
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    self.check_positions_cycle()
                    last_position_check = current_time
                
                # Status reporting cycle
                if current_time - last_status_print >= status_interval:
                    self.print_system_status()
                    last_status_print = current_time
                
                await asyncio.sleep(0.5)  # Reduced sleep for better responsiveness
                
            except Exception as e:
                error_count += 1
                log_error(f"Error in main loop (count: {error_count}): {e}")
                
                if error_count >= max_errors:
                    log_error(f"Too many errors ({error_count}), shutting down")
                    self.running = False
                else:
                    sleep_time = min(2 ** error_count, 45)
                    log_info(f"Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)


async def main() -> int:
    """Main entry point."""
    try:
        system = TradingSystem()
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