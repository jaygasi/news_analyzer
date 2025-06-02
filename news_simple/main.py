"""
Fixed comprehensive news catalyst trading system with simplified deduplication
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
import pandas as pd


@dataclass
class ProcessedArticle:
    """Record of a processed article with simplified tracking"""
    article_hash: str
    symbol: str
    title_hash: str
    processed_time: datetime
    sentiment_score: float
    combined_confidence: float
    was_traded: bool = False
    trade_side: str = ""


class SimplifiedNewsProcessor:
    """Simplified processor focusing on article deduplication only"""
    
    def __init__(self):
        self.processed_articles_file = CONFIG.cache_dir / "processed_articles.json"
        self.processed_hashes: Set[str] = set()
        self.processed_articles: Dict[str, ProcessedArticle] = {}
        
        # Simplified settings - only track articles, not recommendations
        self.max_age_hours = 8  # Shorter window - 8 hours instead of 24
        self.max_articles = 2000  # Smaller cache
        
        self._load_processed_articles()
        self._cleanup_old_articles()
    
    def _generate_article_hash(self, row: pd.Series) -> str:
        """Generate hash that catches true duplicates but allows different articles about same symbol"""
        # Use symbol + normalized title + date for uniqueness
        title = str(row.get('title', '')).strip().lower()
        symbol = str(row.get('symbol', '')).strip().upper()
        
        # Normalize title more aggressively to catch similar articles
        title_cleaned = self._normalize_title_aggressively(title)
        
        # Include date to allow same story on different days
        date_str = str(row.get('publishedDate', ''))[:10]  # Just the date part
        
        content_key = f"{symbol}-{title_cleaned}-{date_str}"
        return hashlib.md5(content_key.encode('utf-8')).hexdigest()
    
    def _normalize_title_aggressively(self, title: str) -> str:
        """More aggressive title normalization to catch real duplicates"""
        import re
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^\s*(breaking|news|update|alert):\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*-\s*(reuters|bloomberg|cnbc|marketwatch|yahoo).*$', '', title, flags=re.IGNORECASE)
        
        # Remove time references that vary
        title = re.sub(r'\b(today|yesterday|this morning|this afternoon|tonight)\b', '', title, flags=re.IGNORECASE)
        
        # Remove excessive whitespace and normalize
        title = ' '.join(title.split())
        
        # Take only first 60 chars of meaningful content
        return title[:60]
    
    def _load_processed_articles(self) -> None:
        """Load processed articles with error handling"""
        if not self.processed_articles_file.exists():
            log_debug("No processed articles cache found, starting fresh")
            return
        
        try:
            with open(self.processed_articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for article_hash, article_data in data.items():
                try:
                    # Handle missing fields for backward compatibility
                    if 'title_hash' not in article_data:
                        article_data['title_hash'] = ""
                    if 'trade_side' not in article_data:
                        article_data['trade_side'] = ""
                    
                    article_data['processed_time'] = datetime.fromisoformat(
                        article_data['processed_time'].replace('Z', '+00:00')
                    )
                    
                    processed_article = ProcessedArticle(**article_data)
                    self.processed_articles[article_hash] = processed_article
                    self.processed_hashes.add(article_hash)
                    
                except Exception as e:
                    log_warning(f"Skipping invalid cached article: {e}")
                    continue
            
            log_info(f"Loaded {len(self.processed_articles)} previously processed articles from cache")
            
        except Exception as e:
            log_warning(f"Error loading processed articles cache: {e}")
            self.processed_articles = {}
            self.processed_hashes = set()
    
    def _save_processed_articles(self) -> None:
        """Save processed articles with atomic write"""
        try:
            data = {}
            for article_hash, article in self.processed_articles.items():
                article_dict = asdict(article)
                article_dict['processed_time'] = article.processed_time.isoformat()
                data[article_hash] = article_dict
            
            temp_file = self.processed_articles_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.processed_articles_file)
            log_debug(f"Saved {len(data)} processed articles to cache")
            
        except Exception as e:
            log_warning(f"Error saving processed articles cache: {e}")
    
    def _cleanup_old_articles(self) -> None:
        """Clean up old articles more aggressively"""
        if not self.processed_articles:
            return
        
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(hours=self.max_age_hours)
        
        # Remove old articles
        old_hashes = [
            article_hash for article_hash, article in self.processed_articles.items()
            if article.processed_time < cutoff_time
        ]
        
        for article_hash in old_hashes:
            self.processed_articles.pop(article_hash, None)
            self.processed_hashes.discard(article_hash)
        
        # Limit total articles
        if len(self.processed_articles) > self.max_articles:
            sorted_articles = sorted(
                self.processed_articles.items(),
                key=lambda x: x[1].processed_time,
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
    
    def filter_truly_new_articles(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out only truly duplicate articles - more permissive"""
        if news_df is None or news_df.empty:
            return news_df
        
        new_articles = []
        skipped_count = 0
        
        for idx, row in news_df.iterrows():
            article_hash = self._generate_article_hash(row)
            
            # Only check exact hash match - no similarity checking
            if article_hash not in self.processed_hashes:
                new_articles.append(row)
            else:
                skipped_count += 1
                log_debug(f"Skipping exact duplicate: {row.get('symbol', 'UNKNOWN')} - {str(row.get('title', ''))[:50]}...")
        
        if new_articles:
            new_df = pd.DataFrame(new_articles).reset_index(drop=True)
            log_info(f"Article filtering: {len(news_df)} total -> {len(new_df)} new articles "
                    f"(skipped {skipped_count} exact duplicates)")
            return new_df
        else:
            log_info(f"No new articles found ({skipped_count} exact duplicates)")
            return pd.DataFrame()
    
    def mark_articles_processed(self, analyses: List, original_news_df: pd.DataFrame) -> None:
        """Mark articles as processed after analysis"""
        current_time = datetime.now(timezone.utc)
        new_processed_count = 0
        
        # Create analysis lookup
        analysis_map = {}
        for analysis in analyses:
            key = analysis.symbol
            if key not in analysis_map or analysis.combined_confidence > analysis_map[key].combined_confidence:
                analysis_map[key] = analysis
        
        for idx, row in original_news_df.iterrows():
            article_hash = self._generate_article_hash(row)
            
            # Skip if already processed
            if article_hash in self.processed_hashes:
                continue
            
            symbol = str(row.get('symbol', '')).strip().upper()
            title = str(row.get('title', ''))
            analysis = analysis_map.get(symbol)
            
            processed_article = ProcessedArticle(
                article_hash=article_hash,
                symbol=symbol,
                title_hash=self._normalize_title_aggressively(title.lower()),
                processed_time=current_time,
                sentiment_score=analysis.sentiment_score if analysis else 0.0,
                combined_confidence=analysis.combined_confidence if analysis else 0.0,
                was_traded=analysis.combined_confidence >= CONFIG.min_confidence_score if analysis else False,
                trade_side=self._determine_trade_side(analysis.sentiment_score) if analysis else ""
            )
            
            self.processed_articles[article_hash] = processed_article
            self.processed_hashes.add(article_hash)
            new_processed_count += 1
        
        if new_processed_count > 0:
            log_info(f"Marked {new_processed_count} new articles as processed")
            self._save_processed_articles()
    
    def _determine_trade_side(self, sentiment_score: float) -> str:
        """Determine trade side from sentiment"""
        if sentiment_score > 0.25:
            return "long"
        elif sentiment_score < -0.25:
            return "short"
        return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processed_articles:
            return {
                'total_processed': 0,
                'high_confidence_signals': 0,
                'actual_trades': 0,
                'cache_age_hours': 0
            }
        
        current_time = datetime.now(timezone.utc)
        
        high_confidence_count = sum(
            1 for article in self.processed_articles.values()
            if article.combined_confidence >= CONFIG.min_confidence_score
        )
        
        trades_count = sum(
            1 for article in self.processed_articles.values()
            if article.was_traded
        )
        
        oldest_article = min(
            self.processed_articles.values(),
            key=lambda x: x.processed_time,
            default=None
        )
        
        cache_age_hours = 0
        if oldest_article:
            cache_age_hours = (current_time - oldest_article.processed_time).total_seconds() / 3600
        
        return {
            'total_processed': len(self.processed_articles),
            'high_confidence_signals': high_confidence_count,
            'actual_trades': trades_count,
            'cache_age_hours': cache_age_hours
        }


class ComprehensiveTradingSystem:
    """Simplified comprehensive trading system with better deduplication logic"""
    
    def __init__(self) -> None:
        """Initialize trading system"""
        api_key = CONFIG.get_api_key('fmp')
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        
        self.fmp_api_key: str = api_key
        self._initialize_components()
        
        # Use simplified processor
        self.news_processor = SimplifiedNewsProcessor()
        
        self.universe: List[str] = []
        self.running = True
        self._setup_signal_handlers()
    
    def _initialize_components(self) -> None:
        """Initialize system components"""
        try:
            self.fmp_loader = SimpleFMPLoader(self.fmp_api_key)
            self.news_analyzer = EnhancedNewsAnalyzer(self.fmp_loader)
            self.trader = EnhancedTrader(self.fmp_loader)
            log_info("All system components initialized successfully")
        except Exception as e:
            log_error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully"""
        log_info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize_universe(self) -> bool:
        """Initialize trading universe"""
        log_info("Initializing comprehensive trading universe...")
        
        try:
            screener_df = self.fmp_loader.get_stock_screener(CONFIG.max_symbols)
            
            if screener_df is not None and not screener_df.empty:
                self.universe = screener_df['symbol'].str.upper().tolist()
                log_info(f"Loaded {len(self.universe)} valid symbols in universe")
                
                sample_symbols = self.universe[:20]
                log_info(f"Sample universe symbols: {sample_symbols}")
                
                return len(self.universe) > 0
            else:
                log_error("Failed to load trading universe - no data returned")
                return False
                
        except Exception as e:
            log_error(f"Error initializing universe: {e}")
            return False
    
    def process_comprehensive_news_cycle(self) -> None:
        """Simplified news processing - filter duplicates at source"""
        try:
            # Get raw news
            raw_news_df = self.fmp_loader.get_news_rss()
            if raw_news_df is None or raw_news_df.empty:
                log_warning("No news data received")
                return
            
            # Filter to universe symbols
            universe_set = set(self.universe)
            filtered_news_df = raw_news_df[
                raw_news_df['symbol'].str.upper().isin(universe_set)
            ].copy()
            
            if filtered_news_df.empty:
                log_info("No news for universe symbols")
                return
            
            log_info(f"Raw news: {len(raw_news_df)} articles, "
                    f"filtered to universe: {len(filtered_news_df)} articles")
            
            # SIMPLIFIED: Only filter truly duplicate articles
            new_articles_df = self.news_processor.filter_truly_new_articles(filtered_news_df)
            
            if new_articles_df.empty:
                log_info("No new articles to process (all were exact duplicates)")
                return
            
            log_info(f"[SIMPLIFIED] Processing {len(new_articles_df)} truly new articles")
            
            # Process new articles
            self._process_filtered_news(new_articles_df)
                
        except Exception as e:
            log_error(f"Error in comprehensive news cycle: {e}")
    
    def _process_filtered_news(self, filtered_news_df: pd.DataFrame) -> None:
        """Process filtered news with simplified logic"""
        # Apply quality filters
        quality_filtered_df = self.trader.pre_filter_news(filtered_news_df)
        if quality_filtered_df is None or quality_filtered_df.empty:
            log_info("All new articles filtered out by quality filters")
            self.news_processor.mark_articles_processed([], filtered_news_df)
            return
        
        log_info(f"Quality filtered news: {len(quality_filtered_df)} articles")
        
        # Get current prices
        relevant_symbols = quality_filtered_df['symbol'].unique().tolist()
        prices_df = self.fmp_loader.get_real_time_prices(relevant_symbols)
        
        if prices_df is None or prices_df.empty:
            log_warning("Failed to get current prices for technical analysis")
            self.news_processor.mark_articles_processed([], filtered_news_df)
            return
        
        # Enhanced news analysis
        analyses = self.news_analyzer.analyze_news_with_technical(
            quality_filtered_df, prices_df
        )
        
        # Mark articles as processed
        self.news_processor.mark_articles_processed(analyses, filtered_news_df)
        
        if not analyses:
            log_info("No valid news analyses generated")
            return
        
        # Process analysis results WITHOUT additional symbol-level filtering
        self._process_analysis_results(analyses, prices_df)
    
    def _process_analysis_results(self, analyses, prices_df) -> None:
        """Process analysis results with simplified deduplication"""
        high_confidence = self.news_analyzer.filter_high_confidence(
            analyses, use_combined_confidence=True
        )
        
        if high_confidence:
            log_info(f"[SIGNALS] Found {len(high_confidence)} high-quality trading opportunities")
            
            # Process signals WITHOUT additional symbol-level deduplication
            # Let the trader handle any necessary deduplication
            self.trader.process_news_signals(high_confidence, prices_df)
        else:
            log_info("No high-confidence signals after filtering")
    
    def check_positions_cycle(self) -> None:
        """Check positions with enhanced error handling"""
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
        """Print enhanced system status"""
        try:
            active_positions = len(self.trader.get_active_positions())
            daily_pnl = self.trader.get_daily_pnl()
            
            # Get processing stats
            processing_stats = self.news_processor.get_processing_stats()
            
            log_info("[STATUS] System Status:")
            log_info(f"   Active Positions: {active_positions}")
            log_info(f"   Daily P&L: ${daily_pnl:.2f}")
            log_info(f"   Confidence Threshold: {CONFIG.min_confidence_score}")
            
            # Simplified processing stats
            log_info("[PROCESSING] Article Statistics:")
            log_info(f"   Total Articles Processed: {processing_stats['total_processed']}")
            log_info(f"   High Confidence Signals: {processing_stats['high_confidence_signals']}")
            log_info(f"   Actual Trades: {processing_stats['actual_trades']}")
            log_info(f"   Cache Age: {processing_stats['cache_age_hours']:.1f} hours")
                
        except Exception as e:
            log_error(f"Error printing status: {e}")
    
    async def run(self) -> None:
        """Main run loop"""
        log_info("[START] Starting SIMPLIFIED News Catalyst Trading System")
        
        if not self.initialize_universe():
            log_error("Failed to initialize trading universe")
            return
        
        await self._main_trading_loop()
        log_info("[STOP] Trading system stopped")
    
    async def _main_trading_loop(self) -> None:
        """Main trading loop"""
        last_news_check = 0.0
        last_position_check = 0.0
        last_status_print = 0.0
        status_interval = 300  # 5 minutes
        error_count = 0
        max_errors = 10
        
        log_info("[TARGET] Trading system running... Press Ctrl+C to stop")
        
        while self.running:
            try:
                current_time = time.time()
                
                # News processing cycle
                if current_time - last_news_check >= CONFIG.news_check_interval:
                    self.process_comprehensive_news_cycle()
                    last_news_check = current_time
                    error_count = 0
                
                # Position monitoring cycle
                if current_time - last_position_check >= CONFIG.price_check_interval:
                    self.check_positions_cycle()
                    last_position_check = current_time
                
                # Status reporting cycle
                if current_time - last_status_print >= status_interval:
                    self.print_comprehensive_status()
                    last_status_print = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                error_count += 1
                log_error(f"Error in main loop (count: {error_count}): {e}")
                
                if error_count >= max_errors:
                    log_error(f"Too many errors ({error_count}), shutting down")
                    self.running = False
                else:
                    sleep_time = min(2 ** error_count, 60)
                    log_info(f"Sleeping {sleep_time}s before retry")
                    await asyncio.sleep(sleep_time)


async def main() -> int:
    """Main entry point"""
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