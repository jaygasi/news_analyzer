"""
SQLite database for tracking processed articles
Python 3.13.3 compatible
"""
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import Config
from utils.simple_logger import log_info, log_error, log_debug


class ArticleTracker:
    """SQLite-based article tracking to prevent reprocessing"""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize article tracker with SQLite database"""
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS processed_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_hash TEXT UNIQUE NOT NULL,
                        ticker TEXT NOT NULL,
                        title TEXT NOT NULL,
                        url TEXT,
                        processed_date TIMESTAMP NOT NULL,
                        decision TEXT,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_article_hash 
                    ON processed_articles(article_hash)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ticker_date 
                    ON processed_articles(ticker, processed_date)
                ''')
                
                conn.commit()
            
            log_info("Article tracker database initialized successfully")
            
        except sqlite3.Error as e:
            log_error(f"Failed to initialize database: {e}")
            raise
    
    def generate_article_hash(self, article: Dict[str, Any]) -> str:
        """Generate unique hash for article based on key fields"""
        try:
            # Use ticker, title, and URL for uniqueness
            ticker = str(article.get('symbol', ''))
            title = str(article.get('title', ''))
            url = str(article.get('url', ''))
            
            # Create hash input
            hash_input = f"{ticker}|{title}|{url}".encode('utf-8')
            return hashlib.sha256(hash_input).hexdigest()
            
        except Exception as e:
            log_error(f"Error generating article hash: {e}")
            # Fallback to timestamp-based hash
            return hashlib.sha256(str(datetime.now()).encode()).hexdigest()
    
    def is_article_processed(self, article: Dict[str, Any]) -> bool:
        """Check if article has already been processed"""
        article_hash = self.generate_article_hash(article)
        
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute(
                    'SELECT 1 FROM processed_articles WHERE article_hash = ?',
                    (article_hash,)
                )
                return cursor.fetchone() is not None
                
        except sqlite3.Error as e:
            log_error(f"Error checking processed article: {e}")
            return False
    
    def mark_article_processed(self, article: Dict[str, Any], 
                             decision: str = '', confidence: float = 0.0) -> bool:
        """Mark article as processed with optional decision data"""
        article_hash = self.generate_article_hash(article)
        
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO processed_articles 
                    (article_hash, ticker, title, url, processed_date, decision, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article_hash,
                    str(article.get('symbol', '')),
                    str(article.get('title', ''))[:500],  # Limit title length
                    str(article.get('url', ''))[:1000],    # Limit URL length
                    datetime.now(timezone.utc),
                    decision,
                    confidence
                ))
                conn.commit()
            
            return True
            
        except sqlite3.Error as e:
            log_error(f"Error marking article as processed: {e}")
            return False
    
    def filter_unprocessed_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out already processed articles"""
        if not articles:
            return []
        
        unprocessed = []
        processed_count = 0
        
        for article in articles:
            if not self.is_article_processed(article):
                unprocessed.append(article)
            else:
                processed_count += 1
        
        log_info(f"Filtered articles: {len(articles)} total, {len(unprocessed)} new, {processed_count} already processed")
        return unprocessed
    
    def get_processed_count_by_ticker(self, ticker: str, days: int = 7) -> int:
        """Get count of processed articles for ticker in last N days"""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp() - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM processed_articles 
                    WHERE ticker = ? AND processed_date >= datetime(?, 'unixepoch')
                ''', (ticker, cutoff_date))
                
                result = cursor.fetchone()
                return result[0] if result else 0
                
        except sqlite3.Error as e:
            log_error(f"Error getting processed count for {ticker}: {e}")
            return 0
    
    def cleanup_old_articles(self, days: int = 30) -> int:
        """Remove articles older than specified days"""
        try:
            cutoff_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp() - (days * 24 * 3600)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    DELETE FROM processed_articles 
                    WHERE processed_date < datetime(?, 'unixepoch')
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
            
            if deleted_count > 0:
                log_info(f"Cleaned up {deleted_count} old articles")
            
            return deleted_count
            
        except sqlite3.Error as e:
            log_error(f"Error cleaning up old articles: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                # Total articles
                total_cursor = conn.execute('SELECT COUNT(*) FROM processed_articles')
                total_articles = total_cursor.fetchone()[0]
                
                # Articles with decisions
                decisions_cursor = conn.execute('''
                    SELECT COUNT(*) FROM processed_articles 
                    WHERE decision != ""
                ''')
                articles_with_decisions = decisions_cursor.fetchone()[0]
                
                # Unique tickers
                tickers_cursor = conn.execute('''
                    SELECT COUNT(DISTINCT ticker) FROM processed_articles
                ''')
                unique_tickers = tickers_cursor.fetchone()[0]
                
                # Recent articles (last 24 hours)
                recent_cursor = conn.execute('''
                    SELECT COUNT(*) FROM processed_articles 
                    WHERE processed_date >= datetime('now', '-1 day')
                ''')
                recent_articles = recent_cursor.fetchone()[0]
                
                return {
                    'total_articles': total_articles,
                    'articles_with_decisions': articles_with_decisions,
                    'unique_tickers': unique_tickers,
                    'recent_articles_24h': recent_articles
                }
                
        except sqlite3.Error as e:
            log_error(f"Error getting statistics: {e}")
            return {}