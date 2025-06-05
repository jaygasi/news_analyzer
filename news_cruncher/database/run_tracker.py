"""
Track last successful run time for dynamic news fetching
Python 3.13.3 compatible
"""
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from config import Config
from utils.simple_logger import log_info, log_error, log_debug, log_warning


class RunTracker:
    """Track application run times for dynamic news fetching"""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize run tracker"""
        self.db_path = db_path or Config.SQLITE_DB_PATH
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize run tracking table"""
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS run_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_type TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        status TEXT NOT NULL,
                        articles_fetched INTEGER DEFAULT 0,
                        articles_processed INTEGER DEFAULT 0,
                        decisions_made INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_run_type_status 
                    ON run_history(run_type, status, end_time)
                ''')
                
                conn.commit()
            
            log_debug("Run tracker database initialized")
            
        except sqlite3.Error as e:
            log_error(f"Failed to initialize run tracker: {e}")
            raise
    
    def get_last_successful_run_time(self) -> datetime:
        """Get the end time of the last successful run with detailed debugging"""
        current_time = datetime.now(timezone.utc)
        
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                # Get the most recent completed run
                cursor = conn.execute('''
                    SELECT id, end_time, start_time FROM run_history 
                    WHERE run_type = 'news_analysis' AND status = 'completed'
                    ORDER BY id DESC 
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                
                if result:
                    run_id, end_time_str, start_time_str = result
                    log_debug(f"Found last run {run_id}: start={start_time_str}, end={end_time_str}")
                    
                    if end_time_str:
                        try:
                            # Parse the stored timestamp
                            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                            
                            # Ensure timezone aware
                            if end_time.tzinfo is None:
                                end_time = end_time.replace(tzinfo=timezone.utc)
                                log_debug(f"Added UTC timezone to stored time: {end_time}")
                            
                            # Sanity check: don't use future times or very recent times
                            time_diff = current_time - end_time
                            
                            if end_time > current_time:
                                log_warning(f"⚠️ Last run time {end_time} is in the future! Using 4 hours ago instead.")
                                return current_time - timedelta(hours=4)
                            elif time_diff.total_seconds() < 300:  # Less than 5 minutes
                                log_warning(f"⚠️ Last run was very recent ({time_diff.total_seconds():.0f}s ago). Using 30 minutes ago to ensure fresh content.")
                                return current_time - timedelta(minutes=30)
                            
                            log_debug(f"Using last successful run time: {end_time} ({time_diff.total_seconds()/3600:.1f} hours ago)")
                            return end_time
                            
                        except Exception as e:
                            log_error(f"Error parsing timestamp '{end_time_str}': {e}")
                    else:
                        log_warning("Last run has no end_time, using start_time")
                        try:
                            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                            if start_time.tzinfo is None:
                                start_time = start_time.replace(tzinfo=timezone.utc)
                            return start_time
                        except Exception as e:
                            log_error(f"Error parsing start_time '{start_time_str}': {e}")
                
                # No successful runs found - use longer default lookback
                default_time = current_time - timedelta(hours=4)  # Increased from 24 to 4
                log_info(f"No successful runs found, using 4-hour lookback: {default_time}")
                return default_time
                    
        except Exception as e:
            log_error(f"Error getting last run time: {e}")
            # Fallback to 4-hour lookback
            return current_time - timedelta(hours=4)
    
    def debug_run_history(self) -> None:
        """Debug method to show recent run history"""
        try:
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    SELECT id, run_type, start_time, end_time, status, articles_fetched, articles_processed, decisions_made
                    FROM run_history 
                    ORDER BY id DESC 
                    LIMIT 5
                ''')
                
                rows = cursor.fetchall()
                log_info("Recent run history (last 5):")
                for row in rows:
                    log_info(f"  Run {row[0]}: {row[1]} | {row[2]} → {row[3]} | {row[4]} | Articles: {row[5]}→{row[6]} | Decisions: {row[7]}")
                    
        except Exception as e:
            log_error(f"Error debugging run history: {e}")
    
    def start_run(self) -> int:
        """Record the start of a new run, return run_id"""
        try:
            current_time = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    INSERT INTO run_history (run_type, start_time, status)
                    VALUES (?, ?, ?)
                ''', ('news_analysis', current_time.isoformat(), 'running'))
                
                run_id = cursor.lastrowid
                conn.commit()
                
                log_debug(f"Started run {run_id} at {current_time}")
                return run_id
                
        except sqlite3.Error as e:
            log_error(f"Error starting run: {e}")
            return 0
    
    def complete_run(self, run_id: int, articles_fetched: int = 0, 
                    articles_processed: int = 0, decisions_made: int = 0) -> None:
        """Mark run as completed with statistics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                conn.execute('''
                    UPDATE run_history 
                    SET end_time = ?, status = ?, articles_fetched = ?, 
                        articles_processed = ?, decisions_made = ?
                    WHERE id = ?
                ''', (
                    current_time.isoformat(),
                    'completed',
                    articles_fetched,
                    articles_processed,
                    decisions_made,
                    run_id
                ))
                
                conn.commit()
                log_debug(f"Completed run {run_id} at {current_time}")
                
        except sqlite3.Error as e:
            log_error(f"Error completing run {run_id}: {e}")
    
    def fail_run(self, run_id: int, error_message: str = '') -> None:
        """Mark run as failed"""
        try:
            current_time = datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                conn.execute('''
                    UPDATE run_history 
                    SET end_time = ?, status = ?
                    WHERE id = ?
                ''', (current_time.isoformat(), f'failed: {error_message[:200]}', run_id))
                
                conn.commit()
                log_debug(f"Failed run {run_id}: {error_message}")
                
        except sqlite3.Error as e:
            log_error(f"Error failing run {run_id}: {e}")
    
    def get_run_statistics(self, days: int = 7) -> dict:
        """Get run statistics for the last N days"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                        SUM(articles_fetched) as total_articles_fetched,
                        SUM(articles_processed) as total_articles_processed,
                        SUM(decisions_made) as total_decisions_made,
                        AVG(CASE WHEN status = 'completed' THEN 
                            (julianday(end_time) - julianday(start_time)) * 24 * 60 
                            ELSE NULL END) as avg_duration_minutes
                    FROM run_history 
                    WHERE start_time >= ?
                ''', (cutoff_time.isoformat(),))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'total_runs': result[0] or 0,
                        'successful_runs': result[1] or 0,
                        'total_articles_fetched': result[2] or 0,
                        'total_articles_processed': result[3] or 0,
                        'total_decisions_made': result[4] or 0,
                        'avg_duration_minutes': round(result[5] or 0, 1),
                        'success_rate': round((result[1] or 0) / max(result[0] or 1, 1) * 100, 1)
                    }
                else:
                    return {}
                    
        except sqlite3.Error as e:
            log_error(f"Error getting run statistics: {e}")
            return {}
    
    def cleanup_old_runs(self, days: int = 30) -> int:
        """Clean up old run records"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            with sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT) as conn:
                cursor = conn.execute('''
                    DELETE FROM run_history 
                    WHERE start_time < ?
                ''', (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    log_info(f"Cleaned up {deleted_count} old run records")
                
                return deleted_count
                
        except sqlite3.Error as e:
            log_error(f"Error cleaning up run history: {e}")
            return 0