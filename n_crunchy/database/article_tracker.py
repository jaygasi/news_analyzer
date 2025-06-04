import sqlite3
import os
from datetime import datetime
from config import ARTICLE_TRACKER_DB
from utils.utils import get_logger
from utils.exceptions import DataProcessingError

logger = get_logger(__name__)

class ArticleTracker:
    def __init__(self, db_path=ARTICLE_TRACKER_DB):
        self.db_path = db_path
        self._ensure_db_directory_exists()
        self._create_table()
        self._create_indexes()

    def _ensure_db_directory_exists(self):
        """Ensures the directory for the SQLite database exists."""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def _create_table(self):
        """Creates the articles table if it doesn't exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT UNIQUE NOT NULL, -- FMP's id or generated hash
                    url TEXT,
                    title TEXT,
                    title_hash TEXT UNIQUE NOT NULL, -- Hash of title+url for deduplication
                    ticker TEXT,
                    processed_date TEXT NOT NULL,
                    article_timestamp TEXT NOT NULL -- Original timestamp from the article
                )
            """)
            conn.commit()
            logger.info("SQLite 'articles' table ensured to exist.")
        except sqlite3.Error as e:
            logger.error(f"Error creating articles table: {e}")
            raise DataProcessingError(f"Failed to create articles table: {e}")
        finally:
            if conn:
                conn.close()

    def _create_indexes(self):
        """Creates indexes on frequently queried columns."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_title_hash ON articles (title_hash);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_article_timestamp ON articles (article_timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON articles (ticker);")
            conn.commit()
            logger.info("SQLite indexes ensured to exist.")
        except sqlite3.Error as e:
            logger.error(f"Error creating indexes: {e}")
            raise DataProcessingError(f"Failed to create SQLite indexes: {e}")
        finally:
            if conn:
                conn.close()

    def add_processed_article(self, article_id: str, url: str, title: str, title_hash: str, ticker: str, article_timestamp: datetime):
        """
        Adds a processed article to the database.
        article_timestamp should be a datetime object.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            processed_date_str = datetime.utcnow().isoformat()
            article_timestamp_str = article_timestamp.isoformat()

            cursor.execute("""
                INSERT INTO articles (article_id, url, title, title_hash, ticker, processed_date, article_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (article_id, url, title, title_hash, ticker, processed_date_str, article_timestamp_str))
            conn.commit()
            logger.debug(f"Article '{title}' (ID: {article_id}) added to tracker.")
        except sqlite3.IntegrityError:
            logger.debug(f"Article '{title}' (ID: {article_id}) already exists in tracker. Skipping.")
        except sqlite3.Error as e:
            logger.error(f"Error adding article '{title}' to tracker: {e}")
            raise DataProcessingError(f"Failed to add article to tracker: {e}")
        finally:
            if conn:
                conn.close()

    def is_article_processed(self, article_id: str | None = None, title_hash: str | None = None) -> bool:
        """
        Checks if an article has been processed based on article_id or title_hash.
        Prefer `title_hash` for robustness if `article_id` is not unique across sources/updates.
        """
        if not article_id and not title_hash:
            raise ValueError("Either article_id or title_hash must be provided.")

        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if article_id:
                cursor.execute("SELECT 1 FROM articles WHERE article_id = ?", (article_id,))
            elif title_hash:
                cursor.execute("SELECT 1 FROM articles WHERE title_hash = ?", (title_hash,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking if article is processed: {e}")
            raise DataProcessingError(f"Failed to check article processing status: {e}")
        finally:
            if conn:
                conn.close()

    def get_last_processed_timestamp(self) -> datetime | None:
        """
        Retrieves the timestamp of the most recently processed article.
        Returns None if no articles have been processed.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(article_timestamp) FROM articles")
            result = cursor.fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving last processed timestamp: {e}")
            raise DataProcessingError(f"Failed to get last processed timestamp: {e}")
        finally:
            if conn:
                conn.close()

    def get_all_processed_articles(self) -> list[dict]:
        """Retrieves all processed articles from the database."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles ORDER BY article_timestamp DESC")
            articles = [dict(row) for row in cursor.fetchall()]
            return articles
        except sqlite3.Error as e:
            logger.error(f"Error fetching all processed articles: {e}")
            raise DataProcessingError(f"Failed to fetch all processed articles: {e}")
        finally:
            if conn:
                conn.close()

    def clear_tracker(self):
        """Clears all entries from the articles table. Use with caution."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles")
            conn.commit()
            logger.warning("All articles cleared from the tracker database.")
        except sqlite3.Error as e:
            logger.error(f"Error clearing article tracker: {e}")
            raise DataProcessingError(f"Failed to clear article tracker: {e}")
        finally:
            if conn:
                conn.close()