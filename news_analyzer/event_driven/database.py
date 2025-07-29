import sqlite3
from typing import Dict, Any, List, Set
import logging
from config import DATABASE_FILE
from utils import generate_news_hash


def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database and creates/updates tables as needed."""
    logging.info("--- Initializing Database ---")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        ticker TEXT NOT NULL,
        decision TEXT NOT NULL,
        entry_price REAL NOT NULL,
        predicted_confidence REAL,
        predicted_volatility REAL,
        news_text TEXT,
        feature_finbert_score REAL,
        feature_finbert_tone_score REAL,
        feature_keyword_score REAL,
        feature_alpha_vantage_relevance REAL,
        feature_alpha_vantage_sentiment REAL,
        perf_30_min_pct REAL,
        perf_30_min_timestamp TEXT,
        perf_60_min_pct REAL,
        perf_60_min_timestamp TEXT,
        perf_240_min_pct REAL,
        perf_240_min_timestamp TEXT,
        perf_eod_pct REAL, perf_eod_timestamp TEXT,
        tracking_status TEXT DEFAULT 'pending' NOT NULL,
        feature_model_explanation TEXT
    );
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS keyword_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL,
        keyword TEXT NOT NULL,
        weight REAL NOT NULL,
        FOREIGN KEY (trade_id) REFERENCES trades (id)
    );
    """
    )

    cursor.execute("CREATE TABLE IF NOT EXISTS processed_news (id TEXT PRIMARY KEY);")
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS failed_tickers (ticker TEXT PRIMARY KEY);"
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_timestamp TEXT NOT NULL,
        model_version TEXT NOT NULL UNIQUE,
        scaler_path TEXT,
        avg_training_loss REAL,
        val_accuracy REAL,
        val_f1_macro REAL,
        val_confidence_mse REAL,
        val_volatility_mse REAL,
        notes TEXT
    );
    """
    )

    cursor.execute("PRAGMA table_info(trades)")
    columns = {row["name"]: row for row in cursor.fetchall()}
    if "feature_keyword_count" in columns:
        logging.info("Renaming 'feature_keyword_count' to 'feature_keyword_score' in trades table.")
        cursor.execute("ALTER TABLE trades RENAME COLUMN feature_keyword_count TO feature_keyword_score")
    if "feature_model_explanation" not in columns:
        logging.info("Adding 'feature_model_explanation' column to trades table.")
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_model_explanation TEXT")
    if "tracking_status" not in columns:
        logging.info("Adding 'tracking_status' column to trades table.")
        cursor.execute("ALTER TABLE trades ADD COLUMN tracking_status TEXT DEFAULT 'pending' NOT NULL")
    if "perf_30_min_timestamp" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN perf_30_min_timestamp TEXT")
    if "perf_60_min_timestamp" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN perf_60_min_timestamp TEXT")
    if "perf_240_min_timestamp" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN perf_240_min_timestamp TEXT")
    if "perf_eod_timestamp" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN perf_eod_timestamp TEXT")
    if "feature_alpha_vantage_relevance" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_alpha_vantage_relevance REAL")
    if "feature_alpha_vantage_sentiment" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_alpha_vantage_sentiment REAL")
    if "feature_beta" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_beta REAL")
    if "feature_market_cap" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_market_cap REAL")
    if "feature_volume" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_volume REAL")
    if "feature_change_percent" not in columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN feature_change_percent REAL")

    cursor.execute("PRAGMA table_info(training_history)")
    training_history_columns = {row["name"]: row for row in cursor.fetchall()}
    if "scaler_path" not in training_history_columns:
        logging.info("Adding 'scaler_path' column to training_history table.")
        cursor.execute("ALTER TABLE training_history ADD COLUMN scaler_path TEXT")

    conn.commit()
    conn.close()
    logging.info("Database initialized successfully.")


def add_to_cache(table_name: str, column_name: str, item_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"INSERT OR IGNORE INTO {table_name} ({column_name}) VALUES (?)", (item_id,)
    )
    conn.commit()
    conn.close()


def log_trade(trade_data: Dict[str, Any], conn: sqlite3.Connection) -> int:
    cursor = conn.cursor()
    columns = list(trade_data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    values = [trade_data.get(col) for col in columns]
    sql = f"INSERT INTO trades ({', '.join(columns)}) VALUES ({placeholders})"
    trade_id = None
    try:
        cursor.execute(sql, values)
        trade_id = cursor.lastrowid
        logging.info(f"Logged trade for {trade_data.get('ticker')} with ID: {trade_id}")
    except sqlite3.Error as e:
        logging.error(f"Error logging trade for {trade_data.get('ticker')}: {e}")
        logging.error(f"Columns: {columns}\nValues: {values}")
    return trade_id

def log_trades_batch(trades_data: List[Dict[str, Any]], conn: sqlite3.Connection) -> List[int]:
    if not trades_data:
        return []
    cursor = conn.cursor()
    columns = list(trades_data[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO trades ({', '.join(columns)}) VALUES ({placeholders})"
    values_to_insert = [[trade.get(col) for col in columns] for trade in trades_data]
    trade_ids = []
    try:
        cursor.executemany(sql, values_to_insert)
        logging.info(f"Logged {len(trades_data)} trades in batch.")
        last_id = cursor.lastrowid
        if last_id is not None:
            trade_ids = list(range(last_id - len(trades_data) + 1, last_id + 1))
        else:
            trade_ids = [None] * len(trades_data)

    except sqlite3.Error as e:
        logging.error(f"Error logging trades in batch: {e}")
        trade_ids = [None] * len(trades_data)
    return trade_ids


def log_keyword_matches(trade_id: int, matched_keywords: Dict[str, float], conn: sqlite3.Connection):
    if not trade_id:
        return
    cursor = conn.cursor()
    match_data = [
        (trade_id, keyword, weight) for keyword, weight in matched_keywords.items()
    ]
    try:
        cursor.executemany(
            "INSERT INTO keyword_matches (trade_id, keyword, weight) VALUES (?, ?, ?)",
            match_data,
        )
        logging.info(
            f"Logged {len(match_data)} keyword matches for trade ID {trade_id}."
        )
    except sqlite3.Error as e:
        logging.error(f"Error logging keyword matches for trade ID {trade_id}: {e}")

def log_keyword_matches_batch(all_matches: List[Dict[str, Any]], conn: sqlite3.Connection):
    if not all_matches:
        return
    cursor = conn.cursor()
    match_data_tuples = []
    for match_dict in all_matches:
        match_data_tuples.append((match_dict['trade_id'], match_dict['keyword'], match_dict['weight']))

    try:
        cursor.executemany(
            "INSERT INTO keyword_matches (trade_id, keyword, weight) VALUES (?, ?, ?)",
            match_data_tuples,
        )
        logging.info(f"Logged {len(match_data_tuples)} keyword matches in batch.")
    except sqlite3.Error as e:
        logging.error(f"Error logging keyword matches in batch: {e}")


def load_cache_set(table_name: str, column_name: str) -> Set[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT {column_name} FROM {table_name}")
    items = {row[column_name] for row in cursor.fetchall()}
    conn.close()
    return items


def is_news_processed(news_hash: str, cache_set: Set[str]) -> bool:
    return news_hash in cache_set


def add_news_hashes_batch(hashes: List[str]):
    if not hashes:
        return
    conn = get_db_connection()
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO processed_news (id) VALUES (?)",
            [(h,) for h in hashes],
        )
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error batch adding news hashes: {e}")
    finally:
        conn.close()


def is_failed_ticker(ticker: str, cache_set: Set[str]) -> bool:
    return ticker in cache_set


def add_failed_ticker(ticker: str):
    add_to_cache("failed_tickers", "ticker", ticker)


def get_untracked_trades(status: str = "pending") -> List[Dict[str, Any]]:
    conn = get_db_connection()
    trades = conn.execute(
        "SELECT * FROM trades WHERE tracking_status = ? ORDER BY timestamp ASC",
        (status,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in trades]


def update_trade_status(trade_id: int, status: str, conn: sqlite3.Connection):
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE trades SET tracking_status = ? WHERE id = ?", (status, trade_id)
        )
    except sqlite3.Error as e:
        logging.error(f"Error updating trade status for ID {trade_id}: {e}")

def update_trades_status_batch(status_updates: List[Dict[str, Any]], conn: sqlite3.Connection):
    """Batch updates the tracking_status for multiple trades."""
    if not status_updates:
        return
    
    cursor = conn.cursor()
    update_tuples = [(update['status'], update['trade_id']) for update in status_updates]
    
    try:
        cursor.executemany("UPDATE trades SET tracking_status = ? WHERE id = ?", update_tuples)
        logging.info(f"Batch updated status for {len(update_tuples)} trades.")
    except sqlite3.Error as e:
        logging.error(f"Error during batch status update for trades: {e}")

def update_trade_performance(
    trade_id: int, performance_data: Dict[str, Any], conn: sqlite3.Connection
):
    cursor = conn.cursor()
    sql = """UPDATE trades SET
                perf_30_min_pct = ?, perf_30_min_timestamp = ?,
                perf_60_min_pct = ?, perf_60_min_timestamp = ?,
                perf_240_min_pct = ?, perf_240_min_timestamp = ?,
                perf_eod_pct = ?, perf_eod_timestamp = ?
             WHERE id = ?"""
    values = (
        performance_data.get("perf_30_min_pct"),
        performance_data.get("perf_30_min_timestamp"),
        performance_data.get("perf_60_min_pct"),
        performance_data.get("perf_60_min_timestamp"),
        performance_data.get("perf_240_min_pct"),
        performance_data.get("perf_240_min_timestamp"),
        performance_data.get("perf_eod_pct"),
        performance_data.get("perf_eod_timestamp"),
        trade_id,
    )
    try:
        cursor.execute(sql, values)
        if cursor.rowcount == 0:
            logging.warning(f"No trade found with ID {trade_id} to update performance.")
    except sqlite3.Error as e:
        logging.error(f"Error updating trade performance for ID {trade_id}: {e}")

def update_trades_performance_batch(performance_updates: List[Dict[str, Any]], conn: sqlite3.Connection):
    if not performance_updates:
        return
    cursor = conn.cursor()
    sql = """UPDATE trades SET
                perf_30_min_pct = ?, perf_30_min_timestamp = ?,
                perf_60_min_pct = ?, perf_60_min_timestamp = ?,
                perf_240_min_pct = ?, perf_240_min_timestamp = ?,
                perf_eod_pct = ?, perf_eod_timestamp = ?,
                tracking_status = ?
             WHERE id = ?"""
    values_to_update = []
    for update_data in performance_updates:
        values_to_update.append((
            update_data.get("perf_30_min_pct"),
            update_data.get("perf_30_min_timestamp"),
            update_data.get("perf_60_min_pct"),
            update_data.get("perf_60_min_timestamp"),
            update_data.get("perf_240_min_pct"),
            update_data.get("perf_240_min_timestamp"),
            update_data.get("perf_eod_pct"),
            update_data.get("perf_eod_timestamp"),
            update_data.get("tracking_status", "completed"),
            update_data.get("trade_id"),
        ))
    try:
        cursor.executemany(sql, values_to_update)
        logging.info(f"Updated performance for {len(performance_updates)} trades in batch.")
    except sqlite3.Error as e:
        logging.error(f"Error updating trades performance in batch: {e}")


def get_training_data() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE tracking_status = 'completed'")
    training_data = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return training_data


def log_training_run(log_data: Dict[str, Any]):
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = "INSERT INTO training_history (training_timestamp, model_version, scaler_path, avg_training_loss, val_accuracy, val_f1_macro, val_confidence_mse, val_volatility_mse, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    values = (
        log_data.get("training_timestamp"),
        log_data.get("model_version"),
        log_data.get("scaler_path"),
        log_data.get("avg_training_loss"),
        log_data.get("val_accuracy"),
        log_data.get("val_f1_macro"),
        log_data.get("val_confidence_mse"),
        log_data.get("val_volatility_mse"),
        log_data.get("notes"),
    )
    cursor.execute(sql, values)
    conn.commit()
    conn.close()


def get_latest_model_path() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT model_version, scaler_path FROM training_history ORDER BY training_timestamp DESC LIMIT 1"
    )
    result = cursor.fetchone()
    conn.close()
    return dict(result) if result else {}


def get_latest_training_history() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM training_history ORDER BY training_timestamp DESC LIMIT 1"
    )
    result = cursor.fetchone()
    conn.close()
    return dict(result) if result else None


def get_trades_column_names() -> List[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    column_names = [row["name"] for row in cursor.fetchall()]
    conn.close()
    return column_names