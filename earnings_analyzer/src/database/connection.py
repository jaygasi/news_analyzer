from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from config.config import Config
from .models import Base
import logging
import os

logger = logging.getLogger(__name__)

# Ensure database directory exists for SQLite
if "sqlite" in Config.DATABASE_URL:
    db_path = Config.DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Created database directory: {db_dir}")

# Create database engine with proper configuration
try:
    if "sqlite" in Config.DATABASE_URL:
        # SQLite specific settings
        engine = create_engine(
            Config.DATABASE_URL,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=Config.LOG_LEVEL == 'DEBUG'
        )
    else:
        # PostgreSQL or other database settings
        engine = create_engine(
            Config.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
            echo=Config.LOG_LEVEL == 'DEBUG'
        )
    
    logger.info(f"Database engine created successfully: {Config.DATABASE_URL.split('://')[0]}")
    
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Log table creation details
        table_names = [table.name for table in Base.metadata.tables.values()]
        logger.info(f"Available tables: {', '.join(table_names)}")
        
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error creating database tables: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating database tables: {e}")
        raise

def get_db():
    """Get database session - for dependency injection"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session():
    """Get database session - for direct use"""
    return SessionLocal()

def test_connection():
    """Test database connection"""
    try:
        db = SessionLocal()
        try:
            # Test basic connection
            db.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        finally:
            db.close()
            
    except SQLAlchemyError as e:
        logger.error(f"Database connection test failed (SQLAlchemy): {e}")
        return False
    except Exception as e:
        logger.error(f"Database connection test failed (Unexpected): {e}")
        return False

def get_db_stats():
    """Get database statistics"""
    try:
        from .models import Company, NewsArticle, TradingSignal, EarningsCalendar
        
        db = SessionLocal()
        try:
            stats = {
                "companies": db.query(Company).count(),
                "news_articles": db.query(NewsArticle).count(), 
                "trading_signals": db.query(TradingSignal).count(),
                "earnings_calendar": db.query(EarningsCalendar).count(),
            }
            
            # Additional useful stats
            processed_articles = db.query(NewsArticle).filter(NewsArticle.processed == True).count()
            high_confidence_signals = db.query(TradingSignal).filter(TradingSignal.confidence_score >= 0.8).count()
            
            stats.update({
                "processed_articles": processed_articles,
                "high_confidence_signals": high_confidence_signals,
                "unprocessed_articles": stats["news_articles"] - processed_articles
            })
            
            return stats
            
        finally:
            db.close()
        
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error getting database stats: {e}")
        return {"error": f"Database error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error getting database stats: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def cleanup_old_data(days_to_keep: int = 30):
    """Clean up old data to keep database size manageable"""
    try:
        from datetime import datetime, timedelta
        from .models import NewsArticle, TradingSignal, SystemLog
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        db = SessionLocal()
        try:
            # Count items to be deleted
            old_articles_query = db.query(NewsArticle).filter(
                NewsArticle.created_at < cutoff_date,
                NewsArticle.processed == True,
                ~NewsArticle.trading_signals.any()  # No associated trading signals
            )
            old_articles_count = old_articles_query.count()
            
            # Delete old news articles that weren't processed into signals
            if old_articles_count > 0:
                old_articles_query.delete(synchronize_session=False)
                logger.info(f"Cleaned up {old_articles_count} old news articles")
            
            # Count and delete old system logs
            old_logs_query = db.query(SystemLog).filter(SystemLog.timestamp < cutoff_date)
            old_logs_count = old_logs_query.count()
            
            if old_logs_count > 0:
                old_logs_query.delete(synchronize_session=False)
                logger.info(f"Cleaned up {old_logs_count} old system logs")
            
            db.commit()
            
            total_cleaned = old_articles_count + old_logs_count
            if total_cleaned > 0:
                logger.info(f"Database cleanup completed: {total_cleaned} total records removed")
            else:
                logger.info("Database cleanup completed: No old records to remove")
                
            return total_cleaned
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
        
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error during database cleanup: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during database cleanup: {e}")
        raise

def initialize_database():
    """Initialize database with tables and basic setup"""
    try:
        logger.info("Initializing database...")
        
        # Test connection first
        if not test_connection():
            raise Exception("Database connection test failed")
        
        # Create tables
        create_tables()
        
        # Get initial stats
        stats = get_db_stats()
        logger.info(f"Database initialization complete. Stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Helper function to safely execute database operations
def safe_db_execute(operation, *args, **kwargs):
    """Safely execute a database operation with proper error handling"""
    db = SessionLocal()
    try:
        result = operation(db, *args, **kwargs)
        db.commit()
        return result
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database operation failed: {e}")
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error in database operation: {e}")
        raise
    finally:
        db.close()
