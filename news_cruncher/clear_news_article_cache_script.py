import sqlite3
from pathlib import Path

def clear_article_cache():
    """Clear the article processing cache to allow reprocessing"""
    
    # Database path
    db_path = Path('data/article_tracking.db')
    
    if not db_path.exists():
        print("❌ Database file doesn't exist yet")
        return
    
    try:
        # Connect and clear the processed articles table
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get count before clearing
            cursor.execute("SELECT COUNT(*) FROM processed_articles")
            count_before = cursor.fetchone()[0]
            
            # Clear all processed articles
            cursor.execute("DELETE FROM processed_articles")
            
            # Get count after clearing
            cursor.execute("SELECT COUNT(*) FROM processed_articles")
            count_after = cursor.fetchone()[0]
            
            conn.commit()
            
            print(f"✅ Cleared article processing cache")
            print(f"   Articles before: {count_before}")
            print(f"   Articles after: {count_after}")
            print(f"   🔄 Next run will reprocess all articles")
            
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

if __name__ == "__main__":
    clear_article_cache()
