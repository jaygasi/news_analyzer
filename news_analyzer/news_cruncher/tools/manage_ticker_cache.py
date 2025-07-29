"""
Manage ticker validation cache
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_loaders.base_fmp_loader import BaseFMPLoader
from config import Config
from utils.simple_logger import log_info, log_error # Using logger for consistency
import sqlite3

def show_cache_stats():
    """Show cache statistics"""
    try:
        fmp_loader = BaseFMPLoader(Config.FMP_API_KEY)
        # Assuming BaseFMPLoader has a method to get stats.
        # If not, this part would need to be implemented in BaseFMPLoader
        # or directly query the DB here similar to show_cached_tickers.
        if hasattr(fmp_loader, 'get_failed_ticker_cache_stats'):
            stats = fmp_loader.get_failed_ticker_cache_stats()
            print("\n📊 Failed Ticker Cache Statistics:")
            print(f"   Cache enabled in Config: {Config.ENABLE_FAILED_TICKER_CACHE}")
            print(f"   Total cached (all time): {stats.get('total_entries_in_db', 'N/A')}")
            print(f"   Currently active cached: {stats.get('active_cached_tickers', 'N/A')}")
            print(f"   Cache DB Path: {Config.FAILED_TICKER_CACHE_DB}")
        else:
            print("\n📊 Failed Ticker Cache Statistics (Direct DB Query):")
            print(f"   Cache enabled in Config: {Config.ENABLE_FAILED_TICKER_CACHE}")
            if Config.FAILED_TICKER_CACHE_DB.exists():
                 with sqlite3.connect(Config.FAILED_TICKER_CACHE_DB) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM failed_tickers")
                    total = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM failed_tickers WHERE expires_at > datetime('now')")
                    active = cursor.fetchone()[0]
                    print(f"   Total cached (all time): {total}")
                    print(f"   Currently active cached: {active}")
            else:
                print("   Cache DB not found.")
            print(f"   Cache DB Path: {Config.FAILED_TICKER_CACHE_DB}")

    except Exception as e:
        log_error(f"Error showing cache stats: {e}")

def show_cached_tickers():
    """Show cached invalid tickers"""
    cache_db = Config.FAILED_TICKER_CACHE_DB # Use the path from Config
    
    if not cache_db.exists():
        print("❌ No cache database found at:", cache_db)
        return
    
    try:
        with sqlite3.connect(cache_db) as conn:
            cursor = conn.execute('''
                SELECT ticker, failure_reason, failure_count, last_failed_at, expires_at
                FROM failed_tickers 
                WHERE expires_at > datetime('now')
                ORDER BY last_failed_at DESC
                LIMIT 20
            ''') # Increased limit to 20
            
            results = cursor.fetchall()
            
            if results:
                print("\n🚫 Actively Cached Invalid Tickers (up to 20 most recent):")
                print(f"{'Ticker':<10} | {'Reason':<30} | {'Fails':<5} | {'Last Failed (UTC)':<20} | {'Expires (UTC)':<20}")
                print("-" * 95)
                for ticker, reason, count, last_failed, expires in results:
                    print(f"{ticker:<10} | {reason:<30} | {count:<5} | {last_failed:<20} | {expires:<20}")
            else:
                print("✅ No invalid tickers currently actively cached.")
                
    except Exception as e:
        print(f"❌ Error reading cache: {e}")

def main():
    """Main function"""
    print("🔧 Ticker Cache Management Utility")
    print("1. Show cache statistics")
    print("2. Show currently cached invalid tickers")
    print("3. (Future: Clear specific ticker from cache)")
    print("4. (Future: Clear entire cache)")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == '1':
        show_cache_stats()
    elif choice == '2':
        show_cached_tickers()
    # Add more choices here as functionality expands
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()