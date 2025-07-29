#!/usr/bin/env python3
"""
Immediate fix for duplicate CSV rows - Run this NOW
"""
import sys
import csv
from pathlib import Path
from datetime import datetime

def fix_csv_duplicates():
    """Remove duplicate rows from trading_decisions.csv based on identical content"""
    csv_path = Path('output/trading_decisions.csv')

    if not csv_path.exists():
        print("❌ CSV file not found")
        return

    print(f"🔍 Processing: {csv_path}")

    # Read all rows
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        headers = reader.fieldnames

    print(f"📊 Original rows: {len(rows)}")

    # Keep only unique content (same ticker + same analysis = duplicate content)
    unique_content = {}
    for row in rows:
        ticker = row.get('ticker')
        news_score = row.get('news_score', '')
        news_confidence = row.get('news_confidence', '')
        reasoning = row.get('reasoning', '')

        if ticker:
            # Create content signature to detect identical analysis
            content_key = f"{ticker}|{news_score}|{news_confidence}|{reasoning}"

            # Keep most recent if duplicate content found
            if content_key not in unique_content or row.get('timestamp', '') > unique_content[content_key].get('timestamp', ''):
                unique_content[content_key] = row

    deduplicated_count = len(unique_content)
    removed_count = len(rows) - deduplicated_count

    if removed_count > 0:
        # Backup original
        backup_path = csv_path.parent / f"{csv_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"📦 Backup created: {backup_path}")

        # Write cleaned data
        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(unique_content.values())

        print(f"✅ Fixed! Removed {removed_count} duplicates, kept {deduplicated_count} unique rows")
        print("   Duplicate detection based on: ticker + news_score + news_confidence + reasoning")
    else:
        print("✅ No duplicates found")

if __name__ == "__main__":
    fix_csv_duplicates()