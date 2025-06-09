"""
CSV Data Migration Script
Migrates old CSV format to new enhanced format with rich context
"""
import pandas as pd
import csv
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from utils.simple_logger import log_info, log_error, log_warning

class CSVMigrator:
    """Migrates old CSV format to new enhanced format"""
    
    def __init__(self):
        self.output_dir = Config.OUTPUT_DIR
        self.current_csv = Config.CSV_OUTPUT_PATH
        
        # Field mapping from old to new
        self.field_mapping = {
            # Direct mappings
            'timestamp': 'timestamp',
            'ticker': 'ticker', 
            'decision': 'decision',
            'confidence': 'confidence',
            'reasoning': 'reasoning',
            'news_score': 'news_score',
            'technical_score': 'technical_score',
            'combined_score': 'combined_score',
            'article_count': 'article_count',
            'news_direction': 'news_direction',
            'technical_direction': 'technical_direction',
            'news_source': 'news_source',
            'analysis_method': 'analysis_method',
            'tracking_status': 'tracking_status',
            
            # Renamed fields
            'news_confidence': 'news_confidence',  # Was in old format
            'technical_strength': 'technical_strength',  # Was in old format
            'entry_price': 'recommendation_price',
            'price_at_45min': 'price_checkpoint1',
            'price_at_60min': 'price_checkpoint2', 
            'price_at_close': 'price_close',
            'price_45min_change_pct': 'price_checkpoint1_change_pct',
            'price_60min_change_pct': 'price_checkpoint2_change_pct',
            'price_close_change_pct': 'price_close_change_pct'
        }
        
        # New fields that need default values
        self.new_fields_defaults = {
            'news_reasoning': '',
            'technical_reasoning': '',
            'sources_used': '',
            'analysis_timestamp': '',
            'recommendation_timestamp': '',
            'price_checkpoint1_timestamp': '',
            'price_checkpoint2_timestamp': '',
            'price_close_timestamp': ''
        }
        
        # Fields that existed in old but not in new (will be dropped)
        self.dropped_fields = ['source_count', 'agreement_score']
        
    def find_backup_files(self):
        """Find all backup CSV files"""
        backup_files = list(self.output_dir.glob("*_backup_*.csv"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
        return backup_files
    
    def analyze_backup_file(self, backup_path: Path):
        """Analyze backup file structure"""
        log_info(f"📊 Analyzing backup file: {backup_path.name}")
        
        try:
            df = pd.read_csv(backup_path)
            
            log_info(f"  📈 Total records: {len(df)}")
            log_info(f"  📋 Columns: {len(df.columns)}")
            
            # Show decision breakdown
            if 'decision' in df.columns:
                decision_counts = df['decision'].value_counts()
                log_info(f"  🎯 Decisions: {dict(decision_counts)}")
            
            # Show tracking status
            if 'tracking_status' in df.columns:
                tracking_counts = df['tracking_status'].value_counts()
                log_info(f"  📊 Tracking: {dict(tracking_counts)}")
            
            return True
            
        except Exception as e:
            log_error(f"Error analyzing backup: {e}")
            return False
    
    def migrate_data(self, backup_path: Path, preview_only=False):
        """Migrate data from backup to new format"""
        log_info(f"🔄 {'Previewing' if preview_only else 'Migrating'} data from: {backup_path.name}")
        
        try:
            # Read backup data
            old_df = pd.read_csv(backup_path)
            log_info(f"📖 Loaded {len(old_df)} records from backup")
            
            # Create new dataframe with new structure
            new_data = []
            
            for _, old_row in old_df.iterrows():
                new_row = {}
                
                # Map existing fields
                for old_field, new_field in self.field_mapping.items():
                    if old_field in old_df.columns:
                        new_row[new_field] = old_row[old_field]
                    else:
                        new_row[new_field] = ''
                
                # Add new fields with defaults
                for new_field, default_value in self.new_fields_defaults.items():
                    new_row[new_field] = default_value
                
                # Special handling for timestamp fields
                if 'timestamp' in old_row and pd.notna(old_row['timestamp']):
                    # Use original timestamp for all price timestamps as fallback
                    base_timestamp = old_row['timestamp']
                    new_row['recommendation_timestamp'] = base_timestamp
                    new_row['price_checkpoint1_timestamp'] = base_timestamp
                    new_row['price_checkpoint2_timestamp'] = base_timestamp
                    new_row['price_close_timestamp'] = base_timestamp
                    new_row['analysis_timestamp'] = base_timestamp
                
                # Enhance reasoning if possible
                if 'reasoning' in new_row and new_row['reasoning']:
                    # Try to extract news/technical reasoning from combined reasoning
                    reasoning = str(new_row['reasoning'])
                    if 'News:' in reasoning and 'Technical:' in reasoning:
                        parts = reasoning.split('Technical:')
                        if len(parts) == 2:
                            new_row['news_reasoning'] = parts[0].replace('News:', '').strip()
                            new_row['technical_reasoning'] = parts[1].strip()
                
                # Set sources_used based on available analysis
                sources = []
                if new_row.get('news_score', 0) != 0:
                    sources.append('news_analysis')
                if new_row.get('technical_score', 0) != 0:
                    sources.append('technical_analysis')
                new_row['sources_used'] = ','.join(sources) if sources else 'unknown'
                
                new_data.append(new_row)
            
            # Create new dataframe
            new_df = pd.DataFrame(new_data)
            
            # Ensure columns are in the correct order for new format
            new_headers = [
                'timestamp', 'ticker', 'decision', 'confidence', 'reasoning',
                'news_score', 'technical_score', 'combined_score', 'article_count',
                'news_direction', 'news_confidence', 'news_reasoning', 'news_source',
                'technical_direction', 'technical_strength', 'technical_reasoning',
                'analysis_method', 'sources_used', 'analysis_timestamp',
                'recommendation_price', 'recommendation_timestamp',
                'price_checkpoint1', 'price_checkpoint1_timestamp', 'price_checkpoint1_change_pct',
                'price_checkpoint2', 'price_checkpoint2_timestamp', 'price_checkpoint2_change_pct',
                'price_close', 'price_close_timestamp', 'price_close_change_pct',
                'tracking_status'
            ]
            
            # Reorder columns and fill missing ones
            for col in new_headers:
                if col not in new_df.columns:
                    new_df[col] = ''
            
            new_df = new_df[new_headers]
            
            if preview_only:
                log_info("📋 Preview of migrated data (first 3 rows):")
                for i, (_, row) in enumerate(new_df.head(3).iterrows()):
                    log_info(f"  Row {i+1}: {row['ticker']} {row['decision']} - {row['confidence']}")
                
                log_info(f"✅ Migration preview successful: {len(new_df)} rows ready")
                return True
            else:
                # Save to current CSV (append mode)
                if self.current_csv.exists():
                    # Append to existing
                    new_df.to_csv(self.current_csv, mode='a', header=False, index=False)
                    log_info(f"📝 Appended {len(new_df)} records to existing CSV")
                else:
                    # Create new file
                    new_df.to_csv(self.current_csv, index=False)
                    log_info(f"📝 Created new CSV with {len(new_df)} records")
                
                log_info(f"✅ Migration completed successfully!")
                return True
                
        except Exception as e:
            log_error(f"Error during migration: {e}")
            return False
    
    def run_interactive_migration(self):
        """Run interactive migration process"""
        print("\n" + "="*60)
        print("🔄 CSV DATA MIGRATION TOOL")
        print("="*60)
        
        # Find backup files
        backup_files = self.find_backup_files()
        
        if not backup_files:
            print("❌ No backup files found in output directory")
            return False
        
        print(f"📁 Found {len(backup_files)} backup files:")
        for i, backup_file in enumerate(backup_files, 1):
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            print(f"  {i}. {backup_file.name} ({file_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Select backup file
        while True:
            try:
                choice = input(f"\nSelect backup file to migrate (1-{len(backup_files)}): ").strip()
                choice_int = int(choice)
                if 1 <= choice_int <= len(backup_files):
                    selected_backup = backup_files[choice_int - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(backup_files)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Migration cancelled")
                return False
        
        # Analyze selected backup
        if not self.analyze_backup_file(selected_backup):
            return False
        
        # Preview migration
        print(f"\n🔍 Previewing migration from {selected_backup.name}...")
        if not self.migrate_data(selected_backup, preview_only=True):
            return False
        
        # Confirm migration
        print(f"\n⚠️  Current CSV: {self.current_csv}")
        print(f"📁 Backup file: {selected_backup}")
        
        proceed = input("\nProceed with migration? (y/N): ").strip().lower()
        if proceed != 'y':
            print("❌ Migration cancelled")
            return False
        
        # Perform actual migration
        return self.migrate_data(selected_backup, preview_only=False)

def main():
    """Main migration function"""
    print("🔄 Starting CSV Migration Process...")
    
    migrator = CSVMigrator()
    success = migrator.run_interactive_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("💡 Your historical data is now available with the new format")
        print("🚀 You can now run the main application normally")
    else:
        print("\n❌ Migration failed or was cancelled")
    
    return success

if __name__ == "__main__":
    main()
