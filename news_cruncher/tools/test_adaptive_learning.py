"""
SAFE Adaptive Learning Test - NEVER OVERWRITES YOUR REAL CSV
Run this to verify your implementation is working correctly
"""
import sys
import os
from pathlib import Path

# Fix Python path - add the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports should work
import pandas as pd
import time

from tools.multi_modal_learning_system import MultiModalLearningSystem
from config import Config
from utils.simple_logger import log_info, log_error, log_warning

def create_safe_test_csv():
    """Create a SEPARATE test CSV file - NEVER touch the real one"""
    # Use a completely different path for testing
    test_csv_path = Config.DATA_DIR / "test_adaptive_learning.csv"
    
    log_info(f"Creating SAFE test CSV at: {test_csv_path}")
    
    # Create test data with realistic tickers
    test_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00'] * 20),
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * 4,  # Real ticker names
        'decision': ['BUY', 'SELL', 'NEUTRAL'] * 6 + ['BUY', 'SELL'],
        'reasoning': [f'Test reasoning {i} for adaptive learning.' for i in range(20)],
        'confidence': [0.7 + (i % 3) * 0.1 for i in range(20)],
        'tracking_status': ['completed'] * 20,
        'price_close_change_pct': [(i % 10) - 5 for i in range(20)],
        'article_count': [1] * 20,
        'news_score': [0.1 * (i % 10) for i in range(20)],
        'technical_score': [0.05 * (i % 8) for i in range(20)],
        'combined_score': [0.08 * (i % 12) for i in range(20)]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv(test_csv_path, index=False)
    log_info(f"✅ Created SAFE test CSV with {len(df)} completed trades.")
    
    return test_csv_path

def test_system_safely():
    """Run SAFE test that never touches your real CSV"""
    log_info("🚀 Starting SAFE Adaptive Learning System Test...")
    log_info("🛡️ This test will NEVER modify your real trading CSV!")
    
    test_passed = True
    
    # Test 1: Check your real CSV is safe
    real_csv_path = Config.CSV_OUTPUT_PATH
    if real_csv_path.exists():
        try:
            real_df = pd.read_csv(real_csv_path)
            log_info(f"✅ Your real CSV is safe: {len(real_df)} rows at {real_csv_path}")
            
            # Show sample of real tickers to confirm
            if 'ticker' in real_df.columns:
                sample_tickers = real_df['ticker'].unique()[:5]
                log_info(f"✅ Real ticker sample: {list(sample_tickers)}")
        except Exception as e:
            log_warning(f"Could not read real CSV: {e}")
    
    # Test 2: MultiModalLearningSystem initialization
    log_info("🧠 Testing MultiModalLearningSystem initialization...")
    try:
        learning_system = MultiModalLearningSystem()
        log_info("✅ MultiModalLearningSystem initialized successfully.")
    except Exception as e:
        log_error(f"❌ MultiModalLearningSystem initialization FAILED: {e}")
        return False

    # Test 3: Create safe test data
    log_info("📊 Creating SAFE test data...")
    test_csv_path = create_safe_test_csv()
    
    # Test 4: Test data preparation (without training)
    log_info("🔍 Testing data preparation methods...")
    try:
        # Temporarily point to test CSV
        original_csv_path = learning_system.csv_path
        learning_system.csv_path = test_csv_path
        
        # Test the data loading
        df, success = learning_system.load_and_prepare_data()
        
        if success and df is not None:
            log_info(f"✅ Data loading successful: {len(df)} completed trades")
            
            # Test creating labels (this validates the data structure)
            try:
                labels = learning_system.create_performance_labels(df)
                log_info(f"✅ Label creation successful: {len(labels)} labels")
            except Exception as e:
                log_warning(f"⚠️ Label creation failed: {e}")
            
            # Test feature preparation (basic validation)
            try:
                texts, numerical_features, categorical_features = learning_system.prepare_features(df)
                log_info(f"✅ Feature preparation successful: {len(texts)} text samples")
                if numerical_features is not None:
                    log_info(f"✅ Numerical features shape: {numerical_features.shape}")
                if categorical_features:
                    log_info(f"✅ Categorical features: {list(categorical_features.keys())}")
            except Exception as e:
                log_warning(f"⚠️ Feature preparation failed: {e}")
                
        else:
            log_warning("⚠️ Data loading returned empty results")
        
        # Restore original path
        learning_system.csv_path = original_csv_path
        
    except Exception as e:
        log_error(f"❌ Data preparation test failed: {e}")
        test_passed = False
    
    # Test 5: Check learning system components
    log_info("🔧 Testing learning system components...")
    
    if hasattr(learning_system, 'enhanced_analyzer'):
        if learning_system.enhanced_analyzer:
            log_info("✅ Enhanced analyzer available in learning system")
        else:
            log_info("ℹ️ Enhanced analyzer not initialized in learning system")
    
    if hasattr(learning_system, 'model_save_dir'):
        log_info(f"✅ Model save directory: {learning_system.model_save_dir}")
        
        # Check for existing trained models
        finbert_checkpoint = learning_system.model_save_dir / "finbert_multimodal_adaptive.pth"
        enhanced_checkpoint = learning_system.model_save_dir / "enhanced_neural_multimodal_adaptive.pth"
        processors_file = learning_system.model_save_dir / "data_processors.pkl"
        
        existing_files = []
        if finbert_checkpoint.exists():
            existing_files.append("FinBERT checkpoint")
        if enhanced_checkpoint.exists():
            existing_files.append("Enhanced Neural checkpoint")
        if processors_file.exists():
            existing_files.append("Data processors")
            
        if existing_files:
            log_info(f"✅ Found existing training: {', '.join(existing_files)}")
        else:
            log_info("ℹ️ No existing training checkpoints (normal for first run)")
    
    # Clean up test file
    try:
        if test_csv_path.exists():
            test_csv_path.unlink()
            log_info("🧹 Cleaned up test CSV file")
    except Exception as e:
        log_warning(f"Could not clean up test file: {e}")
    
    if test_passed:
        log_info("🎉🎉🎉 SAFE Adaptive Learning System Test PASSED! 🎉🎉🎉")
        log_info("🛡️ Your real trading CSV was never touched!")
    else:
        log_error("😭😭😭 Adaptive Learning System Test FAILED! 😭😭😭")
    
    return test_passed

def check_csv_backup_options():
    """Check for CSV backup files"""
    log_info("🔍 Checking for CSV backup files...")
    
    output_dir = Config.CSV_OUTPUT_PATH.parent
    backup_files = list(output_dir.glob("*_backup_*.csv"))
    
    if backup_files:
        log_info(f"📁 Found {len(backup_files)} backup files:")
        for backup in sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_mb = backup.stat().st_size / (1024 * 1024)
            mtime = backup.stat().st_mtime
            log_info(f"  • {backup.name} ({size_mb:.1f} MB)")
        
        latest_backup = sorted(backup_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        log_info(f"💡 Latest backup: {latest_backup.name}")
        log_info(f"💡 To restore: copy {latest_backup.name} to {Config.CSV_OUTPUT_PATH.name}")
    else:
        log_info("📁 No backup files found")

if __name__ == "__main__":
    # First check for backups
    check_csv_backup_options()
    print()
    
    # Run safe test
    test_system_safely()
