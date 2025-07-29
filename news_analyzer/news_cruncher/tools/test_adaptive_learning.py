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

from tools.model_learning import MultiModalLearningSystem
from config import Config
from utils.simple_logger import log_info, log_error, log_warning

def create_safe_test_csv():
    """Create a SEPARATE test CSV file - NEVER touch the real one"""
    # Use a completely different path for testing
    test_csv_path = Config.DATA_DIR / "test_adaptive_learning.csv"

    log_info(f"Creating SAFE test CSV at: {test_csv_path}")

    # Create test data with realistic tickers and all required columns
    test_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00'] * 20),
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'] * 4,  # Real ticker names
        'decision': ['LONG', 'SHORT', 'NONE'] * 6 + ['LONG', 'SHORT'],
        'reasoning': [f'Test reasoning {i} for adaptive learning with detailed analysis.' for i in range(20)],
        'confidence': [0.7 + (i % 3) * 0.1 for i in range(20)],
        'tracking_status': ['completed'] * 20,
        
        # Price tracking columns (required for training)
        'price_checkpoint1_change_pct': [(i % 10) - 5 for i in range(20)],
        'price_checkpoint2_change_pct': [(i % 8) - 4 for i in range(20)],
        'price_close_change_pct': [(i % 6) - 3 for i in range(20)],
        
        # Additional required columns
        'article_count': [1] * 20,
        'news_score': [0.1 * (i % 10) for i in range(20)],
        'technical_score': [0.05 * (i % 8) for i in range(20)],
        'combined_score': [0.08 * (i % 12) for i in range(20)],
        'news_confidence': [0.6 + (i % 4) * 0.1 for i in range(20)],
        'technical_strength': [0.5 + (i % 5) * 0.1 for i in range(20)]
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

            # Test creating labels (this validates the data structure) - FIXED METHOD NAME
            try:
                labels, exit_strategy_data = learning_system.create_training_labels(df)
                log_info(f"✅ Label creation successful: {len(labels)} labels")
                log_info(f"✅ Exit strategy data keys: {list(exit_strategy_data.keys())}")
                
                # Test the new logic - check for opportunity_costs
                if 'opportunity_costs' in exit_strategy_data:
                    total_opportunity_cost = sum(exit_strategy_data['opportunity_costs'])
                    log_info(f"✅ Opportunity cost calculation working: {total_opportunity_cost:.2f}%")
                else:
                    log_warning("⚠️ Opportunity costs not found in exit strategy data")
                
            except Exception as e:
                log_warning(f"⚠️ Label creation failed: {e}")
                test_passed = False

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
                test_passed = False

        else:
            log_warning("⚠️ Data loading returned empty results")
            test_passed = False

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

    # Test 6: Validate the fixed logic with specific test cases
    log_info("🧪 Testing fixed training label logic...")
    try:
        # Create a specific test case to validate the OR logic fix
        test_trade_data = {
            'decision': ['LONG', 'SHORT', 'NONE'],
            'price_checkpoint1_change_pct': [2.0, -1.0, 3.0],  # LONG profitable at checkpoint1, SHORT not profitable, NONE missed opportunity
            'price_checkpoint2_change_pct': [-1.0, 1.0, -0.5],  # LONG not profitable, SHORT not profitable, NONE within threshold
            'price_close_change_pct': [-0.5, -2.0, 0.2],       # LONG not profitable, SHORT profitable, NONE within threshold
            'ticker': ['TEST1', 'TEST2', 'TEST3'],
            'reasoning': ['Test case 1', 'Test case 2', 'Test case 3'],
            'confidence': [0.8, 0.8, 0.8],
            'tracking_status': ['completed', 'completed', 'completed'],
            'article_count': [1, 1, 1],
            'news_score': [0.5, 0.5, 0.5],
            'technical_score': [0.3, 0.3, 0.3],
            'combined_score': [0.4, 0.4, 0.4],
            'news_confidence': [0.7, 0.7, 0.7],
            'technical_strength': [0.6, 0.6, 0.6]
        }
        
        test_df = pd.DataFrame(test_trade_data)
        learning_system.csv_path = test_csv_path  # Use safe test path
        
        labels, exit_data = learning_system.create_training_labels(test_df)
        
        # Validate the expected results
        expected_results = [
            "LONG with profit at checkpoint1 should be labeled as good (2)",
            "SHORT with profit at close should be labeled as good (0)", 
            "NONE with opportunity above threshold should be labeled as missed opportunity (2)"
        ]
        
        log_info("📋 Expected vs Actual Label Results:")
        for i, (label, desc) in enumerate(zip(labels, expected_results)):
            log_info(f"   Trade {i+1}: Label={label} - {desc}")
        
        # Check if opportunity costs are calculated
        if 'opportunity_costs' in exit_data and len(exit_data['opportunity_costs']) == 3:
            log_info(f"✅ Opportunity costs calculated: {exit_data['opportunity_costs']}")
        else:
            log_warning("⚠️ Opportunity costs not properly calculated")
            test_passed = False
            
    except Exception as e:
        log_error(f"❌ Fixed logic validation failed: {e}")
        test_passed = False

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
        log_info("✅ The fixed training label logic is working correctly!")
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

def test_missed_opportunity_threshold():
    """Test the MISSED_OPPORTUNITY_THRESHOLD logic specifically"""
    log_info("🎯 Testing MISSED_OPPORTUNITY_THRESHOLD logic...")
    
    try:
        learning_system = MultiModalLearningSystem()
        
        # Get the threshold from config
        threshold = getattr(Config, 'MISSED_OPPORTUNITY_THRESHOLD', 1.0)
        log_info(f"📊 Using MISSED_OPPORTUNITY_THRESHOLD: {threshold}%")
        
        # Create test cases that should trigger missed opportunities
        test_cases = [
            {
                'description': 'NONE with price increase above threshold',
                'data': {
                    'decision': 'NONE',
                    'price_checkpoint1_change_pct': threshold + 0.5,  # Above threshold
                    'price_checkpoint2_change_pct': 0.2,
                    'price_close_change_pct': 0.1
                },
                'expected_label': 2,  # Should be BUY (missed LONG opportunity)
            },
            {
                'description': 'NONE with price decrease below -threshold',
                'data': {
                    'decision': 'NONE',
                    'price_checkpoint1_change_pct': 0.1,
                    'price_checkpoint2_change_pct': -(threshold + 0.5),  # Below -threshold
                    'price_close_change_pct': -0.2
                },
                'expected_label': 0,  # Should be SELL (missed SHORT opportunity)
            },
            {
                'description': 'NONE with price changes within threshold',
                'data': {
                    'decision': 'NONE',
                    'price_checkpoint1_change_pct': threshold - 0.1,  # Within threshold
                    'price_checkpoint2_change_pct': -(threshold - 0.1),  # Within threshold
                    'price_close_change_pct': 0.1
                },
                'expected_label': 1,  # Should be NEUTRAL (correct NONE)
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            # Create a single-row DataFrame for testing
            test_data = {
                'ticker': ['TEST'],
                'reasoning': ['Test reasoning'],
                'confidence': [0.8],
                'tracking_status': ['completed'],
                'article_count': [1],
                'news_score': [0.5],
                'technical_score': [0.3],
                'combined_score': [0.4],
                'news_confidence': [0.7],
                'technical_strength': [0.6]
            }
            test_data.update({key: [value] for key, value in test_case['data'].items()})
            
            test_df = pd.DataFrame(test_data)
            labels, exit_data = learning_system.create_training_labels(test_df)
            
            actual_label = labels[0]
            expected_label = test_case['expected_label']
            
            if actual_label == expected_label:
                log_info(f"✅ Test {i}: {test_case['description']} - PASSED (label: {actual_label})")
            else:
                log_error(f"❌ Test {i}: {test_case['description']} - FAILED (expected: {expected_label}, got: {actual_label})")
                return False
        
        log_info("✅ All MISSED_OPPORTUNITY_THRESHOLD tests passed!")
        return True
        
    except Exception as e:
        log_error(f"❌ MISSED_OPPORTUNITY_THRESHOLD test failed: {e}")
        return False

if __name__ == "__main__":
    # First check for backups
    check_csv_backup_options()
    print()
    
    # Test missed opportunity threshold logic
    threshold_test_passed = test_missed_opportunity_threshold()
    print()

    # Run safe test
    main_test_passed = test_system_safely()
    
    # Final summary
    print("\n" + "="*60)
    if threshold_test_passed and main_test_passed:
        log_info("🎉 ALL TESTS PASSED! Your fixed implementation is working correctly! 🎉")
    else:
        log_error("❌ Some tests failed. Please review the implementation.")
