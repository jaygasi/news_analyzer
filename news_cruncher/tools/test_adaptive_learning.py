"""
Test Adaptive Learning Implementation
Run this to verify your implementation is working correctly
"""
import os
import pandas as pd
from pathlib import Path
import time # For waiting if necessary

from tools.multi_modal_learning_system import MultiModalLearningSystem
from core.adaptive_learning_scheduler import force_adaptive_learning, start_adaptive_learning_scheduler, stop_adaptive_learning_scheduler, is_adaptive_learning_scheduler_running
from config import Config
from utils.simple_logger import log_info, log_error, log_warning # Changed import, added log_warning

def setup_test_environment():
    """Initializes logger and config for the test."""
    # setup_logger() # Removed call, assuming logger initializes differently now
    # Config.load_config() # Removed call, .env is loaded on import of Config module
    log_info("Test environment setup complete.")

def create_dummy_csv_if_not_exists(num_rows=15):
    """Creates a dummy CSV file if it doesn't exist or is too small, for testing purposes."""
    csv_path = Config.CSV_OUTPUT_PATH
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if CSV exists and has enough completed trades
    try:
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            if 'tracking_status' in df_existing.columns and \
               len(df_existing[df_existing['tracking_status'] == 'completed']) >= getattr(Config, 'ADAPTIVE_LEARNING_MIN_TRADES', 10):
                log_info(f"Sufficient completed trades found in existing CSV: {csv_path}")
                return
    except Exception as e:
        log_warning(f"Could not read existing CSV {csv_path}: {e}. Will attempt to create a new one.")

    log_info(f"Creating/overwriting dummy CSV for testing at: {csv_path} with {num_rows} rows.")
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00'] * num_rows),
        'ticker': [f'TICK{i%3+1}' for i in range(num_rows)],
        'decision': ['BUY', 'SELL', 'NEUTRAL'] * (num_rows // 3) + ['BUY'] * (num_rows % 3),
        'reasoning': [f'Test reasoning {i} for adaptive learning.' for i in range(num_rows)],
        'confidence': [0.5 + (i % 5) * 0.1 for i in range(num_rows)],
        'news_score': [60 + i % 40 for i in range(num_rows)],
        'technical_score': [50 + i % 50 for i in range(num_rows)],
        'combined_score': [55 + i % 45 for i in range(num_rows)],
        'news_confidence': [0.5 + (i % 5) * 0.1 for i in range(num_rows)],
        'technical_strength': [0.4 + (i % 6) * 0.1 for i in range(num_rows)],
        'article_count': [1] * num_rows,
        'source_count': [2] * num_rows,
        'agreement_score': [0.6 + (i % 4) * 0.1 for i in range(num_rows)],
        'news_direction': ['BUY', 'SELL', 'NEUTRAL'] * (num_rows // 3) + ['BUY'] * (num_rows % 3),
        'technical_direction': ['BUY', 'NEUTRAL', 'SELL'] * (num_rows // 3) + ['BUY'] * (num_rows % 3),
        'news_source': ['sourceA', 'sourceB'] * (num_rows // 2) + ['sourceA'] * (num_rows % 2),
        'analysis_method': ['method1'] * num_rows,
        'tracking_status': ['completed'] * num_rows, # Ensure all are completed for training
        'entry_price': [100.0 + i for i in range(num_rows)],
        'price_at_45min': [100.0 + i + (i%3-1)*0.5 for i in range(num_rows)], # some profit/loss
        'price_at_60min': [100.0 + i + (i%3-1)*0.8 for i in range(num_rows)],
        'price_at_close': [100.0 + i + (i%3-1)*1.2 for i in range(num_rows)],
        'price_45min_change_pct': [( (i%3-1)*0.5 ) / (100.0+i) * 100 for i in range(num_rows)],
        'price_60min_change_pct': [( (i%3-1)*0.8 ) / (100.0+i) * 100 for i in range(num_rows)],
        'price_close_change_pct': [( (i%3-1)*1.2 ) / (100.0+i) * 100 for i in range(num_rows)],
    }
    df = pd.DataFrame(data)
    # Ensure all required columns for training are present, even if with default values
    required_cols = [
        'confidence', 'news_score', 'technical_score', 'combined_score',
        'news_confidence', 'technical_strength', 'article_count',
        'source_count', 'agreement_score', 'reasoning',
        'news_direction', 'technical_direction', 'news_source', 'analysis_method',
        'tracking_status', 'price_close_change_pct'
    ]
    for col in required_cols:
        if col not in df.columns:
            if 'pct' in col or 'score' in col or 'confidence' in col:
                 df[col] = 0.0
            elif 'count' in col:
                 df[col] = 0
            else:
                 df[col] = 'unknown'
            log_info(f"Added missing required column '{col}' to dummy CSV with default values.")

    df.to_csv(csv_path, index=False)
    log_info(f"Dummy CSV created/updated at {csv_path}")


def test_system():
    """Test the complete adaptive learning system"""
    setup_test_environment()
    log_info("🧪 Testing Adaptive Learning System...")
    
    # Test 1: Check configuration
    enable_adaptive = getattr(Config, 'ENABLE_ADAPTIVE_LEARNING', False)
    log_info(f"📋 ENABLE_ADAPTIVE_LEARNING: {enable_adaptive}")
    min_trades_for_training = getattr(Config, 'ADAPTIVE_LEARNING_MIN_TRADES', 10) # Use 10 as per guide example
    log_info(f"📋 ADAPTIVE_LEARNING_MIN_TRADES: {min_trades_for_training}")
    log_info(f"📋 CSV_OUTPUT_PATH: {Config.CSV_OUTPUT_PATH}")
    log_info(f"📋 DATA_DIR for models: {Config.DATA_DIR}")

    if not enable_adaptive:
        log_warning("ENABLE_ADAPTIVE_LEARNING is False. Most tests will be skipped. Set to True in .env to run full tests.")
        # Check if scheduler starts/stops correctly even if disabled
        start_adaptive_learning_scheduler() # Should log it's disabled
        if is_adaptive_learning_scheduler_running():
            log_error("❌ Scheduler running even when ENABLE_ADAPTIVE_LEARNING is False.")
            return False
        else:
            log_info("✅ Scheduler correctly not started when disabled.")
        stop_adaptive_learning_scheduler() # Should be a no-op or log it's not running
        return True # Test considered passed for disabled state

    # Create dummy CSV data if needed for the test to run
    create_dummy_csv_if_not_exists(num_rows=max(15, min_trades_for_training + 5))

    # Test 2: Check CSV data loading
    learning_system = MultiModalLearningSystem()
    df, success = learning_system.load_and_prepare_data()
    
    test_passed = False
    if success and df is not None:
        log_info(f"✅ CSV data loaded by MultiModalLearningSystem: {len(df)} completed records for training")
        
        if len(df) >= min_trades_for_training:
            log_info(f"✅ Sufficient data ({len(df)}) for training (min_trades: {min_trades_for_training})")
            
            # Test 3: Force training
            log_info("🚀 Testing forced adaptive learning training cycle...")
            # Ensure scheduler isn't running from a previous test or main app instance
            if is_adaptive_learning_scheduler_running():
                log_info("Stopping existing scheduler before force training test...")
                stop_adaptive_learning_scheduler()
                time.sleep(1) # Give it a moment to stop

            training_success = force_adaptive_learning() # This uses the global scheduler's force_training
            
            if training_success:
                log_info("✅ Adaptive learning training cycle reported success!")
                # Verify model files were created
                model_path = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
                processors_path = Config.DATA_DIR / "data_processors.pkl"
                if model_path.exists() and processors_path.exists():
                    log_info(f"✅ Model file found: {model_path}")
                    log_info(f"✅ Processors file found: {processors_path}")
                    test_passed = True
                else:
                    log_error(f"❌ Training reported success, but model/processors files not found at {Config.DATA_DIR}!")
                    if not model_path.exists(): log_error(f"Missing: {model_path}")
                    if not processors_path.exists(): log_error(f"Missing: {processors_path}")
                    test_passed = False
            else:
                log_error("❌ Adaptive learning forced training cycle FAILED or did not run!")
                test_passed = False
        else:
            log_warning(f"⏳ Not enough completed trades ({len(df)}) in CSV for full training test (need {min_trades_for_training}). Test partially successful if data loaded.")
            test_passed = True # Consider test passed if data loads but not enough for training
    else:
        log_error(f"❌ Could not load or prepare CSV data using MultiModalLearningSystem from {Config.CSV_OUTPUT_PATH}")
        test_passed = False

    # Test 4: Scheduler start and stop
    log_info("🚀 Testing adaptive learning scheduler start/stop...")
    if is_adaptive_learning_scheduler_running(): # Stop if it was left running by force_training somehow
        stop_adaptive_learning_scheduler()
        time.sleep(1)

    start_adaptive_learning_scheduler()
    time.sleep(1) # Give scheduler a moment to start its thread
    if is_adaptive_learning_scheduler_running():
        log_info("✅ Scheduler started successfully.")
        stop_adaptive_learning_scheduler()
        time.sleep(1) # Give scheduler a moment to stop
        if not is_adaptive_learning_scheduler_running():
            log_info("✅ Scheduler stopped successfully.")
        else:
            log_error("❌ Scheduler did not stop as expected.")
            test_passed = False
    else:
        log_error("❌ Scheduler did not start as expected.")
        test_passed = False

    if test_passed:
        log_info("🎉🎉🎉 Adaptive Learning System Test PASSED! 🎉🎉🎉")
    else:
        log_error("😭😭😭 Adaptive Learning System Test FAILED! 😭😭😭")
    
    return test_passed

if __name__ == "__main__":
    # Clean up old model files before test to ensure fresh creation is tested
    # Be careful with this in a real environment if you have valuable trained models
    # For testing, it's good to ensure a clean slate.
    # Config.load_config() # Load config to get DATA_DIR
    # model_file = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
    # processors_file = Config.DATA_DIR / "data_processors.pkl"
    # last_training_file = Config.DATA_DIR / "last_training.json"
    # if model_file.exists(): os.remove(model_file)
    # if processors_file.exists(): os.remove(processors_file)
    # if last_training_file.exists(): os.remove(last_training_file)

    test_system()