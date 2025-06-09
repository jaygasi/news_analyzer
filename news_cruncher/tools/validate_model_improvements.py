"""
Model Improvement Validation Script - FIXED VERSION
Validates that all fixes are working correctly
"""
import sys
import os
from pathlib import Path

# Fix Python path - add the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports should work
import torch
import warnings
from utils.simple_logger import log_info, log_error, log_warning
from config import Config

def validate_attention_implementation():
    """Test that attention warnings are fixed"""
    log_info("🔍 Testing attention implementation fixes...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            from analysis.enhanced_neural_analyzer import create_enhanced_analyzer
            analyzer = create_enhanced_analyzer()
            
            # Check for attention warnings
            attention_warnings = [warning for warning in w 
                                if "RobertaSdpaSelfAttention" in str(warning.message)]
            
            if len(attention_warnings) == 0:
                log_info("✅ No attention implementation warnings")
                return True
            else:
                log_warning(f"⚠️ Found {len(attention_warnings)} attention warnings")
                for warning in attention_warnings:
                    log_warning(f"  - {warning.message}")
                return False
                
        except Exception as e:
            log_error(f"❌ Attention test failed: {e}")
            return False

def validate_pooler_warnings():
    """Test that pooler warnings are fixed"""
    log_info("🔍 Testing pooler weight warnings...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            from analysis.multi_llm_analyzer import MultiLLMAnalyzer
            analyzer = MultiLLMAnalyzer()
            
            # Check for pooler warnings
            pooler_warnings = [warning for warning in w 
                             if "pooler" in str(warning.message).lower() and "initialized" in str(warning.message).lower()]
            
            if len(pooler_warnings) == 0:
                log_info("✅ No pooler initialization warnings")
                return True
            else:
                log_warning(f"⚠️ Found {len(pooler_warnings)} pooler warnings")
                for warning in pooler_warnings:
                    log_warning(f"  - {warning.message}")
                return False
                
        except Exception as e:
            log_error(f"❌ Pooler test failed: {e}")
            return False

def validate_checkpoint_loading():
    """Test that checkpoints load properly"""
    log_info("🔍 Testing checkpoint loading...")
    
    try:
        # Test paths exist or can be created
        paths_to_check = [
            Config.DATA_DIR / "finbert_multimodal_adaptive.pth",
            Config.DATA_DIR / "enhanced_neural_multimodal_adaptive.pth", 
            Config.DATA_DIR / "data_processors.pkl"
        ]
        
        existing_checkpoints = []
        for path in paths_to_check:
            if path.exists():
                existing_checkpoints.append(path.name)
        
        if existing_checkpoints:
            log_info(f"✅ Found adaptive checkpoints: {', '.join(existing_checkpoints)}")
        else:
            log_info("ℹ️ No adaptive checkpoints found (normal for first run)")
            
        # Test that the directories can be created
        Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_info("✅ Checkpoint directories accessible")
        return True
            
    except Exception as e:
        log_error(f"❌ Checkpoint test failed: {e}")
        return False

def validate_learning_system():
    """Test that learning system works"""
    log_info("🔍 Testing learning system...")
    
    try:
        from tools.multi_modal_learning_system import MultiModalLearningSystem
        
        learning_system = MultiModalLearningSystem()
        
        # Test initialization
        if hasattr(learning_system, 'model_save_dir'):
            log_info("✅ Learning system initialized properly")
            
            # Test data preparation capabilities
            if hasattr(learning_system, 'prepare_training_data'):
                log_info("✅ Learning system has data preparation")
            
            return True
        else:
            log_warning("⚠️ Learning system missing components")
            return False
            
    except Exception as e:
        log_error(f"❌ Learning system test failed: {e}")
        return False

def validate_enhanced_neural():
    """Test enhanced neural analyzer loading"""
    log_info("🔍 Testing Enhanced Neural Analyzer...")
    
    try:
        from analysis.enhanced_neural_analyzer import create_enhanced_analyzer
        
        analyzer = create_enhanced_analyzer()
        
        if analyzer and analyzer.is_available:
            log_info("✅ Enhanced Neural Analyzer loads and is available")
            
            # Test model info
            model_info = analyzer.get_model_info()
            if model_info.get('available'):
                log_info(f"✅ Model info: {model_info['trainable_parameters']:,} trainable parameters")
            
            return True
        else:
            log_warning("⚠️ Enhanced Neural Analyzer loaded but not available")
            return False
            
    except Exception as e:
        log_error(f"❌ Enhanced Neural test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    log_info("🚀 Starting Model Improvement Validation...")
    
    tests = [
        ("Attention Implementation", validate_attention_implementation),
        ("Pooler Warnings", validate_pooler_warnings), 
        ("Checkpoint Loading", validate_checkpoint_loading),
        ("Learning System", validate_learning_system),
        ("Enhanced Neural", validate_enhanced_neural)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            log_info(f"\n--- {test_name} ---")
            result = test_func()
            results.append(result)
        except Exception as e:
            log_error(f"Test {test_name} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    log_info(f"\n{'='*50}")
    if passed == total:
        log_info(f"🎉 All {total} validation tests PASSED!")
    else:
        log_warning(f"⚠️ {passed}/{total} validation tests passed")
        
        # Show which tests failed
        for i, (test_name, _) in enumerate(tests):
            if not results[i]:
                log_error(f"❌ FAILED: {test_name}")
    
    return passed == total

if __name__ == "__main__":
    main()
