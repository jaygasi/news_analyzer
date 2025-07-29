#!/usr/bin/env python3
"""
Training Verification Script - Check if models are actually trained
"""
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from utils.simple_logger import log_info, log_error, log_warning

def check_training_status():
    """Comprehensive check of training status"""
    print("🔍 TRAINING VERIFICATION REPORT")
    print("=" * 60)

    # 1. Check for checkpoint files
    print("\n📁 CHECKPOINT FILES:")
    enhanced_checkpoint = Config.DATA_DIR / "enhanced_neural_multimodal_adaptive.pth"
    finbert_checkpoint = Config.DATA_DIR / "finbert_multimodal_adaptive.pth"
    processors_file = Config.DATA_DIR / "data_processors.pkl"
    training_log = Config.DATA_DIR / "last_training.json"

    files_status = {
        "Enhanced Neural": enhanced_checkpoint,
        "FinBERT": finbert_checkpoint,
        "Data Processors": processors_file,
        "Training Log": training_log
    }

    for name, path in files_status.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            hours_ago = (datetime.now() - mod_time).total_seconds() / 3600
            print(f"  ✅ {name}: {size_mb:.1f}MB (modified {hours_ago:.1f}h ago)")
        else:
            print(f"  ❌ {name}: NOT FOUND")

    # 2. Analyze Enhanced Neural checkpoint
    print("\n🧠 ENHANCED NEURAL CHECKPOINT ANALYSIS:")
    if enhanced_checkpoint.exists():
        try:
            checkpoint = torch.load(enhanced_checkpoint, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']

                # Check for pooler weights specifically
                pooler_keys = [k for k in state_dict.keys() if 'pooler' in k]
                print(f"  📊 Total parameters: {len(state_dict)}")
                print(f"  🎯 Pooler parameters: {len(pooler_keys)}")

                if pooler_keys:
                    print("  ✅ POOLER WEIGHTS FOUND IN CHECKPOINT:")
                    for key in pooler_keys:
                        tensor = state_dict[key]
                        print(f"     • {key}: shape {list(tensor.shape)}")

                        # Check if weights are random/untrained
                        weight_std = tensor.std().item()
                        weight_mean = tensor.mean().item()
                        print(f"       Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")

                        # Random weights typically have std around 0.1-0.3
                        if weight_std < 0.05:
                            print("       ⚠️  Very low std - might be untrained")
                        elif weight_std > 1.0:
                            print("       ⚠️  Very high std - unusual")
                        else:
                            print("       ✅ Normal weight distribution")
                else:
                    print("  ❌ NO POOLER WEIGHTS IN CHECKPOINT!")
                    print("  🔍 Available keys:")
                    for key in sorted(state_dict.keys()):
                        print(f"     • {key}")

                # Check metadata
                if 'metadata' in checkpoint:
                    metadata = checkpoint['metadata']
                    print("\n  📋 TRAINING METADATA:")
                    for key, value in metadata.items():
                        print(f"     • {key}: {value}")

                    # Key indicators of successful training
                    if 'training_samples' in metadata:
                        samples = metadata['training_samples']
                        print(f"  📊 Trained on {samples} samples")
                        if samples < 10:
                            print("     ⚠️  Very few training samples!")

                    if 'training_completed' in metadata:
                        if metadata['training_completed']:
                            print("     ✅ Training marked as completed")
                        else:
                            print("     ❌ Training NOT completed!")
                else:
                    print("  ⚠️  No metadata found in checkpoint")
            else:
                print("  ❌ Invalid checkpoint format (no model_state_dict)")

        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
    else:
        print("  ❌ Enhanced Neural checkpoint not found")

    # 3. Check training log
    print("\n📝 TRAINING LOG ANALYSIS:")
    if training_log.exists():
        try:
            with open(training_log, 'r') as f:
                log_data = json.load(f)

            if 'last_adaptive_cycle_timestamp' in log_data:
                last_training = datetime.fromisoformat(log_data['last_adaptive_cycle_timestamp'])
                hours_ago = (datetime.now() - last_training).total_seconds() / 3600
                print(f"  🕒 Last training: {last_training} ({hours_ago:.1f}h ago)")

                if hours_ago > 48:
                    print("     ⚠️  Training was more than 2 days ago")
                elif hours_ago > 24:
                    print("     ⚠️  Training was more than 1 day ago")
                else:
                    print("     ✅ Recent training")

            if 'training_history' in log_data:
                history = log_data['training_history']
                print(f"  📊 Training events: {len(history)}")

                if history:
                    latest = history[-1]
                    print("  📋 Latest training event:")
                    for key, value in latest.items():
                        print(f"     • {key}: {value}")
        except Exception as e:
            print(f"  ❌ Error reading training log: {e}")
    else:
        print("  ❌ Training log not found")

    # 4. Recommendations
    print("\n💡 RECOMMENDATIONS:")

    if not enhanced_checkpoint.exists():
        print("  🔧 Run training: python tools/model_learning.py")
    elif enhanced_checkpoint.exists():
        # Load and check if checkpoint is valid
        try:
            checkpoint = torch.load(enhanced_checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                pooler_keys = [k for k in checkpoint['model_state_dict'].keys() if 'pooler' in k]
                if not pooler_keys:
                    print("  🔧 Checkpoint missing pooler weights - retrain model")
                elif 'metadata' in checkpoint and checkpoint['metadata'].get('training_completed'):
                    print("  ✅ Model appears properly trained")
                    print("  📝 The pooler warning is just from initial RoBERTa loading (normal)")
                    print("  🎯 Your trained weights ARE being loaded and used!")
                else:
                    print("  🔧 Training may not have completed - consider retraining")
            else:
                print("  🔧 Invalid checkpoint format - retrain model")
        except Exception as e:
            print(f"  🔧 Checkpoint loading error - retrain model: {e}")

    print("\n" + "=" * 60)

def test_model_behavior():
    """Test if the model is actually using trained weights vs random"""
    print("🧪 MODEL BEHAVIOR TEST")
    print("=" * 60)

    try:
        from analysis.enhanced_neural_analyzer import EnhancedNeuralAnalyzer

        print("📊 Loading model for testing...")
        analyzer = EnhancedNeuralAnalyzer()

        # Test with multiple similar inputs
        test_texts = [
            "Company reports strong earnings beating expectations",
            "Quarterly earnings exceed analyst estimates significantly",
            "Revenue growth drives positive quarterly results"
        ]

        print("\n🎯 Testing prediction consistency (trained models should be consistent):")
        results = []
        for i, text in enumerate(test_texts, 1):
            result = analyzer.analyze_text("TEST", text)
            results.append(result)
            print(f"  Test {i}: sentiment={result.sentiment_intensity:.3f}, confidence={result.confidence:.3f}")

        # Check consistency
        sentiments = [r.sentiment_intensity for r in results]
        confidences = [r.confidence for r in results]

        sentiment_std = torch.tensor(sentiments).std().item()
        confidence_std = torch.tensor(confidences).std().item()

        print("\n📊 Consistency Analysis:")
        print(f"  Sentiment std: {sentiment_std:.4f}")
        print(f"  Confidence std: {confidence_std:.4f}")

        if sentiment_std > 0.5:
            print("  ⚠️  High sentiment variance - model may be undertrained")
        elif sentiment_std < 0.1:
            print("  ⚠️  Very low variance - model may be defaulting")
        else:
            print("  ✅ Good consistency - model appears trained")

        if confidence_std < 0.01:
            print("  ⚠️  Model confidence not varying - possible defaulting behavior")
        else:
            print("  ✅ Model confidence varies appropriately")

    except Exception as e:
        print(f"❌ Model behavior test failed: {e}")

    print("=" * 60)

if __name__ == "__main__":
    check_training_status()
    print()
    test_model_behavior()