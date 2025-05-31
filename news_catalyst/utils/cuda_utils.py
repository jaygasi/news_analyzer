# Fix for CUDA Detection and Setup

# 1. Enhanced CUDA detection and setup
# Create: utils/cuda_utils.py

"""
CUDA detection and setup utilities
"""
import sys
import subprocess
import platform
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    nvml = None

from utils.log_utils import logi, logw, loge, logd


class CUDADiagnostics:
    """CUDA diagnostics and setup utilities."""
    
    @staticmethod
    def check_cuda_installation() -> Dict[str, Any]:
        """Comprehensive CUDA installation check."""
        diagnostics = {
            'torch_available': TORCH_AVAILABLE,
            'cuda_available': False,
            'cuda_device_count': 0,
            'cuda_version': None,
            'cudnn_version': None,
            'gpu_names': [],
            'driver_version': None,
            'recommendations': []
        }
        
        if not TORCH_AVAILABLE:
            diagnostics['recommendations'].append(
                "Install PyTorch: pip install torch torchvision torchaudio"
            )
            return diagnostics
        
        # Check CUDA availability in PyTorch
        try:
            diagnostics['cuda_available'] = torch.cuda.is_available()
            diagnostics['cuda_device_count'] = torch.cuda.device_count()
            
            if torch.cuda.is_available():
                diagnostics['cuda_version'] = torch.version.cuda
                
                # Get GPU information
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    diagnostics['gpu_names'].append(gpu_name)
                
                # Check cuDNN
                if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                    diagnostics['cudnn_version'] = torch.backends.cudnn.version()
                
            else:
                diagnostics['recommendations'].extend([
                    "CUDA not available in PyTorch installation",
                    "Install CUDA-enabled PyTorch from: https://pytorch.org/get-started/locally/"
                ])
                
        except Exception as e:
            diagnostics['recommendations'].append(f"Error checking CUDA: {e}")
        
        # Check NVIDIA driver
        try:
            if NVML_AVAILABLE:
                nvml.nvmlInit()
                driver_version = nvml.nvmlSystemGetDriverVersion()
                diagnostics['driver_version'] = driver_version.decode('utf-8')
            else:
                # Try command line
                if platform.system() == "Windows":
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=10)
                else:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                          capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    diagnostics['driver_version'] = result.stdout.strip()
                
        except Exception as e:
            diagnostics['recommendations'].append("NVIDIA drivers may not be installed or accessible")
        
        return diagnostics
    
    @staticmethod
    def get_optimal_device() -> str:
        """Get the optimal device for PyTorch operations."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if torch.cuda.is_available():
            # Check GPU memory and select best device
            best_device = "cuda:0"
            max_memory = 0
            
            for i in range(torch.cuda.device_count()):
                try:
                    memory = torch.cuda.get_device_properties(i).total_memory
                    if memory > max_memory:
                        max_memory = memory
                        best_device = f"cuda:{i}"
                except Exception:
                    continue
            
            return best_device
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        return "cpu"
    
    @staticmethod
    def log_cuda_status() -> None:
        """Log comprehensive CUDA status."""
        diagnostics = CUDADiagnostics.check_cuda_installation()
        
        logi("🔍 CUDA Diagnostics:")
        logi(f"   PyTorch Available: {'✅' if diagnostics['torch_available'] else '❌'}")
        
        if diagnostics['torch_available']:
            logi(f"   CUDA Available: {'✅' if diagnostics['cuda_available'] else '❌'}")
            
            if diagnostics['cuda_available']:
                logi(f"   CUDA Version: {diagnostics['cuda_version']}")
                logi(f"   GPU Count: {diagnostics['cuda_device_count']}")
                logi(f"   Driver Version: {diagnostics['driver_version'] or 'Unknown'}")
                
                for i, gpu_name in enumerate(diagnostics['gpu_names']):
                    memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logi(f"   GPU {i}: {gpu_name} ({memory_gb:.1f} GB)")
                
                if diagnostics['cudnn_version']:
                    logi(f"   cuDNN Version: {diagnostics['cudnn_version']}")
            else:
                logw("   CUDA not available - using CPU")
        
        # Log recommendations
        if diagnostics['recommendations']:
            logw("💡 CUDA Setup Recommendations:")
            for rec in diagnostics['recommendations']:
                logw(f"   • {rec}")
    
    @staticmethod
    def install_cuda_pytorch() -> str:
        """Generate PyTorch installation command for CUDA."""
        system = platform.system().lower()
        
        commands = {
            'windows': {
                'cuda11.8': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                'cuda12.1': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
                'cpu': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
            },
            'linux': {
                'cuda11.8': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
                'cuda12.1': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
                'cpu': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
            },
            'darwin': {  # macOS
                'mps': 'pip install torch torchvision torchaudio',
                'cpu': 'pip install torch torchvision torchaudio'
            }
        }
        
        return commands.get(system, commands['linux'])


# 2. Enhanced main.py initialization with better CUDA handling
# Update the initialize_ai_models method in main.py:

async def initialize_ai_models(self) -> tuple[Any, Any, str]:
    """Enhanced AI models initialization with better CUDA handling."""
    model = None
    tokenizer = None
    device = "cpu"
    
    # Log CUDA diagnostics first
    if TORCH_AVAILABLE:
        CUDADiagnostics.log_cuda_status()
        device = CUDADiagnostics.get_optimal_device()
        logi(f"🎯 Selected device: {device}")
    
    if TORCH_AVAILABLE and self.config.ai.finbert_weight > 0:
        try:
            logi("🤖 Loading FinBERT model...")
            
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            
            # Load model with appropriate device handling
            if device.startswith('cuda'):
                logi(f"Loading FinBERT model on {device}...")
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                model = model.to(device)
                
                # Test CUDA functionality
                try:
                    test_input = tokenizer("test", return_tensors="pt").to(device)
                    with torch.no_grad():
                        _ = model(**test_input)
                    logi("✅ CUDA functionality verified")
                except Exception as e:
                    logw(f"CUDA test failed, falling back to CPU: {e}")
                    device = "cpu"
                    model = model.to(device)
            else:
                logi(f"Loading FinBERT model on {device}...")
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                model = model.to(device)
            
            logi(f"✅ FinBERT model loaded successfully on {device}")
            
        except Exception as e:
            logw(f"Failed to load FinBERT model: {e}")
            model = None
            tokenizer = None
            device = "cpu"
            
            # Provide helpful error messages
            if "CUDA" in str(e):
                logw("💡 CUDA error detected. Try:")
                logw("   1. Install CUDA drivers")
                logw("   2. Install CUDA-enabled PyTorch")
                logw("   3. Check GPU compatibility")
    else:
        if not TORCH_AVAILABLE:
            logw("PyTorch not available - install with: pip install torch")
        else:
            logw("FinBERT disabled in configuration")
    
    return model, tokenizer, device


# 3. CUDA setup script
# Create: setup_cuda.py

#!/usr/bin/env python3
"""
CUDA setup and verification script for News Catalyst Trading System
"""
import subprocess
import sys
import platform
from utils.cuda_utils import CUDADiagnostics

def main():
    """Main CUDA setup function."""
    print("🔍 CUDA Setup and Diagnostics")
    print("=" * 50)
    
    # Run diagnostics
    diagnostics = CUDADiagnostics.check_cuda_installation()
    
    print(f"PyTorch Available: {'✅' if diagnostics['torch_available'] else '❌'}")
    print(f"CUDA Available: {'✅' if diagnostics['cuda_available'] else '❌'}")
    
    if diagnostics['cuda_available']:
        print(f"CUDA Version: {diagnostics['cuda_version']}")
        print(f"GPU Count: {diagnostics['cuda_device_count']}")
        print(f"Driver Version: {diagnostics['driver_version']}")
        
        for i, gpu_name in enumerate(diagnostics['gpu_names']):
            print(f"GPU {i}: {gpu_name}")
    
    # Provide recommendations
    if diagnostics['recommendations']:
        print("\n💡 Recommendations:")
        for rec in diagnostics['recommendations']:
            print(f"   • {rec}")
    
    # Installation commands
    if not diagnostics['cuda_available']:
        print("\n🚀 CUDA PyTorch Installation Commands:")
        commands = CUDADiagnostics.install_cuda_pytorch()
        
        print("\nFor CUDA 11.8:")
        print(f"   {commands['cuda11.8']}")
        print("\nFor CUDA 12.1:")
        print(f"   {commands['cuda12.1']}")
        print("\nFor CPU only:")
        print(f"   {commands['cpu']}")
        
        print("\n🔗 More info: https://pytorch.org/get-started/locally/")
    
    return diagnostics['cuda_available']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)