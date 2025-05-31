# Fix for Performance Monitoring psutil Issues

# 1. Enhanced performance monitor with fallback when psutil unavailable
# Create/Update: utils/performance_monitor.py

import time
import threading
import gc
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import logging

# Try to import psutil, fallback gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from utils.log_utils import logi, logw, loge, logd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    thread_count: Optional[int] = None
    component_metrics: Dict[str, Any] = field(default_factory=dict)


class FallbackPerformanceMonitor:
    """Lightweight performance monitor when psutil is not available."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: deque = deque(maxlen=100)
        self.component_metrics: Dict[str, Any] = defaultdict(dict)
        self._lock = threading.RLock()
    
    def get_basic_metrics(self) -> PerformanceMetrics:
        """Get basic performance metrics without psutil."""
        try:
            # Get thread count
            thread_count = threading.active_count()
            
            # Get approximate memory usage from garbage collector
            gc_stats = gc.get_stats()
            object_count = sum(stat.get('collections', 0) for stat in gc_stats)
            
            # Basic timing information
            uptime_seconds = time.time() - self.start_time
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=None,  # Not available without psutil
                memory_percent=None,  # Not available without psutil
                memory_mb=None,  # Not available without psutil
                disk_usage_percent=None,  # Not available without psutil
                thread_count=thread_count,
                component_metrics={
                    'uptime_seconds': uptime_seconds,
                    'gc_object_count': object_count,
                    'python_version': sys.version,
                    'thread_count': thread_count
                }
            )
            
            return metrics
            
        except Exception as e:
            logw(f"Error getting basic metrics: {e}")
            return PerformanceMetrics()
    
    def log_performance_summary(self) -> None:
        """Log basic performance summary."""
        try:
            metrics = self.get_basic_metrics()
            uptime_hours = metrics.component_metrics.get('uptime_seconds', 0) / 3600
            
            logi(f"📊 Basic Performance Summary:")
            logi(f"   Uptime: {uptime_hours:.1f} hours")
            logi(f"   Active Threads: {metrics.thread_count}")
            logi(f"   GC Collections: {metrics.component_metrics.get('gc_object_count', 'N/A')}")
            
        except Exception as e:
            logw(f"Error logging performance summary: {e}")


class EnhancedPerformanceMonitor:
    """Enhanced performance monitor with psutil support and fallback."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.is_running = False
        self.metrics_history: deque = deque(maxlen=1000)
        self.component_metrics: Dict[str, Any] = defaultdict(dict)
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Initialize appropriate monitor
        if PSUTIL_AVAILABLE:
            logi("✅ psutil available - full performance monitoring enabled")
            self.monitor = self._get_full_metrics
        else:
            logw("⚠️ psutil not available - using basic performance monitoring")
            self.fallback_monitor = FallbackPerformanceMonitor()
            self.monitor = self._get_basic_metrics
    
    def _get_full_metrics(self) -> PerformanceMetrics:
        """Get comprehensive metrics using psutil."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            thread_count = process.num_threads()
            process_memory = process.memory_info()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=process_memory.rss / 1024 / 1024,
                disk_usage_percent=disk.percent,
                thread_count=thread_count,
                component_metrics={
                    'system_memory_total_gb': memory.total / 1024 / 1024 / 1024,
                    'system_memory_available_gb': memory.available / 1024 / 1024 / 1024,
                    'process_memory_mb': process_memory.rss / 1024 / 1024,
                    'process_cpu_percent': process.cpu_percent(),
                    'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                    'disk_total_gb': disk.total / 1024 / 1024 / 1024
                }
            )
            
            return metrics
            
        except Exception as e:
            loge(f"Error getting full metrics: {e}")
            return self._get_basic_metrics()
    
    def _get_basic_metrics(self) -> PerformanceMetrics:
        """Get basic metrics when psutil is not available."""
        if hasattr(self, 'fallback_monitor'):
            return self.fallback_monitor.get_basic_metrics()
        else:
            return PerformanceMetrics()
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logi("📊 Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.is_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logi("📊 Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                metrics = self.monitor()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Log performance summary every 10 minutes
                if len(self.metrics_history) % 10 == 0:
                    self._log_performance_summary()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                loge(f"Error in performance monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _log_performance_summary(self) -> None:
        """Log performance summary."""
        try:
            if not self.metrics_history:
                return
            
            latest = self.metrics_history[-1]
            
            if PSUTIL_AVAILABLE and latest.cpu_percent is not None:
                logi(f"📊 Performance Summary:")
                logi(f"   CPU: {latest.cpu_percent:.1f}%")
                logi(f"   Memory: {latest.memory_percent:.1f}% ({latest.memory_mb:.1f} MB)")
                logi(f"   Threads: {latest.thread_count}")
                logi(f"   Disk: {latest.disk_usage_percent:.1f}%")
            else:
                # Basic summary
                if hasattr(self, 'fallback_monitor'):
                    self.fallback_monitor.log_performance_summary()
                else:
                    logi(f"📊 Basic Performance: {latest.thread_count} threads active")
                    
        except Exception as e:
            logw(f"Error logging performance summary: {e}")
    
    def register_component_performance(self, component_name: str, **metrics) -> None:
        """Register performance metrics for a component."""
        with self._lock:
            self.component_metrics[component_name].update({
                'timestamp': datetime.now(),
                **metrics
            })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            if not self.metrics_history:
                return {'status': 'no_data'}
            
            latest = self.metrics_history[-1]
            
            report = {
                'timestamp': latest.timestamp.isoformat(),
                'monitoring_type': 'full' if PSUTIL_AVAILABLE else 'basic',
                'thread_count': latest.thread_count,
                'component_metrics': dict(self.component_metrics)
            }
            
            if PSUTIL_AVAILABLE and latest.cpu_percent is not None:
                report.update({
                    'cpu_percent': latest.cpu_percent,
                    'memory_percent': latest.memory_percent,
                    'memory_mb': latest.memory_mb,
                    'disk_usage_percent': latest.disk_usage_percent
                })
            
            return report


# Global instance
_performance_monitor: Optional[EnhancedPerformanceMonitor] = None


def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = EnhancedPerformanceMonitor()
    
    _performance_monitor.start_monitoring()


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    global _performance_monitor
    
    if _performance_monitor:
        _performance_monitor.stop_monitoring()


def register_component_performance(component_name: str, **metrics) -> None:
    """Register component performance metrics."""
    global _performance_monitor
    
    if _performance_monitor:
        _performance_monitor.register_component_performance(component_name, **metrics)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    global _performance_monitor
    
    if _performance_monitor:
        return _performance_monitor.get_performance_report()
    else:
        return {'status': 'not_initialized'}


# 2. Update requirements.txt to include psutil
# Add this line to requirements.txt:
# psutil>=5.8.0

# 3. Update main.py to handle missing psutil gracefully
# Update the main.py imports:

# At the top of main.py, update the conditional imports:
try:
    from utils.performance_monitor import EnhancedPerformanceMonitor, start_performance_monitoring, stop_performance_monitoring
    PERFORMANCE_MONITOR_AVAILABLE = True
    logi("✅ Performance monitoring available")
except ImportError as e:
    PERFORMANCE_MONITOR_AVAILABLE = False
    logw(f"⚠️ Performance monitoring limited: {e}")
    
    # Create stub functions
    def start_performance_monitoring():
        logd("Performance monitoring stub - install psutil for full monitoring")
    
    def stop_performance_monitoring():
        logd("Performance monitoring stub stopped")

# 4. Installation instructions for the user
installation_instructions = """
# To enable full performance monitoring, install psutil:
pip install psutil>=5.8.0

# Or add to requirements.txt:
echo "psutil>=5.8.0" >> requirements.txt
pip install -r requirements.txt
"""