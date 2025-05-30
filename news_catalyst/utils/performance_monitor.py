"""
Enhanced performance monitoring with improved graceful fallback and optimizations
"""
import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
import json
import os
from enum import Enum
from pathlib import Path

from utils.log_utils import logi, logw, loge, logd

# Graceful import of psutil with comprehensive fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
    logi("✅ psutil available - full performance monitoring enabled")
except ImportError:
    PSUTIL_AVAILABLE = False
    logw("⚠️  psutil not available - using limited performance monitoring")


class AlertLevel(Enum):
    """Alert severity levels with numeric values for comparison"""
    INFO = 1
    WARNING = 2
    CRITICAL = 3


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics with validation"""
    timestamp: datetime
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    processing_rates: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate metrics values"""
        # Ensure percentages are valid
        for attr in ['cpu_percent', 'memory_percent', 'disk_usage_percent']:
            value = getattr(self, attr)
            if value < 0 or value > 100:
                setattr(self, attr, max(0, min(100, value)))


@dataclass
class Alert:
    """Performance alert with enhanced information"""
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    value: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'component': self.component,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold
        }


class AdaptiveThresholds:
    """Enhanced dynamic performance thresholds with statistical analysis"""
    
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Base thresholds (fallback values)
        self.base_thresholds: Dict[str, float] = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.05,
            'queue_size': 1000,
            'processing_time_ms': 1000
        }
        
        # Adaptive multipliers
        self.adaptive_multipliers: Dict[str, float] = defaultdict(lambda: 1.0)
        
        self._lock = threading.RLock()
    
    def update_baseline(self, metric_name: str, value: float) -> None:
        """Update baseline with thread safety and validation"""
        if not isinstance(value, (int, float)) or value < 0:
            return
        
        with self._lock:
            self.baseline_metrics[metric_name].append(value)
    
    def get_adaptive_threshold(self, metric_name: str) -> float:
        """Calculate adaptive threshold with statistical analysis"""
        with self._lock:
            baseline_data = self.baseline_metrics.get(metric_name)
            base_threshold = self.base_thresholds.get(metric_name, 100.0)
            
            if not baseline_data or len(baseline_data) < 10:
                return base_threshold
            
            # Statistical analysis
            values = list(baseline_data)
            mean_value = sum(values) / len(values)
            
            # Calculate standard deviation
            variance = sum((x - mean_value) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            # Adaptive threshold: mean + 2.5 standard deviations
            adaptive_threshold = mean_value + (2.5 * std_dev)
            
            # Apply adaptive multiplier and bounds
            multiplier = self.adaptive_multipliers[metric_name]
            adaptive_threshold *= multiplier
            
            # Ensure reasonable bounds
            min_threshold = base_threshold * 0.5
            max_threshold = base_threshold * 2.0
            
            return max(min_threshold, min(adaptive_threshold, max_threshold))
    
    def adjust_sensitivity(self, metric_name: str, increase: bool = True) -> None:
        """Adjust threshold sensitivity based on alert patterns"""
        with self._lock:
            current_multiplier = self.adaptive_multipliers[metric_name]
            
            if increase:
                # Make more sensitive (lower threshold)
                self.adaptive_multipliers[metric_name] = max(0.7, current_multiplier * 0.9)
            else:
                # Make less sensitive (higher threshold)
                self.adaptive_multipliers[metric_name] = min(1.5, current_multiplier * 1.1)


class FallbackPerformanceCollector:
    """Enhanced fallback collector with more comprehensive metrics"""
    
    def __init__(self) -> None:
        self._start_time = time.time()
        self._last_collection = time.time()
        
        # Track basic metrics without psutil
        self._process_start_memory = self._get_process_memory_fallback()
        
    def _get_process_memory_fallback(self) -> float:
        """Get process memory using resource module fallback"""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # Convert to MB (ru_maxrss is platform-dependent)
            if hasattr(usage, 'ru_maxrss'):
                if os.name == 'posix':
                    return usage.ru_maxrss / 1024  # KB to MB on Linux
                else:
                    return usage.ru_maxrss / (1024 * 1024)  # Bytes to MB on Windows
        except (ImportError, AttributeError):
            pass
        return 0.0
    
    def _get_thread_count(self) -> int:
        """Get current thread count"""
        try:
            import threading
            return threading.active_count()
        except Exception:
            return 0
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect enhanced fallback metrics"""
        current_time = time.time()
        
        try:
            # Basic system information
            uptime_hours = (current_time - self._start_time) / 3600
            thread_count = self._get_thread_count()
            memory_mb = self._get_process_memory_fallback()
            
            # Estimated CPU usage based on collection frequency
            collection_interval = current_time - self._last_collection
            estimated_cpu = min(50.0, max(0.0, (1.0 / max(collection_interval, 0.1)) * 10))
            
            self._last_collection = current_time
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=estimated_cpu,
                memory_percent=min(50.0, memory_mb / 10),  # Rough estimate
                memory_used_mb=memory_mb,
                disk_usage_percent=0.0,  # Cannot measure without psutil
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_threads=thread_count,
                open_files=0  # Cannot measure without psutil
            )
            
        except Exception as e:
            logd(f"Error in fallback performance collection: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                active_threads=self._get_thread_count()
            )


class PerformanceCollector:
    """Enhanced performance collector with comprehensive error handling"""
    
    def __init__(self) -> None:
        self._fallback = FallbackPerformanceCollector()
        
        if not PSUTIL_AVAILABLE:
            return
        
        # Initialize psutil components with error handling
        self._network_baseline: Optional[Any] = None
        self._disk_baseline: Optional[Any] = None
        self._process: Optional[Any] = None
        
        try:
            self._process = psutil.Process()
            self._initialize_baselines()
        except Exception as e:
            logw(f"Failed to initialize psutil process: {e}")
    
    def _initialize_baselines(self) -> None:
        """Initialize baseline measurements"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            self._network_baseline = psutil.net_io_counters()
        except (AttributeError, OSError) as e:
            logd(f"Network baseline initialization failed: {e}")
        
        try:
            self._disk_baseline = psutil.disk_io_counters()
        except (AttributeError, OSError) as e:
            logd(f"Disk baseline initialization failed: {e}")
    
    def _get_safe_cpu_percent(self) -> float:
        """Get CPU percentage with multiple fallbacks"""
        try:
            # Non-blocking call with timeout
            return psutil.cpu_percent(interval=None)
        except Exception:
            try:
                # Fallback to per-cpu average
                per_cpu = psutil.cpu_percent(percpu=True, interval=None)
                return sum(per_cpu) / len(per_cpu) if per_cpu else 0.0
            except Exception:
                return 0.0
    
    def _get_safe_memory_info(self) -> tuple[float, float]:
        """Get memory information with error handling"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent, memory.used / (1024 * 1024)  # MB
        except Exception:
            return 0.0, 0.0
    
    def _get_safe_disk_usage(self) -> float:
        """Get disk usage with multiple path fallbacks"""
        paths_to_try = ['/', '.', os.getcwd(), os.path.expanduser('~')]
        
        for path in paths_to_try:
            try:
                if os.path.exists(path):
                    disk = psutil.disk_usage(path)
                    return disk.percent
            except (OSError, PermissionError):
                continue
        
        return 0.0
    
    def _get_safe_network_stats(self) -> tuple[float, float]:
        """Get network statistics with error handling"""
        try:
            current = psutil.net_io_counters()
            if self._network_baseline is None:
                self._network_baseline = current
                return 0.0, 0.0
            
            sent_mb = max(0, (current.bytes_sent - self._network_baseline.bytes_sent) / (1024 * 1024))
            recv_mb = max(0, (current.bytes_recv - self._network_baseline.bytes_recv) / (1024 * 1024))
            
            return sent_mb, recv_mb
        except (AttributeError, OSError):
            return 0.0, 0.0
    
    def _get_safe_process_info(self) -> tuple[int, int]:
        """Get process-specific information with error handling"""
        if not self._process:
            return 0, 0
        
        threads = 0
        open_files = 0
        
        try:
            threads = self._process.num_threads()
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass
        
        try:
            open_files = len(self._process.open_files())
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            pass
        
        return threads, open_files
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive metrics with graceful degradation"""
        if not PSUTIL_AVAILABLE:
            return self._fallback.collect_metrics()
        
        try:
            # Collect all metrics with individual error handling
            cpu_percent = self._get_safe_cpu_percent()
            memory_percent, memory_mb = self._get_safe_memory_info()
            disk_percent = self._get_safe_disk_usage()
            network_sent, network_recv = self._get_safe_network_stats()
            threads, open_files = self._get_safe_process_info()
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_mb,
                disk_usage_percent=disk_percent,
                network_sent_mb=network_sent,
                network_recv_mb=network_recv,
                active_threads=threads,
                open_files=open_files
            )
            
        except Exception as e:
            loge(f"Error in psutil performance collection, falling back: {e}")
            return self._fallback.collect_metrics()


class EnhancedPerformanceMonitor:
    """Production-ready performance monitor with comprehensive features"""
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 history_size: int = 1000,
                 alert_callback: Optional[Callable[[Alert], None]] = None) -> None:
        
        # Validation
        if collection_interval <= 0:
            raise ValueError("collection_interval must be positive")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.alert_callback = alert_callback
        
        # State management
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # Data storage with thread-safe containers
        self.metrics_history: deque = deque(maxlen=history_size)
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts: deque = deque(maxlen=100)
        
        # Monitoring components
        self.collector = PerformanceCollector()
        self.thresholds = AdaptiveThresholds()
        
        # Thread safety and performance
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self._collection_count = 0
        self._error_count = 0
        self._alert_count = 0
        
        self._log_availability_status()
    
    def _log_availability_status(self) -> None:
        """Log performance monitoring capabilities"""
        if PSUTIL_AVAILABLE:
            logi("🔍 Enhanced performance monitoring active (psutil available)")
        else:
            logw("🔍 Limited performance monitoring active (psutil unavailable)")
    
    def register_component_metrics(self, component: str, **metrics: Union[int, float]) -> None:
        """Thread-safe component metric registration with validation"""
        if not component or not metrics:
            return
        
        with self._lock:
            timestamp = datetime.now()
            
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not isinstance(metric_name, str):
                    continue
                
                key = f"{component}_{metric_name}"
                metric_entry = {
                    'timestamp': timestamp,
                    'value': float(value),
                    'component': component,
                    'metric': metric_name
                }
                
                self.component_metrics[key].append(metric_entry)
                self.thresholds.update_baseline(key, float(value))
    
    def _evaluate_system_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """Evaluate system-level performance alerts"""
        alerts = []
        
        if not PSUTIL_AVAILABLE:
            return alerts  # Skip system alerts without psutil
        
        # System-level checks with adaptive thresholds
        checks = [
            ('cpu_percent', metrics.cpu_percent, "High CPU usage"),
            ('memory_percent', metrics.memory_percent, "High memory usage"),
            ('disk_usage_percent', metrics.disk_usage_percent, "High disk usage")
        ]
        
        for metric_name, value, message in checks:
            if value <= 0:  # Skip invalid measurements
                continue
            
            threshold = self.thresholds.get_adaptive_threshold(metric_name)
            
            if value > threshold:
                severity = AlertLevel.CRITICAL if value > threshold * 1.2 else AlertLevel.WARNING
                
                alert = Alert(
                    timestamp=datetime.now(),
                    level=severity,
                    component="system",
                    message=f"{message}: {value:.1f}% (threshold: {threshold:.1f}%)",
                    value=value,
                    threshold=threshold
                )
                alerts.append(alert)
        
        return alerts
    
    def _process_alerts(self, alerts: List[Alert]) -> None:
        """Process and handle alerts with callbacks and logging"""
        for alert in alerts:
            with self._lock:
                self.alerts.append(alert)
                self._alert_count += 1
            
            # Invoke callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    loge(f"Error in alert callback: {e}")
            
            # Log based on severity
            if alert.level == AlertLevel.CRITICAL:
                loge(alert.message)
            elif alert.level == AlertLevel.WARNING:
                logw(alert.message)
            else:
                logi(alert.message)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current system status"""
        with self._lock:
            if not self.metrics_history:
                return {
                    "status": "no_data",
                    "psutil_available": PSUTIL_AVAILABLE,
                    "monitoring_mode": "full" if PSUTIL_AVAILABLE else "limited"
                }
            
            latest_metrics = self.metrics_history[-1]
            recent_alerts = list(self.alerts)[-5:]
            
            # Calculate trends if sufficient data
            trends = {}
            if len(self.metrics_history) >= 2:
                prev_metrics = self.metrics_history[-2]
                trends = {
                    'cpu_trend': latest_metrics.cpu_percent - prev_metrics.cpu_percent,
                    'memory_trend': latest_metrics.memory_percent - prev_metrics.memory_percent,
                    'threads_trend': latest_metrics.active_threads - prev_metrics.active_threads
                }
            
            return {
                "timestamp": latest_metrics.timestamp.isoformat(),
                "psutil_available": PSUTIL_AVAILABLE,
                "monitoring_mode": "full" if PSUTIL_AVAILABLE else "limited",
                "collection_count": self._collection_count,
                "error_count": self._error_count,
                "alert_count": self._alert_count,
                "system": {
                    "cpu_percent": latest_metrics.cpu_percent,
                    "memory_percent": latest_metrics.memory_percent,
                    "memory_used_mb": latest_metrics.memory_used_mb,
                    "disk_usage_percent": latest_metrics.disk_usage_percent,
                    "active_threads": latest_metrics.active_threads,
                    "open_files": latest_metrics.open_files,
                    **trends
                },
                "components": {
                    name: values[-1] if values else None 
                    for name, values in self.component_metrics.items()
                },
                "recent_alerts": [alert.to_dict() for alert in recent_alerts],
                "status": "healthy" if not any(a.level == AlertLevel.CRITICAL for a in recent_alerts) else "degraded"
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with statistics"""
        with self._lock:
            if not self.metrics_history:
                return {
                    "error": "No metrics available",
                    "psutil_available": PSUTIL_AVAILABLE,
                    "monitoring_mode": "limited"
                }
            
            # Analyze recent metrics window
            recent_window = min(60, len(self.metrics_history))
            recent_metrics = list(self.metrics_history)[-recent_window:]
            
            # Calculate comprehensive statistics
            def calculate_stats(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {}
                return {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'minimum': min(values),
                    'maximum': max(values),
                    'samples': len(values)
                }
            
            # System metrics analysis
            cpu_values = [m.cpu_percent for m in recent_metrics if m.cpu_percent > 0]
            memory_values = [m.memory_percent for m in recent_metrics if m.memory_percent > 0]
            thread_values = [m.active_threads for m in recent_metrics if m.active_threads > 0]
            
            system_summary = {
                'psutil_available': PSUTIL_AVAILABLE,
                'monitoring_mode': "full" if PSUTIL_AVAILABLE else "limited",
                'collection_statistics': {
                    'total_collections': self._collection_count,
                    'error_count': self._error_count,
                    'success_rate': (self._collection_count - self._error_count) / max(self._collection_count, 1)
                },
                'cpu_stats': calculate_stats(cpu_values),
                'memory_stats': calculate_stats(memory_values),
                'thread_stats': calculate_stats(thread_values)
            }
            
            # Component performance analysis
            component_summary = {}
            for component_name, values in self.component_metrics.items():
                if values and len(values) >= 1:
                    recent_values = list(values)[-min(10, len(values)):]
                    numeric_values = []
                    
                    for v in recent_values:
                        if isinstance(v, dict) and 'value' in v:
                            numeric_values.append(v['value'])
                        elif isinstance(v, (int, float)):
                            numeric_values.append(v)
                    
                    if numeric_values:
                        component_summary[component_name] = calculate_stats(numeric_values)
            
            # Alert analysis
            alert_summary = {
                'total_alerts': len(self.alerts),
                'alert_breakdown': {
                    level.name: sum(1 for a in self.alerts if a.level == level)
                    for level in AlertLevel
                },
                'recent_alerts': [alert.to_dict() for alert in list(self.alerts)[-5:]]
            }
            
            return {
                'report_timestamp': datetime.now().isoformat(),
                'system_summary': system_summary,
                'component_summary': component_summary,
                'alert_summary': alert_summary,
                'adaptive_thresholds': dict(self.thresholds.base_thresholds)
            }
    
    def _monitor_loop(self) -> None:
        """Enhanced monitoring loop with comprehensive error handling"""
        mode = "full" if PSUTIL_AVAILABLE else "limited"
        logi(f"🔍 Performance monitor started ({mode} mode)")
        
        while not self._shutdown_event.is_set():
            try:
                collection_start = time.time()
                
                # Collect metrics
                metrics = self.collector.collect_metrics()
                
                if metrics:
                    with self._lock:
                        self.metrics_history.append(metrics)
                        self._collection_count += 1
                    
                    # Update adaptive thresholds
                    if PSUTIL_AVAILABLE:
                        self.thresholds.update_baseline('cpu_percent', metrics.cpu_percent)
                        self.thresholds.update_baseline('memory_percent', metrics.memory_percent)
                        self.thresholds.update_baseline('disk_usage_percent', metrics.disk_usage_percent)
                    
                    # Evaluate and process alerts
                    alerts = self._evaluate_system_alerts(metrics)
                    if alerts:
                        self._process_alerts(alerts)
                else:
                    with self._lock:
                        self._error_count += 1
                
                # Calculate sleep time to maintain consistent interval
                collection_time = time.time() - collection_start
                sleep_time = max(0, self.collection_interval - collection_time)
                
                if self._shutdown_event.wait(sleep_time):
                    break
                    
            except Exception as e:
                with self._lock:
                    self._error_count += 1
                
                loge(f"Error in performance monitor loop: {e}")
                
                # Error recovery delay
                if self._shutdown_event.wait(min(self.collection_interval * 2, 30)):
                    break
        
        logi("🔍 Performance monitor stopped")
    
    def start(self) -> None:
        """Start performance monitoring with validation"""
        if self.is_running:
            logw("Performance monitor already running")
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        mode = "full" if PSUTIL_AVAILABLE else "limited"
        logi(f"🔍 Performance monitoring started ({mode} mode)")
    
    def stop(self, timeout: float = 10.0) -> None:
        """Stop performance monitoring gracefully with timeout"""
        if not self.is_running:
            return
        
        logi("🛑 Stopping performance monitor...")
        self.is_running = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
            if self._monitor_thread.is_alive():
                logw("Performance monitor thread did not stop within timeout")
        
        # Log final statistics
        with self._lock:
            logi(f"✅ Performance monitor stopped - "
                 f"Collections: {self._collection_count}, "
                 f"Errors: {self._error_count}, "
                 f"Alerts: {self._alert_count}")
    
    def save_report(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Save performance report to file with enhanced error handling"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"performance_report_{timestamp}.json"
        
        # Convert to Path object for better handling
        filepath = Path(filepath)
        
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate and save report
            report = self.get_performance_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            
            logi(f"📊 Performance report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            loge(f"Error saving performance report to {filepath}: {e}")
            raise


# Global performance monitor instance with lazy initialization
_performance_monitor: Optional[EnhancedPerformanceMonitor] = None
_monitor_lock = threading.Lock()


def get_performance_monitor() -> EnhancedPerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _performance_monitor
    
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = EnhancedPerformanceMonitor()
    
    return _performance_monitor


# Convenience functions with lazy initialization
def start_performance_monitoring() -> None:
    """Start global performance monitoring"""
    get_performance_monitor().start()


def stop_performance_monitoring() -> None:
    """Stop global performance monitoring"""
    if _performance_monitor is not None:
        _performance_monitor.stop()


def get_performance_status() -> Dict[str, Any]:
    """Get current performance status"""
    return get_performance_monitor().get_current_status()


def register_component_performance(component: str, **metrics: Union[int, float]) -> None:
    """Register component performance metrics"""
    get_performance_monitor().register_component_metrics(component, **metrics)


def save_performance_report(filepath: Optional[Union[str, Path]] = None) -> str:
    """Save performance report to file"""
    return get_performance_monitor().save_report(filepath)


# Enhanced context manager for performance monitoring
class PerformanceMonitorContext:
    """Context manager for temporary performance monitoring"""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 auto_save_report: bool = False,
                 report_path: Optional[Union[str, Path]] = None) -> None:
        self.collection_interval = collection_interval
        self.auto_save_report = auto_save_report
        self.report_path = report_path
        self.monitor: Optional[EnhancedPerformanceMonitor] = None
    
    def __enter__(self) -> EnhancedPerformanceMonitor:
        """Start monitoring context"""
        self.monitor = EnhancedPerformanceMonitor(collection_interval=self.collection_interval)
        self.monitor.start()
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring context and optionally save report"""
        if self.monitor:
            self.monitor.stop()
            
            if self.auto_save_report:
                try:
                    self.monitor.save_report(self.report_path)
                except Exception as e:
                    logw(f"Failed to auto-save performance report: {e}")


# Export key classes and functions
__all__ = [
    'EnhancedPerformanceMonitor',
    'PerformanceMetrics',
    'Alert',
    'AlertLevel',
    'PerformanceMonitorContext',
    'start_performance_monitoring',
    'stop_performance_monitoring',
    'get_performance_status',
    'register_component_performance',
    'save_performance_report',
    'PSUTIL_AVAILABLE'
]