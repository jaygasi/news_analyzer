"""
Enhanced performance monitoring with graceful fallback when psutil is unavailable
"""
import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import json
import os
from enum import Enum

from utils.log_utils import logi, logw, loge

# Graceful import of psutil with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logw("psutil not available - using fallback performance monitoring")


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics with fallback values"""
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


@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    value: float
    threshold: float


class AdaptiveThresholds:
    """Dynamic performance thresholds that adapt to system behavior"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.thresholds: Dict[str, float] = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'error_rate': 0.05,
            'queue_size': 1000
        }
    
    def update_baseline(self, metric_name: str, value: float) -> None:
        """Update baseline for adaptive thresholds"""
        self.baseline_metrics[metric_name].append(value)
    
    def get_adaptive_threshold(self, metric_name: str) -> float:
        """Calculate adaptive threshold based on historical data"""
        if metric_name not in self.baseline_metrics or len(self.baseline_metrics[metric_name]) < 10:
            return self.thresholds.get(metric_name, 100.0)
        
        baseline_data = list(self.baseline_metrics[metric_name])
        avg = sum(baseline_data) / len(baseline_data)
        std_dev = (sum((x - avg) ** 2 for x in baseline_data) / len(baseline_data)) ** 0.5
        
        # Adaptive threshold: average + 2 standard deviations
        adaptive_threshold = avg + (2 * std_dev)
        base_threshold = self.thresholds.get(metric_name, 100.0)
        
        # Don't let adaptive threshold be too permissive
        return min(adaptive_threshold, base_threshold * 1.5)


class FallbackPerformanceCollector:
    """Fallback performance collector when psutil is not available"""
    
    def __init__(self):
        self._start_time = time.time()
        import threading
        self._thread_count = threading.active_count()
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect basic metrics without psutil"""
        try:
            import threading
            current_threads = threading.active_count()
            
            # Basic uptime calculation
            uptime_hours = (time.time() - self._start_time) / 3600
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,  # Cannot measure without psutil
                memory_percent=0.0,  # Cannot measure without psutil
                memory_used_mb=0.0,
                disk_usage_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                active_threads=current_threads,
                open_files=0
            )
        except Exception as e:
            loge(f"Error in fallback performance collection: {e}")
            return PerformanceMetrics(timestamp=datetime.now())


class PerformanceCollector:
    """Optimized performance data collection with psutil"""
    
    def __init__(self):
        if not PSUTIL_AVAILABLE:
            self._fallback = FallbackPerformanceCollector()
            return
            
        self._network_baseline: Optional[Any] = None
        try:
            self._process = psutil.Process()
        except Exception:
            self._process = None
        self._last_collection = datetime.now()
    
    def collect_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect comprehensive performance metrics efficiently"""
        if not PSUTIL_AVAILABLE:
            return self._fallback.collect_metrics()
            
        try:
            now = datetime.now()
            
            # Batch system calls for efficiency
            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
            memory = psutil.virtual_memory()
            
            try:
                disk_usage = psutil.disk_usage('/')
            except (OSError, PermissionError):
                # Fallback for Windows or restricted environments
                disk_usage = type('obj', (object,), {'percent': 0.0})()
            
            # Network stats with baseline
            try:
                network = psutil.net_io_counters()
                if self._network_baseline is None:
                    self._network_baseline = network
                
                network_sent_mb = max(0, (network.bytes_sent - self._network_baseline.bytes_sent) / 1024 / 1024)
                network_recv_mb = max(0, (network.bytes_recv - self._network_baseline.bytes_recv) / 1024 / 1024)
            except (AttributeError, OSError):
                network_sent_mb = 0.0
                network_recv_mb = 0.0
            
            # Process-specific metrics
            try:
                active_threads = self._process.num_threads() if self._process else 0
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                active_threads = 0
                
            try:
                open_files = len(self._process.open_files()) if self._process else 0
            except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
                open_files = 0
            
            self._last_collection = now
            
            return PerformanceMetrics(
                timestamp=now,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk_usage.percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_threads=active_threads,
                open_files=open_files
            )
            
        except Exception as e:
            loge(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(timestamp=datetime.now())


class EnhancedPerformanceMonitor:
    """Production-ready performance monitor with graceful degradation"""
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 history_size: int = 1000,
                 alert_callback: Optional[Callable[[Alert], None]] = None):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.alert_callback = alert_callback
        
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.component_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts: deque = deque(maxlen=100)
        
        # Monitoring components
        self.collector = PerformanceCollector()
        self.thresholds = AdaptiveThresholds()
        
        # Thread safety
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        
        if not PSUTIL_AVAILABLE:
            logw("Running in limited performance monitoring mode (psutil unavailable)")
    
    def register_component_metrics(self, component: str, **metrics) -> None:
        """Thread-safe component metric registration"""
        with self._lock:
            timestamp = datetime.now()
            
            # Store individual metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    key = f"{component}_{metric_name}"
                    self.component_metrics[key].append({
                        'timestamp': timestamp,
                        'value': float(value)
                    })
                    
                    # Update adaptive thresholds
                    self.thresholds.update_baseline(key, float(value))
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for performance alerts with adaptive thresholds"""
        if not PSUTIL_AVAILABLE:
            return  # Skip alert checking without psutil
            
        alerts = []
        
        # System-level alerts
        system_checks = [
            ('cpu_percent', metrics.cpu_percent, "High CPU usage"),
            ('memory_percent', metrics.memory_percent, "High memory usage"),
            ('disk_usage_percent', metrics.disk_usage_percent, "High disk usage")
        ]
        
        for metric_name, value, message in system_checks:
            if value > 0:  # Only check if we have valid data
                threshold = self.thresholds.get_adaptive_threshold(metric_name)
                if value > threshold:
                    alert = Alert(
                        timestamp=datetime.now(),
                        level=AlertLevel.WARNING if value < threshold * 1.2 else AlertLevel.CRITICAL,
                        component="system",
                        message=f"{message}: {value:.1f}%",
                        value=value,
                        threshold=threshold
                    )
                    alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            with self._lock:
                self.alerts.append(alert)
            
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
            recent_alerts = list(self.alerts)[-5:]  # Last 5 alerts
            
            # Calculate performance trends
            if len(self.metrics_history) >= 2:
                prev_metrics = self.metrics_history[-2]
                cpu_trend = latest_metrics.cpu_percent - prev_metrics.cpu_percent
                memory_trend = latest_metrics.memory_percent - prev_metrics.memory_percent
            else:
                cpu_trend = 0
                memory_trend = 0
            
            return {
                "timestamp": latest_metrics.timestamp.isoformat(),
                "psutil_available": PSUTIL_AVAILABLE,
                "monitoring_mode": "full" if PSUTIL_AVAILABLE else "limited",
                "system": {
                    "cpu_percent": latest_metrics.cpu_percent,
                    "cpu_trend": cpu_trend,
                    "memory_percent": latest_metrics.memory_percent,
                    "memory_trend": memory_trend,
                    "memory_used_mb": latest_metrics.memory_used_mb,
                    "active_threads": latest_metrics.active_threads,
                    "open_files": latest_metrics.open_files
                },
                "components": {
                    name: list(values)[-1] if values else None 
                    for name, values in self.component_metrics.items()
                },
                "recent_alerts": [
                    {
                        "timestamp": alert.timestamp.isoformat(),
                        "level": alert.level.value,
                        "component": alert.component,
                        "message": alert.message
                    }
                    for alert in recent_alerts
                ],
                "status": "healthy" if not any(a.level == AlertLevel.CRITICAL for a in recent_alerts) else "degraded"
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            if not self.metrics_history:
                return {
                    "error": "No metrics available",
                    "psutil_available": PSUTIL_AVAILABLE
                }
            
            recent_window = min(60, len(self.metrics_history))  # Last 60 samples or all available
            recent_metrics = list(self.metrics_history)[-recent_window:]
            
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics if m.cpu_percent > 0]
            memory_values = [m.memory_percent for m in recent_metrics if m.memory_percent > 0]
            
            system_summary = {
                'psutil_available': PSUTIL_AVAILABLE,
                'monitoring_mode': "full" if PSUTIL_AVAILABLE else "limited",
                'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max_cpu_percent': max(cpu_values) if cpu_values else 0,
                'avg_memory_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_memory_percent': max(memory_values) if memory_values else 0,
                'current_threads': recent_metrics[-1].active_threads,
                'current_open_files': recent_metrics[-1].open_files,
                'samples_analyzed': len(recent_metrics)
            }
            
            # Component performance summary
            component_summary = {}
            for component_name, values in self.component_metrics.items():
                if values and len(values) >= 2:
                    recent_values = list(values)[-min(10, len(values)):]
                    numeric_values = [
                        v['value'] if isinstance(v, dict) else v 
                        for v in recent_values 
                        if (isinstance(v, dict) and 'value' in v) or isinstance(v, (int, float))
                    ]
                    
                    if numeric_values:
                        component_summary[component_name] = {
                            'current': numeric_values[-1],
                            'average': sum(numeric_values) / len(numeric_values),
                            'samples': len(numeric_values)
                        }
            
            # Recent alerts summary
            alert_summary = {
                'total_alerts': len(self.alerts),
                'critical_alerts': sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL),
                'warning_alerts': sum(1 for a in self.alerts if a.level == AlertLevel.WARNING),
                'recent_alerts': [
                    {
                        'timestamp': a.timestamp.isoformat(),
                        'level': a.level.value,
                        'message': a.message
                    }
                    for a in list(self.alerts)[-5:]
                ]
            }
            
            return {
                'report_timestamp': datetime.now().isoformat(),
                'system_summary': system_summary,
                'component_summary': component_summary,
                'alert_summary': alert_summary,
                'adaptive_thresholds': dict(self.thresholds.thresholds)
            }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        mode = "full" if PSUTIL_AVAILABLE else "limited"
        logi(f"🔍 Performance monitor started ({mode} mode)")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = self.collector.collect_metrics()
                if metrics:
                    with self._lock:
                        self.metrics_history.append(metrics)
                    
                    # Update adaptive thresholds (only if we have real data)
                    if PSUTIL_AVAILABLE:
                        self.thresholds.update_baseline('cpu_percent', metrics.cpu_percent)
                        self.thresholds.update_baseline('memory_percent', metrics.memory_percent)
                        
                        # Check for alerts
                        self._check_alerts(metrics)
                
                # Wait for next collection
                if self._shutdown_event.wait(self.collection_interval):
                    break
                    
            except Exception as e:
                loge(f"Error in performance monitor loop: {e}")
                if self._shutdown_event.wait(5.0):  # Error recovery delay
                    break
        
        logi("🔍 Performance monitor stopped")
    
    def start(self) -> None:
        """Start performance monitoring"""
        if self.is_running:
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
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop performance monitoring gracefully"""
        if not self.is_running:
            return
        
        logi("🛑 Stopping performance monitor...")
        self.is_running = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
            if self._monitor_thread.is_alive():
                logw("Performance monitor thread did not stop gracefully")
        
        logi("✅ Performance monitor stopped")
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save performance report to file"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"performance_report_{timestamp}.json"
        
        try:
            report = self.get_performance_report()
            os.makedirs(os.path.dirname(filepath) or 'reports', exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logi(f"📊 Performance report saved to {filepath}")
            return filepath
            
        except Exception as e:
            loge(f"Error saving performance report: {e}")
            raise


# Global performance monitor instance
performance_monitor = EnhancedPerformanceMonitor()

# Convenience functions
def start_performance_monitoring() -> None:
    """Start global performance monitoring"""
    performance_monitor.start()

def stop_performance_monitoring() -> None:
    """Stop global performance monitoring"""
    performance_monitor.stop()

def get_performance_status() -> Dict[str, Any]:
    """Get current performance status"""
    return performance_monitor.get_current_status()

def register_component_performance(component: str, **metrics) -> None:
    """Register component performance metrics"""
    performance_monitor.register_component_metrics(component, **metrics)