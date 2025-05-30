#!/usr/bin/env python3
"""
Enhanced News Catalyst Trading System - Production-Ready Main Application
Optimized for performance, reliability, and maintainability
"""

import asyncio
import signal
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import threading
import time

# Setup path for imports
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Import optimized components
from config import CONFIG, OptimizedConfig, Environment
from utils.log_utils import loge, logi, logw, logd

# Conditional imports with graceful fallbacks
try:
    from utils.performance_monitor import EnhancedPerformanceMonitor, start_performance_monitoring, stop_performance_monitoring
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError as e:
    PERFORMANCE_MONITOR_AVAILABLE = False
    logw(f"Performance monitor not fully available: {e}")

try:
    from utils.notification_utils import stop_notification_system
    NOTIFICATION_UTILS_AVAILABLE = True
except ImportError as e:
    NOTIFICATION_UTILS_AVAILABLE = False
    logw(f"Notification utilities not available: {e}")

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TORCH_AVAILABLE = True
    logi("✅ PyTorch and Transformers available")
except ImportError:
    TORCH_AVAILABLE = False
    logw("⚠️  PyTorch/Transformers not available - using basic sentiment analysis")
    
    # Mock classes for when torch is not available
    class MockTorch:
        @staticmethod
        def cuda():
            class CUDA:
                @staticmethod
                def is_available() -> bool:
                    return False
            return CUDA()
    
    class MockModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> None:
            return None
    
    torch = MockTorch()  # type: ignore
    AutoModelForSequenceClassification = MockModel  # type: ignore
    AutoTokenizer = MockModel  # type: ignore


@dataclass
class ComponentStatus:
    """Component status tracking"""
    name: str
    is_running: bool = False
    last_error: Optional[str] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0


class ComponentManager:
    """Manages system components with health monitoring and auto-restart"""
    
    def __init__(self):
        self.components: Dict[str, ComponentStatus] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.max_restart_attempts = 3
        self.restart_delay = 30  # seconds
    
    def register_component(self, name: str, instance: Any) -> None:
        """Register a component for monitoring"""
        with self._lock:
            self.components[name] = ComponentStatus(name=name)
            self.instances[name] = instance
            logd(f"Registered component: {name}")
    
    def mark_running(self, name: str) -> None:
        """Mark component as running"""
        with self._lock:
            if name in self.components:
                self.components[name].is_running = True
                self.components[name].start_time = datetime.now()
                self.components[name].last_error = None
    
    def mark_failed(self, name: str, error: str) -> None:
        """Mark component as failed"""
        with self._lock:
            if name in self.components:
                self.components[name].is_running = False
                self.components[name].last_error = error
                loge(f"Component {name} failed: {error}")
    
    async def restart_component(self, name: str) -> bool:
        """Attempt to restart a failed component"""
        with self._lock:
            if name not in self.components:
                return False
            
            status = self.components[name]
            if status.restart_count >= self.max_restart_attempts:
                loge(f"Component {name} exceeded max restart attempts")
                return False
            
            status.restart_count += 1
            logi(f"Attempting to restart {name} (attempt {status.restart_count})")
        
        try:
            # Cancel existing task if any
            if name in self.tasks:
                self.tasks[name].cancel()
                try:
                    await self.tasks[name]
                except asyncio.CancelledError:
                    pass
            
            # Wait before restart
            await asyncio.sleep(self.restart_delay)
            
            # Restart component
            instance = self.instances[name]
            if hasattr(instance, 'start'):
                if asyncio.iscoroutinefunction(instance.start):
                    task = asyncio.create_task(instance.start())
                    self.tasks[name] = task
                else:
                    instance.start()
            
            self.mark_running(name)
            logi(f"Successfully restarted {name}")
            return True
            
        except Exception as e:
            self.mark_failed(name, str(e))
            return False
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all component statuses"""
        with self._lock:
            return {
                name: {
                    'is_running': status.is_running,
                    'last_error': status.last_error,
                    'restart_count': status.restart_count,
                    'uptime': (
                        (datetime.now() - status.start_time).total_seconds()
                        if status.start_time else 0
                    )
                }
                for name, status in self.components.items()
            }


class EnhancedTradingSystem:
    """Production-ready trading system with comprehensive monitoring"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.component_manager = ComponentManager()
        
        if PERFORMANCE_MONITOR_AVAILABLE:
            try:
                self.performance_monitor = EnhancedPerformanceMonitor()
            except Exception as e:
                logw(f"Failed to initialize performance monitor: {e}")
                self.performance_monitor = None
        else:
            self.performance_monitor = None
        
        # System state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.startup_time = datetime.now()
        
        # Components will be initialized in setup
        self.components: Dict[str, Any] = {}
        self.notification_client: Optional[Any] = None
        self.notification_type = "none"
        
        # Monitoring
        self.health_check_interval = 60  # seconds
        self.last_health_check = datetime.now()
        
    async def initialize_notification_system(self) -> None:
        """Initialize optimized notification system"""
        try:
            if self.config.notifications.use_gmail:
                logi("📧 Initializing Gmail notification system...")
                
                from notification_tools.gmail_client import EnterpriseGmailClient
                
                gmail_email = self.config.get_api_key('gmail_email') or CONFIG.get_api_key('gmail_email')
                gmail_password = self.config.get_api_key('gmail_app_password') or CONFIG.get_api_key('gmail_app_password')
                
                if not gmail_email or not gmail_password:
                    logw("Gmail credentials not configured, disabling notifications")
                    return
                
                self.notification_client = EnterpriseGmailClient(
                    gmail_email=gmail_email,
                    gmail_app_password=gmail_password,
                    max_rate_per_minute=self.config.notifications.max_emails_per_minute
                )
                
                self.notification_client.start()
                self.notification_type = "gmail"
                logi(f"✅ Gmail notifications initialized for {gmail_email}")
                
        except Exception as e:
            loge(f"Failed to initialize notification system: {e}")
            self.notification_client = None
    
    async def initialize_ai_models(self) -> tuple[Any, Any, str]:
        """Initialize AI models with error handling"""
        model = None
        tokenizer = None
        device = "cpu"
        
        if TORCH_AVAILABLE and self.config.ai.finbert_weight > 0:
            try:
                logi("🤖 Loading FinBERT model...")
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                logi(f"Using device: {device}")
                
                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
                
                logi("✅ FinBERT model loaded successfully")
                
            except Exception as e:
                logw(f"Failed to load FinBERT model: {e}")
                model = None
                tokenizer = None
        else:
            logw("FinBERT disabled or PyTorch unavailable")
        
        return model, tokenizer, device
    
    async def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            logi("🚀 Initializing Enhanced Trading System...")
            
            # Validate configuration
            config_errors = self.config.validate_config()
            if config_errors:
                for error in config_errors:
                    loge(f"Configuration error: {error}")
                return False
            
            # Get API keys
            fmp_api_key = self.config.get_api_key('fmp')
            tiingo_api_key = self.config.get_api_key('tiingo')
            
            if not fmp_api_key:
                loge("FMP API key required but not configured")
                return False
            
            # Initialize notification system
            await self.initialize_notification_system()
            
            # Initialize AI models
            model, tokenizer, device = await self.initialize_ai_models()
            
            # Initialize core components
            logi("💰 Initializing core trading components...")
            
            # Import optimized components
            from universe_selection.universe_selector import UniverseSelector
            from price_trackers.price_tracker import OptimizedPriceTracker
            from event_processors.news_processor_manager import NewsProcessorManager
            from event_trackers.news_event_tracker import NewsEventTracker
            from event_trackers.press_release_tracker import PressReleaseTracker
            from momentum_trackers.momentum_tracker_manager import MomentumTrackerManager
            from trade_management.trade_entry_manager import TradeEntryManager
            from trade_management.trade_exit_manager import TradeExitManager
            from trade_management.trade_log import TradeLog
            
            # Universe selection
            universe_selector = UniverseSelector(fmp_api_key)
            universe_selector.perform_selection()
            self.components['universe_selector'] = universe_selector
            self.component_manager.register_component('universe_selector', universe_selector)
            
            # Price tracking
            price_tracker = OptimizedPriceTracker(
                fmp_api_key=fmp_api_key,
                update_interval=self.config.data.price_tick_interval
            )
            self.components['price_tracker'] = price_tracker
            self.component_manager.register_component('price_tracker', price_tracker)
            
            # News processing
            news_processor_manager = NewsProcessorManager(
                num_workers=max(1, self.config.data.api_rate_limit_calls // 20),  # Dynamic worker count
                model=model,
                tokenizer=tokenizer,
                device=device,
                notification_client=self.notification_client,
                notification_type=self.notification_type
            )
            self.components['news_processor_manager'] = news_processor_manager
            self.component_manager.register_component('news_processor_manager', news_processor_manager)
            
            # Event tracking
            news_event_tracker = NewsEventTracker(
                fmp_api_key=fmp_api_key,
                tiingo_api_key=tiingo_api_key or "",
                data_collection_interval=int(self.config.data.news_check_interval)
            )
            self.components['news_event_tracker'] = news_event_tracker
            self.component_manager.register_component('news_event_tracker', news_event_tracker)
            
            press_release_tracker = PressReleaseTracker(
                fmp_api_key=fmp_api_key,
                data_collection_interval=int(self.config.data.news_check_interval)
            )
            self.components['press_release_tracker'] = press_release_tracker
            self.component_manager.register_component('press_release_tracker', press_release_tracker)
            
            # Momentum tracking
            momentum_tracker_manager = MomentumTrackerManager(
                num_workers=max(1, int(self.config.trading.max_position_size * 20)),  # Scale with position limits
                data_collection_interval=int(self.config.data.momentum_calc_interval),
                notification_client=self.notification_client,
                notification_type=self.notification_type
            )
            self.components['momentum_tracker_manager'] = momentum_tracker_manager
            self.component_manager.register_component('momentum_tracker_manager', momentum_tracker_manager)
            
            # Trade management
            trade_entry_manager = TradeEntryManager(num_workers=3)
            trade_exit_manager = TradeExitManager(num_workers=3)
            trade_log = TradeLog()
            
            self.components.update({
                'trade_entry_manager': trade_entry_manager,
                'trade_exit_manager': trade_exit_manager,
                'trade_log': trade_log
            })
            
            for name, component in [
                ('trade_entry_manager', trade_entry_manager),
                ('trade_exit_manager', trade_exit_manager),
                ('trade_log', trade_log)
            ]:
                self.component_manager.register_component(name, component)
            
            logi("✅ All components initialized successfully")
            
            # Send startup notification
            if self.notification_client:
                await self.send_startup_notification()
            
            return True
            
        except Exception as e:
            loge(f"Failed to initialize components: {e}")
            loge(traceback.format_exc())
            return False
    
    async def send_startup_notification(self) -> None:
        """Send enhanced startup notification"""
        try:
            if not self.notification_client:
                return
            
            universe_selector = self.components.get('universe_selector')
            symbol_count = 0
            if universe_selector:
                symbol_list = universe_selector.get_symbol_list()
                symbol_count = len(symbol_list) if symbol_list else 0
            
            # Get basic system info
            try:
                import threading
                thread_count = threading.active_count()
            except Exception:
                thread_count = 0
            
            subject = "🚀 Enhanced Trading System Started"
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; margin: 20px;">
                <h2>🚀 Enhanced News Catalyst Trading System Online</h2>
                
                <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <h3>📊 System Configuration:</h3>
                    <ul>
                        <li><strong>Environment:</strong> {self.config.env.value}</li>
                        <li><strong>Active Threads:</strong> {thread_count}</li>
                        <li><strong>PyTorch Available:</strong> {'✅' if TORCH_AVAILABLE else '❌'}</li>
                        <li><strong>Performance Monitor:</strong> {'✅' if PERFORMANCE_MONITOR_AVAILABLE else '❌'}</li>
                    </ul>
                </div>
                
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <h3>📈 Trading Configuration:</h3>
                    <ul>
                        <li><strong>Universe Size:</strong> {symbol_count} stocks</li>
                        <li><strong>Max Position Size:</strong> {self.config.trading.max_position_size*100:.1f}%</li>
                        <li><strong>Price Update Interval:</strong> {self.config.data.price_tick_interval}s</li>
                        <li><strong>News Check Interval:</strong> {self.config.data.news_check_interval}s</li>
                        <li><strong>Notifications:</strong> {self.notification_type.upper()}</li>
                    </ul>
                </div>
                
                <div style="background-color: #fff8e1; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <h3>🤖 AI Configuration:</h3>
                    <ul>
                        <li><strong>FinBERT Weight:</strong> {self.config.ai.finbert_weight}</li>
                        <li><strong>Ensemble Mode:</strong> {'✅' if self.config.ai.use_ensemble else '❌'}</li>
                        <li><strong>Confidence Threshold:</strong> {self.config.ai.min_confidence_threshold}</li>
                    </ul>
                </div>
                
                <p><strong>Started:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Status:</strong> All systems operational ✅</p>
                
                <hr>
                <p style="font-size: 12px; color: #666;">
                    🤖 Enhanced News Catalyst Trading System v2.0
                </p>
            </body>
            </html>
            """
            
            # Send to first configured email address
            email_addresses = CONFIG.get_api_key('alert_to_email') or CONFIG.get_api_key('gmail_email')
            if email_addresses and hasattr(self.notification_client, 'send_email_async'):
                success = self.notification_client.send_email_async(
                    to_email=email_addresses.split(',')[0] if ',' in email_addresses else email_addresses,
                    subject=subject,
                    body=body,
                    priority=1
                )
                
                if success:
                    logi("📧 Startup notification sent")
                else:
                    logw("📧 Startup notification failed")
                    
        except Exception as e:
            logw(f"Startup notification error: {e}")
    
    async def start_services(self) -> None:
        """Start all services with health monitoring"""
        try:
            logi("🚀 Starting optimized services...")
            
            # Start synchronous services
            sync_services = ['momentum_tracker_manager', 'news_processor_manager']
            for service_name in sync_services:
                service = self.components.get(service_name)
                if service and hasattr(service, 'start'):
                    logi(f"▶️  Starting {service_name}")
                    service.start()
                    self.component_manager.mark_running(service_name)
            
            # Start asynchronous services
            async_services = [
                'price_tracker', 'news_event_tracker', 'press_release_tracker',
                'trade_log', 'trade_entry_manager', 'trade_exit_manager'
            ]
            
            for service_name in async_services:
                service = self.components.get(service_name)
                if service and hasattr(service, 'start'):
                    logi(f"▶️  Starting {service_name}")
                    
                    # Create monitored task
                    async def create_monitored_task(svc: Any, name: str) -> None:
                        try:
                            self.component_manager.mark_running(name)
                            await svc.start()
                        except asyncio.CancelledError:
                            logi(f"🛑 {name} cancelled during shutdown")
                            raise
                        except Exception as e:
                            self.component_manager.mark_failed(name, str(e))
                            loge(f"❌ {name} failed: {e}")
                            
                            # Attempt restart
                            if not self.shutdown_event.is_set():
                                logw(f"Attempting to restart {name}...")
                                await self.component_manager.restart_component(name)
                    
                    task = asyncio.create_task(
                        create_monitored_task(service, service_name),
                        name=service_name
                    )
                    self.component_manager.tasks[service_name] = task
            
            logi("✅ All services started successfully")
            
        except Exception as e:
            loge(f"Failed to start services: {e}")
            raise
    
    async def health_monitor(self) -> None:
        """Monitor system health and component status"""
        logi("🔍 Starting health monitor...")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Check component health
                    component_status = self.component_manager.get_status_summary()
                    failed_components = [
                        name for name, status in component_status.items()
                        if not status['is_running'] and status['restart_count'] < 3
                    ]
                    
                    if failed_components:
                        logw(f"Failed components detected: {failed_components}")
                        
                        # Attempt to restart failed components
                        for component_name in failed_components:
                            await self.component_manager.restart_component(component_name)
                    
                    # Log health status periodically
                    if (datetime.now() - self.last_health_check).total_seconds() > 300:  # Every 5 minutes
                        running_count = sum(1 for s in component_status.values() if s['is_running'])
                        total_count = len(component_status)
                        
                        logi(f"💓 Health check: {running_count}/{total_count} components running")
                        
                        # Register performance metrics if available
                        if PERFORMANCE_MONITOR_AVAILABLE and self.performance_monitor:
                            try:
                                from utils.performance_monitor import register_component_performance
                                register_component_performance(
                                    'system_health',
                                    running_components=running_count,
                                    total_components=total_count,
                                    uptime_hours=(datetime.now() - self.startup_time).total_seconds() / 3600
                                )
                            except Exception as e:
                                logd(f"Performance monitoring error: {e}")
                        
                        self.last_health_check = datetime.now()
                    
                    # Wait for next check or shutdown
                    try:
                        await asyncio.wait_for(
                            self.shutdown_event.wait(), 
                            timeout=self.health_check_interval
                        )
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Normal timeout, continue monitoring
                        
                except Exception as e:
                    loge(f"Error in health monitor: {e}")
                    await asyncio.sleep(30)  # Error recovery delay
                    
        except asyncio.CancelledError:
            logi("🔍 Health monitor cancelled")
            raise
        finally:
            logi("🔍 Health monitor stopped")
    
    async def shutdown_services(self) -> None:
        """Enhanced graceful shutdown with timeout handling"""
        try:
            logi("🛑 Initiating enhanced graceful shutdown...")
            
            # Stop notification system first
            try:
                if NOTIFICATION_UTILS_AVAILABLE:
                    stop_notification_system()
                if self.notification_client and hasattr(self.notification_client, 'stop'):
                    self.notification_client.stop(timeout=5)
                logi("✅ Notification system stopped")
            except Exception as e:
                logw(f"Error stopping notification system: {e}")
            
            # Cancel all async tasks
            tasks_to_cancel = list(self.component_manager.tasks.values())
            if tasks_to_cancel:
                logi(f"🔄 Cancelling {len(tasks_to_cancel)} async tasks...")
                
                for task in tasks_to_cancel:
                    if not task.done():
                        task.cancel()
                
                # Wait for tasks to complete with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=10.0
                    )
                    logi("✅ All async tasks cancelled successfully")
                except asyncio.TimeoutError:
                    logw("⏰ Some tasks did not stop within timeout")
            
            # Stop synchronous services
            sync_services = ['momentum_tracker_manager', 'news_processor_manager']
            for service_name in sync_services:
                service = self.components.get(service_name)
                if service and hasattr(service, 'stop'):
                    logi(f"⏹️  Stopping {service_name}")
                    try:
                        service.stop()
                    except Exception as e:
                        logw(f"Error stopping {service_name}: {e}")
            
            # Stop remaining services
            for service_name, service in self.components.items():
                if service_name not in sync_services and hasattr(service, 'stop'):
                    try:
                        service.stop()
                    except Exception as e:
                        logw(f"Error stopping {service_name}: {e}")
            
            logi("✅ Enhanced graceful shutdown completed")
            
        except Exception as e:
            loge(f"Error during shutdown: {e}")
    
    async def run(self) -> bool:
        """Main application run loop"""
        try:
            logi("🎯 Enhanced News Catalyst Trading System Starting...")
            logi(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logi(f"🐍 Python: {sys.version}")
            logi(f"🚀 PyTorch Available: {TORCH_AVAILABLE}")
            if TORCH_AVAILABLE:
                logi(f"🚀 CUDA Available: {torch.cuda.is_available()}")
            logi(f"⚙️  Environment: {self.config.env.value}")
            
            # Initialize components
            if not await self.initialize_components():
                loge("❌ Component initialization failed")
                return False
            
            # Start performance monitoring
            if PERFORMANCE_MONITOR_AVAILABLE:
                try:
                    start_performance_monitoring()
                except Exception as e:
                    logw(f"Performance monitoring failed to start: {e}")
            
            # Start services
            await self.start_services()
            
            logi("🎉 Enhanced system fully operational!")
            logi(f"📧 Notifications: {self.notification_type.upper()}")
            logi("🛑 Press Ctrl+C to shutdown gracefully")
            
            # Start health monitoring
            health_task = asyncio.create_task(self.health_monitor(), name="HealthMonitor")
            
            try:
                # Wait for shutdown signal
                await self.shutdown_event.wait()
                logi("🛑 Shutdown signal received")
            finally:
                # Cancel health monitoring
                if not health_task.done():
                    health_task.cancel()
                    try:
                        await health_task
                    except asyncio.CancelledError:
                        pass
            
            return True
            
        except Exception as e:
            loge(f"❌ Fatal error: {e}")
            loge(traceback.format_exc())
            return False
        
        finally:
            await self.shutdown_services()
            if PERFORMANCE_MONITOR_AVAILABLE:
                try:
                    stop_performance_monitoring()
                except Exception as e:
                    logw(f"Error stopping performance monitoring: {e}")


# Global shutdown handling
_system_instance: Optional[EnhancedTradingSystem] = None
_shutdown_requested = False


def handle_shutdown_signal(signum: int, frame: Any) -> None:
    """Enhanced shutdown signal handler"""
    global _shutdown_requested, _system_instance
    
    if not _shutdown_requested:
        _shutdown_requested = True
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        
        if _system_instance:
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(_system_instance.shutdown_event.set)
            except RuntimeError:
                if hasattr(_system_instance, 'shutdown_event'):
                    try:
                        _system_instance.shutdown_event.set()
                    except Exception:
                        pass
    else:
        print(f"\n🔥 Second signal {signum} received, forcing exit...")
        sys.exit(1)


async def main() -> None:
    """Enhanced main entry point"""
    global _system_instance
    
    exit_code = 0
    
    try:
        # Detect environment
        env = Environment.PRODUCTION
        if '--dev' in sys.argv:
            env = Environment.DEVELOPMENT
        elif '--test' in sys.argv:
            env = Environment.TESTING
        
        # Setup signal handlers
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, handle_shutdown_signal)
            signal.signal(signal.SIGTERM, handle_shutdown_signal)
        else:
            signal.signal(signal.SIGINT, handle_shutdown_signal)
        
        # Create and run system
        config = OptimizedConfig(env)
        _system_instance = EnhancedTradingSystem(config)
        
        success = await _system_instance.run()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("🔥 Keyboard interrupt received")
        exit_code = 0
        
    except Exception as e:
        print(f"💥 Unhandled exception: {e}")
        logging.error(traceback.format_exc())
        exit_code = 1
        
    finally:
        print("👋 Enhanced system shutdown complete")
        await asyncio.sleep(0.5)  # Allow logging to flush


if __name__ == "__main__":
    try:
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            sys.exit(1)
        
        print("=" * 70)
        print("🚀 ENHANCED NEWS CATALYST TRADING SYSTEM")
        print("⚡ Production-Ready with Advanced Monitoring")
        print("=" * 70)
        
        # Handle Windows event loop policy
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run the enhanced application
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🔥 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal startup error: {e}")
        print(traceback.format_exc())
        sys.exit(1)