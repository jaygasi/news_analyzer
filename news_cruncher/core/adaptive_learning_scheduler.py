"""
Adaptive Learning Scheduler - Automatically trigger training when conditions are met
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Optional # Added import for Optional
from config import Config
from utils.simple_logger import log_info, log_debug, log_error # Added log_error
from tools.multi_modal_learning_system import MultiModalLearningSystem

class AdaptiveLearningScheduler:
    """Background scheduler for adaptive learning"""
    
    def __init__(self):
        self.learning_system = MultiModalLearningSystem()
        self.running = False
        self.thread = None
        
        # Get configuration
        self.check_interval_hours = getattr(Config, 'ADAPTIVE_LEARNING_CHECK_HOURS', 6)
        self.check_interval_seconds = self.check_interval_hours * 3600  # Convert to seconds
        self.enabled = getattr(Config, 'ENABLE_ADAPTIVE_LEARNING', False)
    
    def start(self):
        """Start the background scheduler"""
        if not self.enabled:
            log_info("📚 Adaptive learning disabled in configuration. Scheduler not starting.")
            return
        
        if self.running:
            log_debug("Adaptive learning scheduler already running.")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.name = "AdaptiveLearningSchedulerThread" # Give thread a name
        self.thread.start()
        log_info(f"🎓 Adaptive learning scheduler started. Will check every {self.check_interval_hours} hours.")
    
    def stop(self):
        """Stop the background scheduler"""
        if not self.running:
            log_debug("Adaptive learning scheduler is not running.")
            return
            
        self.running = False
        if self.thread and self.thread.is_alive():
            log_info("📚 Attempting to stop adaptive learning scheduler...")
            self.thread.join(timeout=10) # Wait for thread to finish
            if self.thread.is_alive():
                log_warning("Adaptive learning scheduler thread did not stop in time.")
            else:
                log_info("📚 Adaptive learning scheduler stopped successfully.")
        self.thread = None # Clear the thread object
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        log_info("Adaptive learning scheduler loop initiated.")
        while self.running:
            try:
                log_debug(f"Scheduler check: Checking training trigger conditions at {datetime.now()}")
                # Check if training should be triggered
                if self.learning_system.check_training_trigger():
                    log_info("🚀 Triggering adaptive learning cycle from scheduler...")
                    success = self.learning_system.run_adaptive_learning()
                    
                    if success:
                        log_info("✅ Adaptive learning cycle completed successfully via scheduler!")
                    else:
                        log_warning("⚠️ Adaptive learning cycle via scheduler encountered issues or did not run.") # Changed to warning
                else:
                    log_debug("Training trigger conditions not met in current scheduler check.")
                
                # Sleep for configured interval
                for _ in range(self.check_interval_seconds): # Sleep in 1-second intervals to allow faster shutdown
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                log_error(f"Error in adaptive learning scheduler loop: {e}", exc_info=True)
                # Prevent rapid error looping if persistent issue
                log_info("Scheduler loop error. Sleeping for 5 minutes before retrying.")
                for _ in range(300): 
                    if not self.running:
                        break
                    time.sleep(1)
        log_info("Adaptive learning scheduler loop terminated.")
    
    def force_training(self) -> bool:
        """Force training immediately (for testing or manual trigger)"""
        if not self.enabled:
            log_warning("Adaptive learning is disabled in config. Forced training will not run.")
            return False
        log_info("🔧 Forcing adaptive learning training cycle...")
        return self.learning_system.run_adaptive_learning()

# Global scheduler instance (singleton-like pattern)
_scheduler_instance: Optional[AdaptiveLearningScheduler] = None
_scheduler_lock = threading.Lock()

def get_scheduler() -> AdaptiveLearningScheduler:
    """Gets the global adaptive learning scheduler instance, creating if necessary."""
    global _scheduler_instance
    if _scheduler_instance is None:
        with _scheduler_lock:
            if _scheduler_instance is None: # Double-check locking
                _scheduler_instance = AdaptiveLearningScheduler()
    return _scheduler_instance

def start_adaptive_learning_scheduler():
    """Start the global adaptive learning scheduler."""
    scheduler = get_scheduler()
    scheduler.start()

def stop_adaptive_learning_scheduler():
    """Stop the global adaptive learning scheduler."""
    scheduler = get_scheduler()
    scheduler.stop()

def force_adaptive_learning() -> bool:
    """Force adaptive learning training immediately using the global scheduler instance."""
    scheduler = get_scheduler()
    return scheduler.force_training()

def is_adaptive_learning_scheduler_running() -> bool:
    """Check if the global adaptive learning scheduler is running."""
    global _scheduler_instance
    if _scheduler_instance:
        return _scheduler_instance.running and _scheduler_instance.thread is not None and _scheduler_instance.thread.is_alive()
    return False