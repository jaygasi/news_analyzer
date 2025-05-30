"""
Optimized notification utilities with improved rate limiting and error handling
"""
from typing import Any, List, Dict, Optional, Set
import threading
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass

from config import *
from utils.log_utils import *


@dataclass
class NotificationRecord:
    """Track notification history for rate limiting"""
    symbol: str
    timestamp: datetime
    alert_type: str
    fingerprint: str


class NotificationRateLimiter:
    """Enhanced rate limiting for notifications"""
    
    def __init__(self) -> None:
        self.notification_history: deque = deque(maxlen=1000)
        self.symbol_cooldowns: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        self.duplicate_fingerprints: Set[str] = set()
        self.last_cleanup = datetime.now()
    
    def can_send_notification(self, symbol: str, alert_type: str, message: Dict[str, Any]) -> bool:
        """Check if notification can be sent based on rate limits"""
        with self.lock:
            now = datetime.now()
            
            # Clean up old records periodically
            self._cleanup_old_records(now)
            
            # Check symbol-specific cooldown
            if symbol in self.symbol_cooldowns:
                if now < self.symbol_cooldowns[symbol]:
                    return False
            
            # Check for duplicates
            fingerprint = self._create_fingerprint(symbol, alert_type, message)
            if fingerprint in self.duplicate_fingerprints:
                return False
            
            # Check global rate limit
            recent_count = sum(1 for record in self.notification_history 
                             if (now - record.timestamp).total_seconds() < 60)
            
            if recent_count >= NOTIFICATION_RATE_LIMIT:
                return False
            
            # Record this notification
            self._record_notification(symbol, alert_type, fingerprint, now)
            return True
    
    def _create_fingerprint(self, symbol: str, alert_type: str, message: Dict[str, Any]) -> str:
        """Create unique fingerprint for notification"""
        title = message.get('title', '')[:50]
        return f"{symbol}:{alert_type}:{title}"
    
    def _record_notification(self, symbol: str, alert_type: str, fingerprint: str, timestamp: datetime) -> None:
        """Record notification for rate limiting"""
        # Add to history
        record = NotificationRecord(symbol, timestamp, alert_type, fingerprint)
        self.notification_history.append(record)
        
        # Add to duplicates
        self.duplicate_fingerprints.add(fingerprint)
        
        # Set cooldown for symbol
        cooldown_minutes = NOTIFICATION_COOLDOWN_MINUTES
        self.symbol_cooldowns[symbol] = timestamp + timedelta(minutes=cooldown_minutes)
    
    def _cleanup_old_records(self, now: datetime) -> None:
        """Clean up old records to prevent memory leaks"""
        if (now - self.last_cleanup).total_seconds() < 300:  # Clean every 5 minutes
            return
        
        # Remove old cooldowns
        expired_symbols = [symbol for symbol, cooldown_time in self.symbol_cooldowns.items() 
                          if now > cooldown_time]
        for symbol in expired_symbols:
            del self.symbol_cooldowns[symbol]
        
        # Clean duplicate fingerprints (keep only recent ones)
        recent_fingerprints = {record.fingerprint for record in self.notification_history 
                             if (now - record.timestamp).total_seconds() < NOTIFICATION_DUPLICATE_WINDOW}
        self.duplicate_fingerprints = recent_fingerprints
        
        self.last_cleanup = now


class NotificationBatcher:
    """Enhanced notification batcher with better batching logic"""
    
    def __init__(self) -> None:
        self.pending_notifications: List[Dict[str, Any]] = []
        self.batch_lock = threading.Lock()
        self.last_batch_time = datetime.now()
        self.batch_timeout = NOTIFICATION_BATCH_TIMEOUT
        self.max_batch_size = NOTIFICATION_BATCH_SIZE
    
    def add_notification(self, notification: Dict[str, Any]) -> None:
        """Add notification to pending batch"""
        with self.batch_lock:
            self.pending_notifications.append(notification)
    
    def should_flush_batch(self) -> bool:
        """Check if batch should be flushed"""
        with self.batch_lock:
            if not self.pending_notifications:
                return False
            
            time_since_last = (datetime.now() - self.last_batch_time).total_seconds()
            return (len(self.pending_notifications) >= self.max_batch_size or 
                   time_since_last >= self.batch_timeout)
    
    def flush_batch(self) -> List[Dict[str, Any]]:
        """Get and clear pending notifications"""
        with self.batch_lock:
            batch = self.pending_notifications.copy()
            self.pending_notifications.clear()
            self.last_batch_time = datetime.now()
            return batch


# Global instances
notification_rate_limiter = NotificationRateLimiter()
notification_batcher = NotificationBatcher()
_notification_system_shutdown_event = threading.Event()


def parse_email_addresses(email_string: str) -> List[str]:
    """Parse and validate comma-separated email addresses"""
    if not email_string:
        return []
    
    addresses = []
    for addr in email_string.split(','):
        addr = addr.strip()
        if addr and '@' in addr and '.' in addr:  # Basic email validation
            addresses.append(addr)
    
    return addresses


def send_notifications(notification_client: Any, symbol: str, message: Dict[str, Any], 
                      notification_type: str = "gmail") -> bool:
    """Enhanced notification sending with rate limiting"""
    try:
        if not ENABLE_NOTIFICATIONS or not SEND_EMAIL_NOTIFICATIONS:
            return False
        
        # Check if we can send this notification
        alert_type = _determine_alert_type(message)
        if not notification_rate_limiter.can_send_notification(symbol, alert_type, message):
            logd(f"📧 Notification rate limited for {symbol}")
            return False
        
        success = False
        
        if notification_type == "gmail" and hasattr(notification_client, 'send_trading_alert'):
            success = _send_gmail_notifications(notification_client, symbol, alert_type, message)
        elif notification_type == "sendgrid":
            success = _send_sendgrid_notification(notification_client, symbol, alert_type, message)
        
        if success:
            logi(f"📧 Notification sent for {symbol} ({alert_type})")
        
        return success
        
    except Exception as e:
        loge(f"Notification error for {symbol}: {str(e)}")
        return False


def _determine_alert_type(message: Dict[str, Any]) -> str:
    """Determine alert type from message"""
    topic_sentiment = message.get('topic_sentiment', '')
    
    if topic_sentiment == "positive":
        return "POSITIVE CATALYST"
    elif topic_sentiment == "negative":
        return "NEGATIVE CATALYST"
    else:
        return "CATALYST"


def _send_gmail_notifications(notification_client: Any, symbol: str, alert_type: str, 
                            message: Dict[str, Any]) -> bool:
    """Send Gmail notifications with enhanced error handling"""
    try:
        email_addresses = parse_email_addresses(ALERT_TO_EMAIL or "")
        
        if not email_addresses:
            logw("No email addresses configured for notifications")
            return False
        
        # Limit to first 3 addresses to prevent spam
        limited_addresses = email_addresses[:3]
        success_count = 0
        
        for email_addr in limited_addresses:
            try:
                result = notification_client.send_trading_alert(
                    to_email=email_addr,
                    symbol=symbol,
                    alert_type=alert_type,
                    message=message
                )
                
                if result:
                    success_count += 1
                    logd(f"📧 Alert queued for {email_addr}")
                else:
                    logw(f"📧 Failed to queue alert for {email_addr}")
                    
            except Exception as ex:
                logw(f"Failed to send notification to {email_addr}: {str(ex)}")
        
        if success_count > 0:
            logi(f"📧 Notifications queued for {symbol} to {success_count}/{len(limited_addresses)} addresses")
            return True
            
        return False
        
    except Exception as e:
        loge(f"Gmail notification error: {str(e)}")
        return False


def _send_sendgrid_notification(notification_client: Any, symbol: str, alert_type: str, 
                              message: Dict[str, Any]) -> bool:
    """Send notification via SendGrid (fallback)"""
    try:
        title = message.get('title', '')
        if len(title) > 100:
            title = title[:100] + "..."
        
        subject = f"🚨 {alert_type}: {symbol}"
        body = f"""
        Catalyst detected for {symbol}!
        
        📰 News: {title}
        📊 Topic: {message.get('news_topic', '')}
        💫 Sentiment: {message.get('topic_sentiment', '')}
        🔑 Keywords: {message.get('topic_keyword_count', '')}
        
        ⚠️ This is not financial advice.
        """
        
        notification_client.send_email(ALERT_FROM_EMAIL, subject, body, ALERT_TO_EMAIL)
        return True
        
    except Exception as e:
        loge(f"SendGrid notification failed: {str(e)}")
        return False


def send_system_status_email(notification_client: Any, stats: Dict[str, Any], 
                           notification_type: str = "gmail") -> bool:
    """Send enhanced system status email"""
    try:
        if not ENABLE_NOTIFICATIONS or not SEND_DAILY_SUMMARY:
            return False
            
        subject = "📊 News Catalyst System - Daily Status Report"
        
        if notification_type == "gmail":
            html_body = _create_status_email_body(stats)
            
            # Send to first email address only for status reports
            email_addresses = parse_email_addresses(ALERT_TO_EMAIL or "")
            if email_addresses:
                return notification_client.send_email_async(
                    to_email=email_addresses[0], 
                    subject=subject, 
                    body=html_body,
                    priority=3
                )
    except Exception as e:
        loge(f"System status email failed: {str(e)}")
        return False


def _create_status_email_body(stats: Dict[str, Any]) -> str:
    """Create HTML body for status email"""
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; margin: 20px;">
        <h2>📊 News Catalyst Trading System - Daily Status</h2>
        <h3>📈 Performance Summary:</h3>
        <ul>
            <li><strong>Catalysts Detected:</strong> {stats.get('catalysts_detected', 0)}</li>
            <li><strong>Trades Executed:</strong> {stats.get('trades_executed', 0)}</li>
            <li><strong>Success Rate:</strong> {stats.get('success_rate', 0)}%</li>
            <li><strong>Total P&L:</strong> ${stats.get('total_pnl', 0):.2f}</li>
        </ul>
        <h3>🔧 System Health:</h3>
        <ul>
            <li><strong>Uptime:</strong> {stats.get('uptime', 'N/A')}</li>
            <li><strong>Memory Usage:</strong> {stats.get('memory_usage', 'N/A')}</li>
            <li><strong>CPU Usage:</strong> {stats.get('cpu_usage', 'N/A')}</li>
        </ul>
        <h3>📧 Email System:</h3>
        <ul>
            <li><strong>Emails Sent:</strong> {stats.get('emails_sent', 0)}</li>
            <li><strong>Rate Limits Hit:</strong> {stats.get('rate_limits_hit', 0)}</li>
            <li><strong>Queue Size:</strong> {stats.get('queue_size', 0)}</li>
            <li><strong>Daily Emails Remaining:</strong> {stats.get('daily_emails_remaining', 'N/A')}</li>
        </ul>
    </body>
    </html>
    """


def send_startup_notification(notification_client: Any, symbol_count: int = 0, 
                            notification_type: str = "gmail") -> bool:
    """Send enhanced startup notification"""
    try:
        if notification_type == "gmail" and notification_client:
            email_addresses = parse_email_addresses(ALERT_TO_EMAIL or "")
            
            if not email_addresses:
                return False
            
            # Send to first email address only for startup notifications
            success = notification_client.send_email_async(
                to_email=email_addresses[0],
                subject="🚀 News Catalyst Trading System Started",
                body=_create_startup_email_body(symbol_count, len(email_addresses)),
                priority=1
            )
            
            if success:
                logi("📧 Startup notification queued")
            return success
    except Exception as e:
        loge(f"Startup notification error: {str(e)}")
        return False


def _create_startup_email_body(symbol_count: int, address_count: int) -> str:
    """Create HTML body for startup notification"""
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; margin: 20px;">
        <h2>🚀 News Catalyst Trading System Online</h2>
        <p><strong>Started:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Notification Method:</strong> Gmail ✅</p>
        <p><strong>Status:</strong> All systems operational</p>
        <p><strong>Monitoring:</strong> {symbol_count} stocks</p>
        <p><strong>Configured Addresses:</strong> {address_count}</p>
        <p><strong>Rate Limit:</strong> {GMAIL_MAX_RATE_PER_MINUTE} emails/minute</p>
        <hr>
        <p><small>🤖 You will receive alerts when catalysts are detected.</small></p>
    </body>
    </html>
    """


def send_shutdown_notification(notification_client: Any, notification_type: str = "gmail") -> bool:
    """Send shutdown notification"""
    try:
        if notification_type == "gmail" and notification_client:
            email_addresses = parse_email_addresses(ALERT_TO_EMAIL or "")
            
            if not email_addresses:
                return False
            
            # Send to first email address only for shutdown notifications
            success = notification_client.send_email(
                to_email=email_addresses[0],
                subject="🛑 News Catalyst Trading System Shutdown", 
                body=f"""
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <h2>🛑 System Shutdown Notice</h2>
                    <p><strong>Shutdown Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Status:</strong> Graceful shutdown initiated</p>
                    <p><small>🤖 System will be offline until manually restarted.</small></p>
                </body>
                </html>
                """
            )
            
            if success:
                logi("📧 Shutdown notification sent")
            return success
    except Exception as e:
        logw(f"Shutdown notification error: {str(e)}")
        return False


def start_notification_batch_monitor(notification_client: Any) -> None:
    """Start a background thread to periodically flush notification batches"""
    def batch_monitor() -> None:
        logi("📦 Enhanced notification batch monitor started")
        while not _notification_system_shutdown_event.is_set():
            try:
                if notification_batcher.should_flush_batch():
                    flush_notification_batch(notification_client)
                
                _notification_system_shutdown_event.wait(2.0)  # Check more frequently
            except Exception as e:
                if _notification_system_shutdown_event.is_set():
                    break
                loge(f"Error in notification batch monitor: {str(e)}")
                _notification_system_shutdown_event.wait(5.0)
        logi("📦 Notification batch monitor stopped")
    
    monitor_thread = threading.Thread(target=batch_monitor, name="NotificationBatchMonitor", daemon=True)
    monitor_thread.start()


def flush_notification_batch(notification_client: Any) -> None:
    """Enhanced batch flushing with better error handling"""
    try:
        batch = notification_batcher.flush_batch()
        if not batch:
            return
        
        logd(f"📦 Flushing notification batch: {len(batch)} notifications")
        
        # Sort by priority
        batch.sort(key=lambda x: x.get('priority', 2))
        
        successful_sends = 0
        for notification in batch:
            try:
                success = notification_client.send_email_async(
                    to_email=notification['to_email'],
                    subject=notification['subject'],
                    body=notification['body'],
                    priority=notification.get('priority', 2)
                )
                
                if success:
                    successful_sends += 1
                    logd(f"📧 Notification queued for {notification.get('symbol', 'unknown')}")
                else:
                    logw(f"📧 Notification failed for {notification.get('symbol', 'unknown')}")
                    
            except Exception as ex:
                loge(f"📧 Notification error for {notification.get('symbol', 'unknown')}: {str(ex)}")
        
        if batch:
            logi(f"📧 Batch processed: {successful_sends}/{len(batch)} notifications sent")
            
    except Exception as e:
        loge(f"Error flushing notification batch: {str(e)}")


def stop_notification_system() -> None:
    """Stop the notification system"""
    global _notification_system_shutdown_event
    _notification_system_shutdown_event.set()
    logi("🛑 Notification system shutdown requested")
    
    if notification_batcher:
        logi("📦 Flushing final notification batch...")
        try:
            # Give the system a moment to process
            time.sleep(1)
        except Exception as e:
            logw(f"Error during final notification batch flush: {str(e)}")

    logi("✅ Notification system stopped")
    notification_rate_limiter.notification_history.clear()