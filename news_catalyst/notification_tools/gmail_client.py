"""
Enterprise-grade Gmail client with advanced features and reliability
"""
import smtplib
import threading
import time
import ssl
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from queue import Empty, PriorityQueue, Queue
from typing import Dict, List, Optional, Callable, Any, Set
import heapq
import json

from utils.log_utils import logd, loge, logi, logw
from utils.performance_monitor import register_component_performance


@dataclass
class EmailTemplate:
    """Email template for different notification types"""
    subject_template: str
    body_template: str
    priority: int = 2
    max_length: int = 2000


@dataclass
class EmailMessage:
    """Enhanced email message with priority and retry logic"""
    to_email: str
    subject: str
    body: str
    priority: int = 2  # 1=high, 2=normal, 3=low
    from_email: Optional[str] = None
    cc_emails: Optional[List[str]] = None
    attachment_path: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    template_used: Optional[str] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


class RateLimiter:
    """Advanced rate limiter with burst capacity and adaptive limits"""
    
    def __init__(self, max_rate: int = 25, burst_capacity: int = 5, window_seconds: int = 60):
        self.max_rate = max_rate
        self.burst_capacity = burst_capacity
        self.window_seconds = window_seconds
        
        self._tokens = burst_capacity
        self._last_refill = time.time()
        self._send_times: deque = deque(maxlen=max_rate * 2)
        self._lock = threading.RLock()
        
        # Adaptive rate limiting
        self._recent_failures: deque = deque(maxlen=100)
        self._adaptive_factor = 1.0
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire permission to send with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._lock:
                self._refill_tokens()
                
                if self._tokens >= 1:
                    self._tokens -= 1
                    self._send_times.append(time.time())
                    return True
            
            # Brief sleep before retry
            time.sleep(0.1)
        
        return False
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self._last_refill
        
        if elapsed > 0:
            # Add tokens based on rate limit
            tokens_to_add = elapsed * (self.max_rate * self._adaptive_factor / self.window_seconds)
            self._tokens = min(self.burst_capacity, self._tokens + tokens_to_add)
            self._last_refill = now
    
    def record_failure(self) -> None:
        """Record a failure for adaptive rate limiting"""
        with self._lock:
            self._recent_failures.append(time.time())
            
            # Reduce rate if too many recent failures
            recent_failure_count = sum(
                1 for failure_time in self._recent_failures
                if time.time() - failure_time < 300  # Last 5 minutes
            )
            
            if recent_failure_count > 10:
                self._adaptive_factor = max(0.1, self._adaptive_factor * 0.8)
                logi(f"Adaptive rate limiting: reduced factor to {self._adaptive_factor:.2f}")
    
    def record_success(self) -> None:
        """Record a success to gradually restore rate"""
        with self._lock:
            self._adaptive_factor = min(1.0, self._adaptive_factor * 1.01)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            now = time.time()
            recent_sends = sum(
                1 for send_time in self._send_times
                if now - send_time < self.window_seconds
            )
            
            return {
                'current_tokens': self._tokens,
                'recent_sends': recent_sends,
                'adaptive_factor': self._adaptive_factor,
                'recent_failures': len([
                    f for f in self._recent_failures
                    if now - f < 300
                ])
            }


class EmailValidator:
    """Email validation and sanitization"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email.strip()))
    
    @staticmethod
    def sanitize_subject(subject: str, max_length: int = 100) -> str:
        """Sanitize email subject"""
        # Remove dangerous characters
        sanitized = ''.join(c for c in subject if c.isprintable())
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length-3] + "..."
        return sanitized
    
    @staticmethod
    def sanitize_body(body: str, max_length: int = 5000) -> str:
        """Sanitize email body"""
        # Remove or escape dangerous content
        sanitized = body.replace('<script', '&lt;script').replace('</script>', '&lt;/script&gt;')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length-20] + "\n\n[Content truncated]"
        
        return sanitized


class ConnectionPool:
    """SMTP connection pool for better performance"""
    
    def __init__(self, smtp_server: str, smtp_port: int, email: str, password: str, pool_size: int = 3):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email = email
        self.password = password
        self.pool_size = pool_size
        
        self._connections: Queue = Queue(maxsize=pool_size)
        self._lock = threading.RLock()
        self._total_created = 0
        self._active_connections = 0
    
    def get_connection(self) -> Optional[smtplib.SMTP]:
        """Get connection from pool or create new one"""
        try:
            # Try to get existing connection
            connection = self._connections.get_nowait()
            
            # Test if connection is still alive
            try:
                status = connection.noop()
                if status[0] == 250:
                    return connection
            except:
                pass
            
            # Connection is dead, close it
            try:
                connection.quit()
            except:
                pass
        except Empty:
            pass
        
        # Create new connection
        try:
            connection = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)
            connection.starttls(context=ssl.create_default_context())
            connection.login(self.email, self.password)
            
            with self._lock:
                self._total_created += 1
                self._active_connections += 1
            
            return connection
        except Exception as e:
            loge(f"Failed to create SMTP connection: {e}")
            return None
    
    def return_connection(self, connection: smtplib.SMTP) -> None:
        """Return connection to pool"""
        try:
            if connection and self._connections.qsize() < self.pool_size:
                self._connections.put_nowait(connection)
            else:
                connection.quit()
                with self._lock:
                    self._active_connections -= 1
        except:
            with self._lock:
                self._active_connections -= 1
    
    def close_all(self) -> None:
        """Close all connections in pool"""
        while True:
            try:
                connection = self._connections.get_nowait()
                try:
                    connection.quit()
                except:
                    pass
            except Empty:
                break
        
        with self._lock:
            self._active_connections = 0


class EnterpriseGmailClient:
    """Enterprise-grade Gmail client with advanced features"""
    
    def __init__(self, 
                 gmail_email: str, 
                 gmail_app_password: str,
                 max_rate_per_minute: int = 25,
                 connection_pool_size: int = 3):
        
        self.gmail_email = gmail_email
        self.gmail_app_password = gmail_app_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Validation
        if not EmailValidator.validate_email(gmail_email):
            raise ValueError(f"Invalid Gmail address: {gmail_email}")
        
        # Core components
        self.rate_limiter = RateLimiter(max_rate_per_minute)
        self.connection_pool = ConnectionPool(
            self.smtp_server, self.smtp_port, 
            gmail_email, gmail_app_password,
            connection_pool_size
        )
        
        # Queue management
        self.email_queue: PriorityQueue = PriorityQueue()
        self.failed_queue: Queue = Queue()
        
        # Email templates
        self.templates = self._load_email_templates()
        
        # State management
        self.is_running = False
        self._shutdown_event = threading.Event()
        self._processor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'emails_sent': 0,
            'emails_failed': 0,
            'emails_queued': 0,
            'rate_limits_hit': 0,
            'connections_created': 0,
            'daily_limit_hit': False,
            'last_reset': datetime.now(),
            'processing_times': deque(maxlen=100)
        }
        self._stats_lock = threading.RLock()
        
        # Duplicate detection
        self._recent_emails: Set[str] = set()
        self._dedupe_lock = threading.RLock()
        
        # Test connection
        if not self._test_connection():
            raise ConnectionError("Failed to establish initial Gmail connection")
    
    def _load_email_templates(self) -> Dict[str, EmailTemplate]:
        """Load email templates for different notification types"""
        return {
            'trading_alert': EmailTemplate(
                subject_template="🚨 {alert_type}: {symbol}",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <h2 style="color: #2E8B57;">📈 Trading Alert: {symbol}</h2>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <p><strong>Alert Type:</strong> {alert_type}</p>
                        <p><strong>Symbol:</strong> {symbol}</p>
                        <p><strong>Topic:</strong> {topic}</p>
                        <p><strong>Sentiment:</strong> {sentiment}</p>
                        <p><strong>Confidence:</strong> {confidence}/5</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    <h3>📰 News Summary:</h3>
                    <p>{news_title}</p>
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        🤖 News Catalyst Trading System | 
                        <a href="{url}">View Full Article</a>
                    </p>
                </body>
                </html>
                """,
                priority=1,
                max_length=2000
            ),
            'system_status': EmailTemplate(
                subject_template="📊 System Status: {status}",
                body_template="""
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <h2>📊 Trading System Status Report</h2>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <h3>📈 Performance Summary:</h3>
                        <ul>
                            <li><strong>Status:</strong> {status}</li>
                            <li><strong>Uptime:</strong> {uptime}</li>
                            <li><strong>Alerts Sent:</strong> {alerts_sent}</li>
                            <li><strong>Memory Usage:</strong> {memory_usage}</li>
                        </ul>
                    </div>
                    <hr>
                    <p style="font-size: 12px; color: #666;">
                        🤖 Automated System Report - {timestamp}
                    </p>
                </body>
                </html>
                """,
                priority=3
            )
        }
    
    def _test_connection(self) -> bool:
        """Test Gmail connection"""
        try:
            connection = self.connection_pool.get_connection()
            if connection:
                self.connection_pool.return_connection(connection)
                logi("✅ Gmail connection test successful")
                return True
            return False
        except Exception as e:
            loge(f"❌ Gmail connection test failed: {e}")
            return False
    
    def _generate_dedup_key(self, to_email: str, subject: str) -> str:
        """Generate deduplication key"""
        import hashlib
        content = f"{to_email}:{subject}:{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, to_email: str, subject: str) -> bool:
        """Check if email is duplicate"""
        dedup_key = self._generate_dedup_key(to_email, subject)
        
        with self._dedupe_lock:
            if dedup_key in self._recent_emails:
                return True
            
            self._recent_emails.add(dedup_key)
            
            # Cleanup old entries (keep last 1000)
            if len(self._recent_emails) > 1000:
                self._recent_emails.clear()
        
        return False
    
    def _send_email_immediate(self, message: EmailMessage) -> bool:
        """Send email immediately with connection pooling"""
        if self._shutdown_event.is_set() or self.stats['daily_limit_hit']:
            return False
        
        start_time = time.time()
        
        try:
            # Check for duplicates
            if self._is_duplicate(message.to_email, message.subject):
                logd(f"Skipping duplicate email to {message.to_email}")
                return True
            
            # Rate limiting
            if not self.rate_limiter.acquire(timeout=30):
                with self._stats_lock:
                    self.stats['rate_limits_hit'] += 1
                logw("Rate limit hit, email queued for retry")
                return False
            
            # Get connection from pool
            connection = self.connection_pool.get_connection()
            if not connection:
                return False
            
            try:
                # Build email
                msg = MIMEMultipart('alternative')
                msg["From"] = message.from_email or self.gmail_email
                msg["To"] = message.to_email
                msg["Subject"] = EmailValidator.sanitize_subject(message.subject)
                
                if message.cc_emails:
                    msg["Cc"] = ", ".join(message.cc_emails)
                
                # Attach body (support both HTML and plain text)
                sanitized_body = EmailValidator.sanitize_body(message.body)
                if "<html>" in sanitized_body.lower():
                    msg.attach(MIMEText(sanitized_body, "html", "utf-8"))
                else:
                    msg.attach(MIMEText(sanitized_body, "plain", "utf-8"))
                
                # Add attachment if specified
                if message.attachment_path:
                    self._add_attachment(msg, message.attachment_path)
                
                # Send email
                recipients = [message.to_email]
                if message.cc_emails:
                    recipients.extend(message.cc_emails)
                
                text = msg.as_string()
                result = connection.sendmail(self.gmail_email, recipients, text)
                
                # Update statistics
                with self._stats_lock:
                    self.stats['emails_sent'] += 1
                    processing_time = (time.time() - start_time) * 1000
                    self.stats['processing_times'].append(processing_time)
                
                self.rate_limiter.record_success()
                logd(f"📧 Email sent to {message.to_email}: {message.subject}")
                
                return True
                
            finally:
                self.connection_pool.return_connection(connection)
                
        except smtplib.SMTPException as e:
            error_str = str(e).lower()
            
            # Check for daily limit
            if any(phrase in error_str for phrase in [
                "daily sending limit", "550 5.4.5", "quota exceeded", "sending limits"
            ]):
                with self._stats_lock:
                    self.stats['daily_limit_hit'] = True
                loge(f"📧 Gmail daily sending limit reached")
                return False
            
            # Other SMTP errors
            with self._stats_lock:
                self.stats['emails_failed'] += 1
            
            self.rate_limiter.record_failure()
            loge(f"❌ SMTP error sending to {message.to_email}: {e}")
            return False
            
        except Exception as e:
            with self._stats_lock:
                self.stats['emails_failed'] += 1
            
            loge(f"❌ Failed to send email to {message.to_email}: {e}")
            return False
    
    def _add_attachment(self, message: MIMEMultipart, attachment_path: str) -> None:
        """Add attachment to email"""
        try:
            import os
            import mimetypes
            
            if not os.path.exists(attachment_path):
                logw(f"Attachment file not found: {attachment_path}")
                return
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(attachment_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            main_type, sub_type = mime_type.split('/', 1)
            
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase(main_type, sub_type)
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(attachment_path)}'
            )
            message.attach(part)
            
        except Exception as e:
            logw(f"Failed to add attachment {attachment_path}: {e}")
    
    def _processor_loop(self) -> None:
        """Main email processing loop"""
        logi("📧 Gmail processor started")
        
        while not self._shutdown_event.is_set():
            try:
                # Process high priority emails first
                try:
                    message = self.email_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                success = self._send_email_immediate(message)
                
                if not success and message.retry_count < message.max_retries:
                    # Retry failed messages
                    message.retry_count += 1
                    delay = min(2 ** message.retry_count, 60)  # Exponential backoff, max 60s
                    
                    def delayed_retry():
                        if not self._shutdown_event.is_set():
                            self.email_queue.put(message)
                    
                    timer = threading.Timer(delay, delayed_retry)
                    timer.daemon = True
                    timer.start()
                    
                    logd(f"📧 Retrying email to {message.to_email} in {delay}s (attempt {message.retry_count})")
                
                self.email_queue.task_done()
                
                # Brief pause between emails
                if not self._shutdown_event.wait(0.2):
                    continue
                    
            except Exception as e:
                loge(f"Error in Gmail processor loop: {e}")
                if not self._shutdown_event.wait(5.0):
                    continue
        
        logi("📧 Gmail processor stopped")
    
    def start(self) -> None:
        """Start email processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            name="GmailProcessor",
            daemon=True
        )
        self._processor_thread.start()
        logi("📧 Gmail client started")
    
    def stop(self, timeout: float = 10.0) -> None:
        """Stop email processing gracefully"""
        if not self.is_running:
            return
        
        logi("🛑 Stopping Gmail client...")
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for processor to stop
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=timeout)
            if self._processor_thread.is_alive():
                logw("Gmail processor thread did not stop gracefully")
        
        # Process remaining high-priority emails quickly
        remaining_processed = 0
        start_time = time.time()
        
        while (not self.email_queue.empty() and 
               time.time() - start_time < timeout and
               remaining_processed < 10):  # Limit processing
            
            try:
                message = self.email_queue.get_nowait()
                if message.priority <= 2:  # Only high/normal priority
                    if self._send_email_immediate(message):
                        remaining_processed += 1
                self.email_queue.task_done()
            except Empty:
                break
            except Exception as e:
                logw(f"Error processing remaining email: {e}")
                break
        
        # Close connection pool
        self.connection_pool.close_all()
        
        if remaining_processed > 0:
            logi(f"📧 Processed {remaining_processed} remaining emails")
        
        logi("✅ Gmail client stopped")
    
    def send_email_async(self, 
                        to_email: str, 
                        subject: str, 
                        body: str,
                        priority: int = 2,
                        from_email: Optional[str] = None,
                        cc_emails: Optional[List[str]] = None,
                        attachment_path: Optional[str] = None,
                        template: Optional[str] = None) -> bool:
        """Queue email for asynchronous sending"""
        
        if self._shutdown_event.is_set() or self.stats['daily_limit_hit']:
            return False
        
        # Validate inputs
        if not EmailValidator.validate_email(to_email):
            loge(f"Invalid email address: {to_email}")
            return False
        
        if cc_emails:
            cc_emails = [email for email in cc_emails if EmailValidator.validate_email(email)]
        
        try:
            message = EmailMessage(
                to_email=to_email,
                subject=subject,
                body=body,
                priority=priority,
                from_email=from_email,
                cc_emails=cc_emails,
                attachment_path=attachment_path,
                template_used=template
            )
            
            self.email_queue.put(message)
            
            with self._stats_lock:
                self.stats['emails_queued'] += 1
            
            logd(f"📧 Queued email to {to_email}: {subject}")
            return True
            
        except Exception as e:
            loge(f"❌ Failed to queue email: {e}")
            return False
    
    def send_email(self, to_email: str, subject: str, body: str, **kwargs) -> bool:
        """Send email synchronously (for compatibility)"""
        if self._shutdown_event.is_set() or self.stats['daily_limit_hit']:
            return False
        
        message = EmailMessage(
            to_email=to_email,
            subject=subject,
            body=body,
            from_email=kwargs.get('from_email'),
            cc_emails=kwargs.get('cc_emails'),
            attachment_path=kwargs.get('attachment_path'),
            priority=1  # High priority for sync sends
        )
        
        return self._send_email_immediate(message)
    
    def send_trading_alert(self, 
                          to_email: str, 
                          symbol: str, 
                          alert_type: str, 
                          message: Dict[str, Any]) -> bool:
        """Send formatted trading alert"""
        
        template = self.templates['trading_alert']
        
        subject = template.subject_template.format(
            alert_type=alert_type,
            symbol=symbol
        )
        
        body = template.body_template.format(
            symbol=symbol,
            alert_type=alert_type,
            topic=message.get('news_topic', 'N/A'),
            sentiment=message.get('topic_sentiment', 'N/A'),
            confidence=message.get('topic_keyword_count', 'N/A'),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            news_title=message.get('title', 'N/A')[:200],
            url=message.get('url', '#')
        )
        
        return self.send_email_async(
            to_email=to_email,
            subject=subject,
            body=body,
            priority=template.priority,
            template='trading_alert'
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._stats_lock:
            queue_size = self.email_queue.qsize()
            rate_limiter_stats = self.rate_limiter.get_stats()
            
            # Calculate average processing time
            avg_processing_time = 0
            if self.stats['processing_times']:
                avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            
            return {
                **self.stats,
                'queue_size': queue_size,
                'is_running': self.is_running,
                'success_rate': (
                    self.stats['emails_sent'] / 
                    max(self.stats['emails_sent'] + self.stats['emails_failed'], 1)
                ),
                'avg_processing_time_ms': avg_processing_time,
                'rate_limiter': rate_limiter_stats,
                'connection_pool_active': self.connection_pool._active_connections
            }
    
    def reset_daily_limits(self) -> None:
        """Reset daily limits (call this daily)"""
        with self._stats_lock:
            self.stats['daily_limit_hit'] = False
            self.stats['last_reset'] = datetime.now()
        logi("📧 Gmail daily limits reset")


# Backward compatibility
OptimizedGmailClient = EnterpriseGmailClient
GmailClient = EnterpriseGmailClient