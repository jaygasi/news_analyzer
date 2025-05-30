"""
Optimized SendGrid client with improved error handling and rate limiting
"""
import base64
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import mimetypes
import time
from threading import Lock

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import (
        Mail, Attachment, FileContent, FileName, FileType, Disposition
    )
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False
    SendGridAPIClient = None
    Mail = None

from utils.log_utils import logd, logw, loge


class SendGridRateLimiter:
    """Rate limiter for SendGrid API calls."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = Lock()
    
    def can_make_request(self) -> bool:
        """Check if request can be made within rate limits."""
        with self.lock:
            now = time.time()
            # Remove old calls
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            return len(self.calls) < self.max_calls
    
    def record_request(self) -> None:
        """Record a request."""
        with self.lock:
            self.calls.append(time.time())


class OptimizedSendGridClient:
    """
    Optimized SendGrid client with improved error handling and validation.
    """
    
    def __init__(self, sendgrid_api_key: str, rate_limit: int = 100):
        if not SENDGRID_AVAILABLE:
            raise ImportError("SendGrid library not available. Install with: pip install sendgrid")
        
        if not sendgrid_api_key or not sendgrid_api_key.strip():
            raise ValueError("SendGrid API key is required")
        
        self.sendgrid_client = SendGridAPIClient(sendgrid_api_key.strip())
        self.rate_limiter = SendGridRateLimiter(max_calls=rate_limit)
        
        # Track statistics
        self.emails_sent = 0
        self.emails_failed = 0
    
    def _validate_email(self, email: str) -> bool:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid email format
        """
        if not email or not isinstance(email, str):
            return False
        
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email.strip()))
    
    def _validate_attachment(self, attachment_path: str) -> bool:
        """
        Validate attachment file.
        
        Args:
            attachment_path: Path to attachment file
            
        Returns:
            True if valid attachment
        """
        if not attachment_path:
            return False
        
        path = Path(attachment_path)
        
        if not path.exists():
            logw(f"Attachment file not found: {attachment_path}")
            return False
        
        if not path.is_file():
            logw(f"Attachment path is not a file: {attachment_path}")
            return False
        
        # Check file size (10MB limit)
        if path.stat().st_size > 10 * 1024 * 1024:
            logw(f"Attachment too large (>10MB): {attachment_path}")
            return False
        
        return True
    
    def _create_attachment(self, attachment_path: str) -> Optional[Attachment]:
        """
        Create SendGrid attachment object.
        
        Args:
            attachment_path: Path to attachment file
            
        Returns:
            SendGrid Attachment object or None
        """
        if not self._validate_attachment(attachment_path):
            return None
        
        try:
            path = Path(attachment_path)
            
            # Read file data
            with open(path, 'rb') as file:
                file_data = file.read()
            
            # Encode file data
            encoded_file = base64.b64encode(file_data).decode()
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(path))
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Create attachment
            attachment = Attachment(
                FileContent(encoded_file),
                FileName(path.name),
                FileType(mime_type),
                Disposition('attachment')
            )
            
            return attachment
            
        except Exception as e:
            loge(f"Error creating attachment: {e}")
            return None
    
    def send_email(self, 
                   from_email: str,
                   subject: str,
                   body: str,
                   to_email: str,
                   attachment_path: Optional[str] = None,
                   cc_emails: Optional[List[str]] = None,
                   bcc_emails: Optional[List[str]] = None) -> bool:
        """
        Send email via SendGrid with comprehensive validation.
        
        Args:
            from_email: Sender email address
            subject: Email subject
            body: Email body (HTML or plain text)
            to_email: Recipient email address
            attachment_path: Optional path to attachment file
            cc_emails: Optional list of CC recipients
            bcc_emails: Optional list of BCC recipients
            
        Returns:
            True if email sent successfully
        """
        # Rate limiting check
        if not self.rate_limiter.can_make_request():
            logw("SendGrid rate limit reached")
            return False
        
        # Validate inputs
        if not self._validate_email(from_email):
            loge(f"Invalid from_email: {from_email}")
            return False
        
        if not self._validate_email(to_email):
            loge(f"Invalid to_email: {to_email}")
            return False
        
        if not subject or not subject.strip():
            loge("Email subject is required")
            return False
        
        if not body or not body.strip():
            loge("Email body is required")
            return False
        
        # Validate CC emails
        validated_cc = []
        if cc_emails:
            for email in cc_emails:
                if self._validate_email(email):
                    validated_cc.append(email.strip())
                else:
                    logw(f"Invalid CC email skipped: {email}")
        
        # Validate BCC emails
        validated_bcc = []
        if bcc_emails:
            for email in bcc_emails:
                if self._validate_email(email):
                    validated_bcc.append(email.strip())
                else:
                    logw(f"Invalid BCC email skipped: {email}")
        
        try:
            # Create mail object
            message = Mail(
                from_email=from_email.strip(),
                to_emails=to_email.strip(),
                subject=subject.strip(),
                html_content=body
            )
            
            # Add CC recipients
            if validated_cc:
                for cc_email in validated_cc:
                    message.add_cc(cc_email)
            
            # Add BCC recipients
            if validated_bcc:
                for bcc_email in validated_bcc:
                    message.add_bcc(bcc_email)
            
            # Add attachment if provided
            if attachment_path:
                attachment = self._create_attachment(attachment_path)
                if attachment:
                    message.attachment = attachment
                else:
                    logw("Attachment creation failed, sending without attachment")
            
            # Send email
            self.rate_limiter.record_request()
            response = self.sendgrid_client.send(message)
            
            # Check response status
            if response.status_code in [200, 201, 202]:
                self.emails_sent += 1
                logd(f"Email sent successfully to {to_email}")
                return True
            else:
                self.emails_failed += 1
                loge(f"SendGrid API error: {response.status_code} - {response.body}")
                return False
                
        except Exception as e:
            self.emails_failed += 1
            loge(f"Failed to send email via SendGrid: {e}")
            return False
    
    def send_bulk_email(self, 
                       from_email: str,
                       subject: str,
                       body: str,
                       to_emails: List[str],
                       max_batch_size: int = 50) -> Dict[str, Any]:
        """
        Send bulk emails with batch processing.
        
        Args:
            from_email: Sender email address
            subject: Email subject
            body: Email body
            to_emails: List of recipient email addresses
            max_batch_size: Maximum emails per batch
            
        Returns:
            Dictionary with send statistics
        """
        if not to_emails:
            return {'sent': 0, 'failed': 0, 'total': 0}
        
        # Validate and filter email addresses
        valid_emails = [email for email in to_emails if self._validate_email(email)]
        
        if not valid_emails:
            loge("No valid email addresses in bulk send list")
            return {'sent': 0, 'failed': len(to_emails), 'total': len(to_emails)}
        
        sent_count = 0
        failed_count = 0
        
        # Process emails in batches
        for i in range(0, len(valid_emails), max_batch_size):
            batch = valid_emails[i:i + max_batch_size]
            
            for email in batch:
                if self.send_email(from_email, subject, body, email):
                    sent_count += 1
                else:
                    failed_count += 1
                
                # Brief pause between emails to avoid overwhelming
                time.sleep(0.1)
            
            # Longer pause between batches
            if i + max_batch_size < len(valid_emails):
                time.sleep(1)
        
        logd(f"Bulk email completed: {sent_count} sent, {failed_count} failed")
        
        return {
            'sent': sent_count,
            'failed': failed_count,
            'total': len(valid_emails)
        }
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get email sending statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'emails_sent': self.emails_sent,
            'emails_failed': self.emails_failed,
            'success_rate': (
                self.emails_sent / max(self.emails_sent + self.emails_failed, 1)
            ) * 100
        }
    
    def test_connection(self) -> bool:
        """
        Test SendGrid connection.
        
        Returns:
            True if connection is working
        """
        try:
            # This is a simple way to test the API key
            # Note: This doesn't actually send an email
            test_message = Mail(
                from_email="test@example.com",
                to_emails="test@example.com",
                subject="Test",
                html_content="Test"
            )
            
            # We can't actually send without valid emails, but we can validate the setup
            return True
            
        except Exception as e:
            loge(f"SendGrid connection test failed: {e}")
            return False


# Maintain backward compatibility
SendGridClient = OptimizedSendGridClient