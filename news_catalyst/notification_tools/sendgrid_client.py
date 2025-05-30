from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
from utils.log_utils import *
import os


class SendGridClient:
    def __init__(self, sendgrid_api_key):
        self.sendgrid_client = SendGridAPIClient(sendgrid_api_key)

    def send_email(self, from_email, subject, body, to_email, attachment_path=None):
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            html_content=body)

        # Add attachment
        if attachment_path and os.path.exists(attachment_path):
            # Read the attachment file
            with open(attachment_path, 'rb') as f:
                file_data = f.read()
                f.close()

            # Create the attachment
            encoded_file = base64.b64encode(file_data).decode()
            attachment = Attachment(
                FileContent(encoded_file),
                FileName(os.path.basename(attachment_path)),
                FileType('image/png'),  # Change the MIME type if necessary
                Disposition('attachment')
            )
            message.attachment = attachment

        try:
            self.sendgrid_client.send(message)
            logd(f"Email sent successfully!")
        except Exception as e:
            loge(f"Failed to send email: {e}")
