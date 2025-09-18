import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailUtil:
    @staticmethod
    def send_email(smtp_server: str, port: int, login: str, password: str, recipient: str, subject: str, body: str) -> str:
        """
        Send an email using the specified SMTP server.

        :param smtp_server: Address of the SMTP server.
        :param port: Port number for the SMTP server.
        :param login: Email account login.
        :param password: Email account password.
        :param recipient: Recipient email address.
        :param subject: Email subject.
        :param body: Email body content.
        :return: Confirmation of email sent.
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = login
            msg['To'] = recipient
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(login, password)
                server.send_message(msg)

            return f"Email to {recipient} with subject '{subject}' sent successfully."
        except Exception as e:
            return f"Failed to send email: {e}"

# Example usage:
# EmailUtil.send_email(
#     smtp_server="smtp.example.com",
#     port=587,
#     login="your_email@example.com",
#     password="your_password",
#     recipient="recipient@example.com",
#     subject="Test Email",
#     body="This is a test email."
# )
