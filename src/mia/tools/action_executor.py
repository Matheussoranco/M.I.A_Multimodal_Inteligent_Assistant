"""
Action Executor: Handles execution of external APIs, device control, system commands, web automation, email, messaging, smart home, with permission checks, error handling, and logging.
"""
import os
import subprocess
import shutil
import smtplib
from email.message import EmailMessage
# For web automation, import Selenium if available
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
except ImportError:
    webdriver = None
# For clipboard, notifications, system settings (stubs)

class ActionExecutor:
    def __init__(self, permissions=None, logger=None, consent_callback=None):
        self.permissions = permissions or {}
        self.logger = logger or (lambda msg: print(f"[LOG] {msg}"))
        self.consent_callback = consent_callback or (lambda action: True)

    def execute(self, action, params):
        if not self.permissions.get(action, False):
            self.logger(f"Permission denied for action: {action}")
            raise PermissionError(f"Action '{action}' is not permitted.")
        if not self.consent_callback(action):
            self.logger(f"User denied consent for action: {action}")
            return "User denied consent."
        try:
            if action == "open_file":
                return self.open_file(params.get("path"))
            elif action == "move_file":
                return self.move_file(params.get("src"), params.get("dst"))
            elif action == "delete_file":
                return self.delete_file(params.get("path"))
            elif action == "search_file":
                return self.search_file(params.get("name"), params.get("directory", "."))
            elif action == "launch_app":
                return self.launch_app(params.get("app"))
            elif action == "close_app":
                return self.close_app(params.get("app"))
            elif action == "clipboard":
                return self.clipboard_action(params)
            elif action == "notify":
                return self.notify(params.get("message"))
            elif action == "system_setting":
                return self.system_setting(params)
            elif action == "run_command":
                return self.run_command(params.get("command"))
            elif action == "web_automation":
                return self.web_automation(params)
            elif action == "send_email":
                return self.send_email(params)
            elif action == "calendar_event":
                return self.calendar_event(params)
            elif action == "send_message":
                return self.send_message(params)
            elif action == "smart_home":
                return self.smart_home(params)
            else:
                self.logger(f"Unknown action: {action}")
                return f"Unknown action: {action}"
        except Exception as e:
            self.logger(f"Error executing {action}: {e}")
            return f"Error: {e}"

    # OS File Management
    def open_file(self, path):
        if not path:
            return "No file path provided."
        os.startfile(path)
        return f"Opened file: {path}"

    def move_file(self, src, dst):
        if not src or not dst:
            return "Source or destination missing."
        shutil.move(src, dst)
        return f"Moved {src} to {dst}"

    def delete_file(self, path):
        if not path:
            return "No file path provided."
        os.remove(path)
        return f"Deleted file: {path}"

    def search_file(self, name, directory="."):
        matches = []
        for root, dirs, files in os.walk(directory):
            if name in files:
                matches.append(os.path.join(root, name))
        return matches or f"No file named {name} found."

    # Application Control
    def launch_app(self, app):
        if not app:
            return "No application specified."
        subprocess.Popen(app)
        return f"Launched application: {app}"

    def close_app(self, app):
        # Stub: Implement with psutil or taskkill
        return f"Closed application: {app} (stub)"

    # Clipboard
    def clipboard_action(self, params):
        # Stub: Use pyperclip or tkinter for clipboard
        return "Clipboard action performed (stub)."

    # Notifications
    def notify(self, message):
        # Stub: Use plyer or win10toast for notifications
        return f"Notification: {message} (stub)"

    # System Settings
    def system_setting(self, params):
        # Stub: Implement system settings changes
        return "System setting changed (stub)."

    def run_command(self, command):
        if not command:
            return "No command provided."
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr

    # Web Automation (Selenium)
    def web_automation(self, params):
        if webdriver is None:
            return "Selenium not installed. Run: pip install selenium."
        url = params.get("url")
        if not url:
            return "No URL provided."
        driver = webdriver.Chrome()
        driver.get(url)
        # Add more automation as needed
        driver.quit()
        return f"Web automation on {url} complete."

    # Email (SMTP)
    def send_email(self, params):
        to = params.get("to")
        subject = params.get("subject", "(No Subject)")
        body = params.get("body", "")
        if not to:
            return "No recipient provided."
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = params.get("from", "your@email.com")
        msg["To"] = to
        msg.set_content(body)
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.starttls()
                smtp.login(params.get("from", "your@email.com"), params.get("password", ""))
                smtp.send_message(msg)
            return f"Email sent to {to}"
        except Exception as e:
            return f"Email error: {e}"

    # Calendar (Google Calendar API stub)
    def calendar_event(self, params):
        # Stub: Integrate Google Calendar API
        return "Calendar event created (stub). See Google Calendar API for real integration."

    # Messaging (WhatsApp/Telegram stub)
    def send_message(self, params):
        # Stub: Integrate WhatsApp/Telegram API
        return "Message sent (stub). See WhatsApp/Telegram API for real integration."

    # Smart Home (Home Assistant stub)
    def smart_home(self, params):
        # Stub: Integrate Home Assistant API
        return "Smart home action executed (stub). See Home Assistant API for real integration."

    def set_permission(self, action, allowed):
        self.permissions[action] = allowed
