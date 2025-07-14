"""
Action Executor: Comprehensive tool for executing external APIs, device control, system commands, 
web automation, email, messaging, smart home, file operations, research, and more.
"""
import os
import subprocess
import shutil
import smtplib
import json
import csv
import requests
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from email.message import EmailMessage

# Optional imports with fallbacks
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import openpyxl
    from openpyxl import Workbook
    HAS_OPENPYXL = True
except ImportError:
    openpyxl = None
    Workbook = None
    HAS_OPENPYXL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    HAS_SELENIUM = True
except ImportError:
    webdriver = None
    HAS_SELENIUM = False

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    pyperclip = None
    HAS_PYPERCLIP = False

try:
    import plyer
    HAS_PLYER = True
except ImportError:
    plyer = None
    HAS_PLYER = False

try:
    import pywhatkit
    HAS_PYWHATKIT = True
except ImportError:
    pywhatkit = None
    HAS_PYWHATKIT = False

try:
    import wikipedia
    HAS_WIKIPEDIA = True
except ImportError:
    wikipedia = None
    HAS_WIKIPEDIA = False

class ActionExecutor:
    def __init__(self, permissions=None, logger=None, consent_callback=None):
        self.permissions = permissions or {}
        self.logger = logger or logging.getLogger(__name__)
        self.consent_callback = consent_callback or (lambda action: True)
        self.notes_file = "mia_notes.md"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config file or environment variables."""
        config = {
            "email": {
                "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("EMAIL_USERNAME", ""),
                "password": os.getenv("EMAIL_PASSWORD", "")
            },
            "research": {
                "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
                "google_cse_id": os.getenv("GOOGLE_CSE_ID", ""),
                "default_search_engine": "duckduckgo"
            },
            "smart_home": {
                "home_assistant_url": os.getenv("HOME_ASSISTANT_URL", ""),
                "home_assistant_token": os.getenv("HOME_ASSISTANT_TOKEN", "")
            },
            "telegram": {
                "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "")
            },
            "whatsapp": {
                "phone_number": os.getenv("WHATSAPP_PHONE", "")
            }
        }
        return config

    def execute(self, action, params):
        """Execute an action with given parameters."""
        if not action:
            return "No action provided."
        
        self.logger.info(f"Executing action: {action}")
        
        # Check permissions (allow all actions by default unless specifically restricted)
        if self.permissions and action in self.permissions and not self.permissions[action]:
            if not self.consent_callback(action):
                self.logger.warning(f"Permission denied for action: {action}")
                return f"Permission denied for action: {action}"
        
        try:
            # File operations
            if action == "open_file":
                return self.open_file(params.get("path"))
            elif action == "create_file":
                return self.create_file(params.get("path"), params.get("content", ""))
            elif action == "read_file":
                return self.read_file(params.get("path"))
            elif action == "write_file":
                return self.write_file(params.get("path"), params.get("content"))
            elif action == "move_file":
                return self.move_file(params.get("src"), params.get("dst"))
            elif action == "delete_file":
                return self.delete_file(params.get("path"))
            elif action == "search_file":
                return self.search_file(params.get("name"), params.get("directory", "."))
            elif action == "open_directory":
                return self.open_directory(params.get("path"))
            elif action == "create_directory":
                return self.create_directory(params.get("path"))
            
            # Code generation
            elif action == "create_code":
                return self.create_code(params.get("language"), params.get("description"), params.get("filename"))
            elif action == "analyze_code":
                return self.analyze_code(params.get("path"))
            
            # Notes and documentation
            elif action == "make_note":
                return self.make_note(params.get("content"), params.get("title"))
            elif action == "read_notes":
                return self.read_notes()
            elif action == "search_notes":
                return self.search_notes(params.get("query"))
            
            # Spreadsheet operations
            elif action == "create_sheet":
                return self.create_sheet(params.get("filename"), params.get("data"))
            elif action == "read_sheet":
                return self.read_sheet(params.get("filename"))
            elif action == "write_sheet":
                return self.write_sheet(params.get("filename"), params.get("data"))
            
            # Research and web operations
            elif action == "web_search":
                return self.web_search(params.get("query"))
            elif action == "web_scrape":
                return self.web_scrape(params.get("url"))
            elif action == "research_topic":
                return self.research_topic(params.get("topic"))
            elif action == "wikipedia_search":
                return self.wikipedia_search(params.get("query"))
            
            # Smart home integration
            elif action == "control_device":
                device_type = params.get("device_type")
                device_action = params.get("action")
                # Remove these from params to avoid duplicate keyword arguments
                filtered_params = {k: v for k, v in params.items() if k not in ["device_type", "action"]}
                return self.control_device(device_type, device_action, **filtered_params)
            
            # System integration
            elif action == "clipboard_copy":
                return self.clipboard_copy(params.get("text"))
            elif action == "clipboard_paste":
                return self.clipboard_paste()
            elif action == "show_notification":
                return self.show_notification(params.get("title"), params.get("message"))
            elif action == "open_application":
                return self.open_application(params.get("app_name"))
            elif action == "get_system_info":
                return self.get_system_info()
            
            # Application control
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
            
            # Communication
            elif action == "send_email":
                return self.send_email(params)
            elif action == "send_whatsapp":
                return self.send_whatsapp(params)
            elif action == "send_message":
                return self.send_message(params)
            
            # Calendar and scheduling
            elif action == "calendar_event":
                return self.calendar_event(params)
            
            # Smart home
            elif action == "smart_home":
                return self.smart_home(params)
            elif action == "control_lights":
                return self.control_lights(params)
            elif action == "control_temperature":
                return self.control_temperature(params)
            
            else:
                self.logger.error(f"Unknown action: {action}")
                return f"Unknown action: {action}"
        except Exception as e:
            self.logger.error(f"Error executing {action}: {e}")
            return f"Error: {e}"

    # Enhanced File Operations
    def create_file(self, path: str, content: str = "") -> str:
        """Create a new file with specified content."""
        if not path:
            return "No file path provided."
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Created file: {path}"
        except Exception as e:
            return f"Error creating file: {e}"

    def read_file(self, path: str) -> str:
        """Read content from a file."""
        if not path:
            return "No file path provided."
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"Content of {path}:\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file."""
        if not path:
            return "No file path provided."
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Written to file: {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def open_directory(self, path: str) -> str:
        """Open a directory in the file explorer."""
        if not path:
            return "No directory path provided."
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.run(['open', path] if sys.platform == 'darwin' else ['xdg-open', path])
            return f"Opened directory: {path}"
        except Exception as e:
            return f"Error opening directory: {e}"

    def create_directory(self, path: str) -> str:
        """Create a new directory."""
        if not path:
            return "No directory path provided."
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return f"Created directory: {path}"
        except Exception as e:
            return f"Error creating directory: {e}"

    # Code Generation and Analysis
    def create_code(self, language: str, description: str, filename: str = None) -> str:
        """Generate code based on description and language."""
        if not language or not description:
            return "Language and description required."
        
        # Basic code templates
        templates = {
            "python": f"""# {description}
# Generated by M.I.A

def main():
    \"\"\"
    {description}
    \"\"\"
    print("Hello, World!")
    # Add your code here
    pass

if __name__ == "__main__":
    main()
""",
            "javascript": f"""// {description}
// Generated by M.I.A

function main() {{
    console.log("Hello, World!");
    // Add your code here
}}

main();
""",
            "java": f"""// {description}
// Generated by M.I.A

public class Main {{
    public static void main(String[] args) {{
        System.out.println("Hello, World!");
        // Add your code here
    }}
}}
""",
            "html": f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{description}</title>
</head>
<body>
    <h1>{description}</h1>
    <!-- Add your HTML here -->
</body>
</html>
""",
            "css": f"""/* {description} */
/* Generated by M.I.A */

body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
}}

/* Add your CSS here */
"""
        }
        
        code = templates.get(language.lower(), f"// {description}\n// Code template not available for {language}")
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                return f"Code created and saved to {filename}"
            except Exception as e:
                return f"Error saving code: {e}"
        
        return f"Generated {language} code:\n{code}"

    def analyze_code(self, path: str) -> str:
        """Analyze code file and provide insights."""
        if not path:
            return "No file path provided."
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            lines = code.split('\n')
            analysis = {
                "total_lines": len(lines),
                "non_empty_lines": len([line for line in lines if line.strip()]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')]),
                "file_size": os.path.getsize(path),
                "language": self._detect_language(path)
            }
            
            return f"Code analysis for {path}:\n{json.dumps(analysis, indent=2)}"
        except Exception as e:
            return f"Error analyzing code: {e}"

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        extension = Path(path).suffix.lower()
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.html': 'HTML',
            '.css': 'CSS',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        return lang_map.get(extension, 'Unknown')

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

    # Notes and Documentation
    def make_note(self, content: str, title: str = None) -> str:
        """Create or append to notes file."""
        if not content:
            return "No content provided for note."
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            note_entry = f"\n## {title or 'Note'} - {timestamp}\n\n{content}\n\n---\n"
            
            with open(self.notes_file, 'a', encoding='utf-8') as f:
                f.write(note_entry)
            
            return f"Note saved to {self.notes_file}"
        except Exception as e:
            return f"Error saving note: {e}"

    def read_notes(self) -> str:
        """Read all notes from the notes file."""
        try:
            if not os.path.exists(self.notes_file):
                return "No notes file found."
            
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = f.read()
            
            return f"Notes from {self.notes_file}:\n{notes}"
        except Exception as e:
            return f"Error reading notes: {e}"

    def search_notes(self, query: str) -> str:
        """Search for specific content in notes."""
        if not query:
            return "No search query provided."
        
        try:
            if not os.path.exists(self.notes_file):
                return "No notes file found."
            
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = f.read()
            
            matching_lines = []
            for i, line in enumerate(notes.split('\n'), 1):
                if query.lower() in line.lower():
                    matching_lines.append(f"Line {i}: {line}")
            
            if matching_lines:
                return f"Found {len(matching_lines)} matches for '{query}':\n" + "\n".join(matching_lines)
            else:
                return f"No matches found for '{query}'"
        except Exception as e:
            return f"Error searching notes: {e}"

    # Spreadsheet Operations
    def create_sheet(self, filename: str, data: List[List[str]] = None) -> str:
        """Create a new spreadsheet file."""
        if not filename:
            return "No filename provided."
        
        try:
            if filename.endswith('.xlsx'):
                return self._create_excel_sheet(filename, data)
            elif filename.endswith('.csv'):
                return self._create_csv_sheet(filename, data)
            else:
                return "Unsupported file format. Use .xlsx or .csv"
        except Exception as e:
            return f"Error creating sheet: {e}"

    def _create_excel_sheet(self, filename: str, data: List[List[str]] = None) -> str:
        """Create Excel spreadsheet."""
        if not HAS_OPENPYXL:
            return "openpyxl not installed. Run: pip install openpyxl"
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        
        if data:
            for row in data:
                ws.append(row)
        else:
            ws.append(["Column1", "Column2", "Column3"])
            ws.append(["Sample", "Data", "Here"])
        
        wb.save(filename)
        return f"Excel sheet created: {filename}"

    def _create_csv_sheet(self, filename: str, data: List[List[str]] = None) -> str:
        """Create CSV file."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if data:
                writer.writerows(data)
            else:
                writer.writerow(["Column1", "Column2", "Column3"])
                writer.writerow(["Sample", "Data", "Here"])
        
        return f"CSV sheet created: {filename}"

    def read_sheet(self, filename: str) -> str:
        """Read data from spreadsheet file."""
        if not filename:
            return "No filename provided."
        
        try:
            if filename.endswith('.xlsx'):
                return self._read_excel_sheet(filename)
            elif filename.endswith('.csv'):
                return self._read_csv_sheet(filename)
            else:
                return "Unsupported file format. Use .xlsx or .csv"
        except Exception as e:
            return f"Error reading sheet: {e}"

    def _read_excel_sheet(self, filename: str) -> str:
        """Read Excel spreadsheet."""
        if not HAS_OPENPYXL:
            return "openpyxl not installed. Run: pip install openpyxl"
        
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        
        data = []
        for row in ws.iter_rows(values_only=True):
            data.append(row)
        
        return f"Excel sheet data from {filename}:\n{json.dumps(data, indent=2, default=str)}"

    def _read_csv_sheet(self, filename: str) -> str:
        """Read CSV file."""
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        
        return f"CSV sheet data from {filename}:\n{json.dumps(data, indent=2)}"

    def write_sheet(self, filename: str, data: List[List[str]]) -> str:
        """Write data to spreadsheet file."""
        if not filename or not data:
            return "Filename and data required."
        
        try:
            if filename.endswith('.xlsx'):
                return self._write_excel_sheet(filename, data)
            elif filename.endswith('.csv'):
                return self._write_csv_sheet(filename, data)
            else:
                return "Unsupported file format. Use .xlsx or .csv"
        except Exception as e:
            return f"Error writing sheet: {e}"

    def _write_excel_sheet(self, filename: str, data: List[List[str]]) -> str:
        """Write to Excel spreadsheet."""
        if not HAS_OPENPYXL:
            return "openpyxl not installed. Run: pip install openpyxl"
        
        wb = Workbook()
        ws = wb.active
        
        for row in data:
            ws.append(row)
        
        wb.save(filename)
        return f"Data written to Excel sheet: {filename}"

    def _write_csv_sheet(self, filename: str, data: List[List[str]]) -> str:
        """Write to CSV file."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        return f"Data written to CSV sheet: {filename}"

    # Application Control
    def launch_app(self, app):
        if not app:
            return "No application specified."
        subprocess.Popen(app)
        return f"Launched application: {app}"

    def close_app(self, app):
        """Close application by name."""
        if not app:
            return "No application name provided."
        
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', f'{app}.exe'], 
                             capture_output=True, text=True)
                return f"Attempting to close {app}"
            else:  # Unix-like systems
                subprocess.run(['pkill', '-f', app], capture_output=True, text=True)
                return f"Attempting to close {app}"
        except Exception as e:
            return f"Error closing app: {e}"

    # Clipboard
    def clipboard_action(self, params):
        """Handle clipboard operations."""
        action = params.get("action", "")
        
        if action == "copy":
            return self.clipboard_copy(params.get("text", ""))
        elif action == "paste":
            return self.clipboard_paste()
        else:
            return "Unknown clipboard action. Use 'copy' or 'paste'."

    # Notifications
    def notify(self, message):
        """Send notification."""
        if not message:
            return "No message provided."
        
        return self.show_notification("M.I.A", message)

    # System Settings
    def system_setting(self, params):
        """Change system settings."""
        setting = params.get("setting", "")
        value = params.get("value", "")
        
        if not setting:
            return "No setting specified."
        
        try:
            if setting == "volume":
                return self._set_volume(value)
            elif setting == "brightness":
                return self._set_brightness(value)
            elif setting == "wifi":
                return self._control_wifi(value)
            else:
                return f"Setting '{setting}' not supported yet."
        except Exception as e:
            return f"Error changing setting: {e}"

    def _set_volume(self, value):
        """Set system volume."""
        try:
            if os.name == 'nt':
                # Windows volume control
                subprocess.run(['nircmd', 'setsysvolume', str(int(value) * 655.35)], 
                             capture_output=True)
                return f"Volume set to {value}%"
            else:
                # Linux volume control
                subprocess.run(['amixer', 'set', 'Master', f'{value}%'], 
                             capture_output=True)
                return f"Volume set to {value}%"
        except:
            return "Volume control not available on this system."

    def _set_brightness(self, value):
        """Set screen brightness."""
        try:
            if os.name == 'nt':
                # Windows brightness control (requires powershell)
                cmd = f'(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{value})'
                subprocess.run(['powershell', '-Command', cmd], capture_output=True)
                return f"Brightness set to {value}%"
            else:
                # Linux brightness control
                subprocess.run(['xbacklight', '-set', str(value)], capture_output=True)
                return f"Brightness set to {value}%"
        except:
            return "Brightness control not available on this system."

    def _control_wifi(self, action):
        """Control WiFi connection."""
        try:
            if os.name == 'nt':
                if action == "on":
                    subprocess.run(['netsh', 'interface', 'set', 'interface', 'Wi-Fi', 'enabled'], 
                                 capture_output=True)
                    return "WiFi enabled"
                elif action == "off":
                    subprocess.run(['netsh', 'interface', 'set', 'interface', 'Wi-Fi', 'disabled'], 
                                 capture_output=True)
                    return "WiFi disabled"
            else:
                if action == "on":
                    subprocess.run(['nmcli', 'radio', 'wifi', 'on'], capture_output=True)
                    return "WiFi enabled"
                elif action == "off":
                    subprocess.run(['nmcli', 'radio', 'wifi', 'off'], capture_output=True)
                    return "WiFi disabled"
        except:
            return "WiFi control not available on this system."

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
        """Create calendar event."""
        title = params.get("title", "")
        date = params.get("date", "")
        time = params.get("time", "")
        
        if not title:
            return "No event title provided."
        
        # For now, create a simple text file for calendar events
        try:
            event_info = f"Event: {title}\nDate: {date}\nTime: {time}\n\n"
            with open("calendar_events.txt", "a") as f:
                f.write(event_info)
            return f"Calendar event '{title}' created for {date} at {time}"
        except Exception as e:
            return f"Error creating calendar event: {e}"

    # Messaging (WhatsApp/Telegram stub)
    def send_message(self, params):
        """Send message via various platforms."""
        platform = params.get("platform", "").lower()
        to = params.get("to", "")
        message = params.get("message", "")
        
        if not to or not message:
            return "Recipient and message required."
        
        try:
            if platform == "whatsapp":
                return self.send_whatsapp(params)
            elif platform == "telegram":
                return self._send_telegram(to, message)
            elif platform == "sms":
                return self._send_sms(to, message)
            else:
                return f"Platform '{platform}' not supported. Use 'whatsapp', 'telegram', or 'sms'."
        except Exception as e:
            return f"Error sending message: {e}"

    def _send_telegram(self, to, message):
        """Send Telegram message."""
        bot_token = self.config.get("telegram", {}).get("bot_token", "")
        if not bot_token:
            return "Telegram bot token not configured."
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": to, "text": message}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return f"Telegram message sent to {to}"
            else:
                return f"Telegram error: {response.text}"
        except Exception as e:
            return f"Error sending Telegram message: {e}"

    def _send_sms(self, to, message):
        """Send SMS message."""
        # This would typically use a service like Twilio
        return f"SMS functionality requires Twilio or similar service integration. Message: {message} to {to}"

    # Smart Home (Home Assistant stub)
    def smart_home(self, params):
        """Control smart home devices via Home Assistant."""
        device = params.get("device", "")
        action = params.get("action", "")
        
        if not device or not action:
            return "Device and action required."
        
        ha_url = self.config.get("smart_home", {}).get("home_assistant_url", "")
        ha_token = self.config.get("smart_home", {}).get("home_assistant_token", "")
        
        if not ha_url or not ha_token:
            return "Home Assistant URL and token not configured."
        
        try:
            headers = {"Authorization": f"Bearer {ha_token}"}
            url = f"{ha_url}/api/services/homeassistant/turn_{action}"
            data = {"entity_id": device}
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                return f"Smart home action: {action} on {device}"
            else:
                return f"Home Assistant error: {response.text}"
        except Exception as e:
            return f"Error controlling smart home: {e}"

    # Smart Home Integration
    def control_device(self, device_type: str, action: str, **kwargs) -> str:
        """Control smart home devices."""
        if not device_type or not action:
            return "Device type and action required."
        
        try:
            device_type = device_type.lower()
            action = action.lower()
            
            if device_type == "light":
                return self._control_light(action, **kwargs)
            elif device_type == "temperature":
                return self._control_temperature(action, **kwargs)
            elif device_type == "music":
                return self._control_music(action, **kwargs)
            elif device_type == "security":
                return self._control_security(action, **kwargs)
            else:
                return self._generic_device_control(device_type, action, **kwargs)
        except Exception as e:
            return f"Error controlling device: {e}"

    def _control_light(self, action: str, **kwargs) -> str:
        """Control lighting system."""
        room = kwargs.get('room', 'living room')
        brightness = kwargs.get('brightness', 100)
        color = kwargs.get('color', 'white')
        
        if action == "on":
            return f"Turning on {room} lights at {brightness}% brightness"
        elif action == "off":
            return f"Turning off {room} lights"
        elif action == "dim":
            return f"Dimming {room} lights to {brightness}%"
        elif action == "color":
            return f"Setting {room} lights to {color}"
        else:
            return f"Unknown light action: {action}"

    def _control_temperature(self, action: str, **kwargs) -> str:
        """Control temperature system."""
        temperature = kwargs.get('temperature', 22)
        mode = kwargs.get('mode', 'auto')
        
        if action == "set":
            return f"Setting temperature to {temperature}°C in {mode} mode"
        elif action == "heat":
            return f"Setting heating to {temperature}°C"
        elif action == "cool":
            return f"Setting cooling to {temperature}°C"
        elif action == "auto":
            return f"Setting thermostat to auto mode at {temperature}°C"
        else:
            return f"Unknown temperature action: {action}"

    def _control_music(self, action: str, **kwargs) -> str:
        """Control music system."""
        volume = kwargs.get('volume', 50)
        song = kwargs.get('song', '')
        
        if action == "play":
            return f"Playing music{' - ' + song if song else ''} at {volume}% volume"
        elif action == "pause":
            return "Pausing music"
        elif action == "stop":
            return "Stopping music"
        elif action == "volume":
            return f"Setting music volume to {volume}%"
        else:
            return f"Unknown music action: {action}"

    def _control_security(self, action: str, **kwargs) -> str:
        """Control security system."""
        zone = kwargs.get('zone', 'all')
        
        if action == "arm":
            return f"Arming security system for {zone}"
        elif action == "disarm":
            return f"Disarming security system for {zone}"
        elif action == "status":
            return f"Security system status for {zone}: Armed"
        else:
            return f"Unknown security action: {action}"

    def _generic_device_control(self, device_type: str, action: str, **kwargs) -> str:
        """Handle generic device control."""
        return f"Controlling {device_type}: {action} with parameters {kwargs}"

    # System Integration
    def clipboard_copy(self, text: str) -> str:
        """Copy text to clipboard."""
        if not text:
            return "No text provided to copy."
        
        try:
            if HAS_PYPERCLIP:
                pyperclip.copy(text)
                return f"Text copied to clipboard: {text[:50]}..."
            else:
                return "pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"Error copying to clipboard: {e}"

    def clipboard_paste(self) -> str:
        """Get text from clipboard."""
        try:
            if HAS_PYPERCLIP:
                text = pyperclip.paste()
                return f"Clipboard content: {text}"
            else:
                return "pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"Error reading clipboard: {e}"

    def show_notification(self, title: str, message: str) -> str:
        """Show system notification."""
        if not title or not message:
            return "Title and message required for notification."
        
        try:
            if HAS_PLYER:
                plyer.notification.notify(
                    title=title,
                    message=message,
                    app_name="M.I.A",
                    timeout=10
                )
                return f"Notification sent: {title} - {message}"
            else:
                return "plyer not installed. Run: pip install plyer"
        except Exception as e:
            return f"Error showing notification: {e}"

    def open_application(self, app_name: str) -> str:
        """Open an application."""
        if not app_name:
            return "No application name provided."
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(app_name)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['open', app_name] if sys.platform == 'darwin' else ['xdg-open', app_name])
            
            return f"Opening application: {app_name}"
        except Exception as e:
            return f"Error opening application: {e}"

    def get_system_info(self) -> str:
        """Get system information."""
        try:
            import platform
            
            info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
            
            if HAS_PSUTIL:
                info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
                    "disk_usage": f"{psutil.disk_usage('/').percent}%"
                })
            else:
                info["note"] = "psutil not installed. Run: pip install psutil for detailed system info"
            
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error getting system info: {e}"

    def set_permission(self, action, allowed):
        self.permissions[action] = allowed

    # Research and web operations
    def web_search(self, query: str) -> str:
        """Search the web for information."""
        if not query:
            return "No search query provided."
        
        try:
            # Use DuckDuckGo as default search engine
            search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
            response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code == 200:
                return f"Web search completed for: {query}. Found results at {search_url}"
            else:
                return f"Web search failed with status code: {response.status_code}"
        except Exception as e:
            return f"Error performing web search: {e}"

    def web_scrape(self, url: str) -> str:
        """Scrape content from a web page."""
        if not url:
            return "No URL provided."
        
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                return f"Web scraping completed for: {url}. Content length: {len(response.text)} characters"
            else:
                return f"Web scraping failed with status code: {response.status_code}"
        except Exception as e:
            return f"Error scraping web page: {e}"

    def research_topic(self, topic: str) -> str:
        """Research a topic using multiple sources."""
        if not topic:
            return "No topic provided."
        
        try:
            results = []
            
            # Web search
            web_result = self.web_search(topic)
            results.append(f"Web Search: {web_result}")
            
            # Wikipedia search
            wiki_result = self.wikipedia_search(topic)
            results.append(f"Wikipedia: {wiki_result}")
            
            return f"Research on '{topic}':\n" + "\n".join(results)
        except Exception as e:
            return f"Error researching topic: {e}"

    def wikipedia_search(self, query: str) -> str:
        """Search Wikipedia for information."""
        if not query:
            return "No search query provided."
        
        try:
            if HAS_WIKIPEDIA:
                summary = wikipedia.summary(query, sentences=3)
                return f"Wikipedia summary for '{query}': {summary}"
            else:
                return "Wikipedia library not installed. Run: pip install wikipedia"
        except Exception as e:
            return f"Error searching Wikipedia: {e}"
