"""Desktop automation module using pywinauto for Windows applications."""

import logging
import os
from typing import Any, Dict, List, Optional

try:
    from pywinauto import Application, Desktop
    from pywinauto.findwindows import find_windows
    from pywinauto.keyboard import send_keys
    from pywinauto.mouse import click

    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False
    Application = None
    Desktop = None
    find_windows = None
    send_keys = None
    click = None

logger = logging.getLogger(__name__)


class DesktopAutomation:
    """Windows desktop automation using pywinauto."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        self.logger = logger_instance or logging.getLogger(__name__)
        if not PYWINAUTO_AVAILABLE:
            raise RuntimeError("pywinauto not installed. Install with 'pip install pywinauto'.")

        self.desktop = Desktop(backend="uia")
        self.apps: Dict[str, Application] = {}

    def _get_app(self, app_name: str) -> Optional[Application]:
        """Get or connect to an application."""
        if app_name in self.apps:
            return self.apps[app_name]

        try:
            # Try to find running application
            windows = find_windows(title_re=f".*{app_name}.*")
            if windows:
                app = Application(backend="uia").connect(handle=windows[0])
                self.apps[app_name] = app
                return app

            # Try to start application
            app = Application(backend="uia").start(app_name)
            self.apps[app_name] = app
            return app
        except Exception as e:
            self.logger.error(f"Failed to connect/start app {app_name}: {e}")
            return None

    def open_application(self, app_path: str) -> str:
        """Open an application."""
        try:
            app_name = os.path.basename(app_path).split('.')[0]
            app = self._get_app(app_path)
            if app:
                return f"Opened application: {app_name}"
            else:
                return f"Failed to open application: {app_path}"
        except Exception as e:
            return f"Error opening application: {e}"

    def close_application(self, app_name: str) -> str:
        """Close an application."""
        try:
            if app_name in self.apps:
                self.apps[app_name].kill()
                del self.apps[app_name]
                return f"Closed application: {app_name}"
            else:
                # Find and close
                windows = find_windows(title_re=f".*{app_name}.*")
                if windows:
                    app = Application(backend="uia").connect(handle=windows[0])
                    app.kill()
                    return f"Closed application: {app_name}"
                else:
                    return f"Application not found: {app_name}"
        except Exception as e:
            return f"Error closing application: {e}"

    def type_text(self, app_name: str, text: str, window_title: Optional[str] = None) -> str:
        """Type text into an application."""
        try:
            app = self._get_app(app_name)
            if not app:
                return f"Application not found: {app_name}"

            if window_title:
                window = app.window(title_re=f".*{window_title}.*")
            else:
                window = app.top_window()

            window.type_keys(text, with_spaces=True)
            return f"Typed text into {app_name}"
        except Exception as e:
            return f"Error typing text: {e}"

    def click_element(self, app_name: str, element_name: str, window_title: Optional[str] = None) -> str:
        """Click an element in an application."""
        try:
            app = self._get_app(app_name)
            if not app:
                return f"Application not found: {app_name}"

            if window_title:
                window = app.window(title_re=f".*{window_title}.*")
            else:
                window = app.top_window()

            element = window.child_window(title=element_name)
            element.click()
            return f"Clicked {element_name} in {app_name}"
        except Exception as e:
            return f"Error clicking element: {e}"

    def send_keys_to_app(self, app_name: str, keys: str, window_title: Optional[str] = None) -> str:
        """Send keyboard keys to an application."""
        try:
            app = self._get_app(app_name)
            if not app:
                return f"Application not found: {app_name}"

            if window_title:
                window = app.window(title_re=f".*{window_title}.*")
            else:
                window = app.top_window()

            window.send_keys(keys)
            return f"Sent keys '{keys}' to {app_name}"
        except Exception as e:
            return f"Error sending keys: {e}"

    def get_window_text(self, app_name: str, window_title: Optional[str] = None) -> str:
        """Get text from a window."""
        try:
            app = self._get_app(app_name)
            if not app:
                return f"Application not found: {app_name}"

            if window_title:
                window = app.window(title_re=f".*{window_title}.*")
            else:
                window = app.top_window()

            text = window.window_text()
            return f"Window text: {text}"
        except Exception as e:
            return f"Error getting window text: {e}"

    def execute_action_schema(self, schema: Dict[str, Any]) -> str:
        """Execute a predefined action schema."""
        action_type = schema.get("type")
        app_name = schema.get("app")
        params = schema.get("params", {})

        if action_type == "open":
            return self.open_application(params.get("path", app_name))
        elif action_type == "close":
            return self.close_application(app_name)
        elif action_type == "type":
            return self.type_text(app_name, params.get("text", ""), params.get("window"))
        elif action_type == "click":
            return self.click_element(app_name, params.get("element", ""), params.get("window"))
        elif action_type == "keys":
            return self.send_keys_to_app(app_name, params.get("keys", ""), params.get("window"))
        elif action_type == "get_text":
            return self.get_window_text(app_name, params.get("window"))
        else:
            return f"Unknown action type: {action_type}"


# Predefined action schemas for common apps
ACTION_SCHEMAS = {
    "office": {
        "word": {
            "new_doc": {"type": "keys", "app": "WINWORD.EXE", "params": {"keys": "^n"}},
            "save": {"type": "keys", "app": "WINWORD.EXE", "params": {"keys": "^s"}},
            "type_text": {"type": "type", "app": "WINWORD.EXE", "params": {"text": "{text}"}},
        },
        "excel": {
            "new_sheet": {"type": "keys", "app": "EXCEL.EXE", "params": {"keys": "^n"}},
            "save": {"type": "keys", "app": "EXCEL.EXE", "params": {"keys": "^s"}},
            "select_cell": {"type": "keys", "app": "EXCEL.EXE", "params": {"keys": "{cell}"}},
        },
        "powerpoint": {
            "new_slide": {"type": "keys", "app": "POWERPNT.EXE", "params": {"keys": "^m"}},
            "save": {"type": "keys", "app": "POWERPNT.EXE", "params": {"keys": "^s"}},
        }
    },
    "browser": {
        "chrome": {
            "new_tab": {"type": "keys", "app": "chrome.exe", "params": {"keys": "^t"}},
            "close_tab": {"type": "keys", "app": "chrome.exe", "params": {"keys": "^w"}},
            "refresh": {"type": "keys", "app": "chrome.exe", "params": {"keys": "F5"}},
        },
        "firefox": {
            "new_tab": {"type": "keys", "app": "firefox.exe", "params": {"keys": "^t"}},
            "close_tab": {"type": "keys", "app": "firefox.exe", "params": {"keys": "^w"}},
            "refresh": {"type": "keys", "app": "firefox.exe", "params": {"keys": "F5"}},
        }
    },
    "file_explorer": {
        "explorer": {
            "new_folder": {"type": "keys", "app": "explorer.exe", "params": {"keys": "^+n"}},
            "select_all": {"type": "keys", "app": "explorer.exe", "params": {"keys": "^a"}},
            "copy": {"type": "keys", "app": "explorer.exe", "params": {"keys": "^c"}},
            "paste": {"type": "keys", "app": "explorer.exe", "params": {"keys": "^v"}},
        }
    }
}