#!/usr/bin/env python3
"""
M.I.A Desktop App - Standalone native application
Uses pywebview to create a native window for the web interface
"""

import os
import sys
import threading
import time
import logging

# Set environment variables before any imports
os.environ['TESTING'] = 'true'  # Skip interactive mode

# Add src to path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = os.path.dirname(sys.executable)
    src_path = os.path.join(application_path, 'src')
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(application_path, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Suppress logging during startup
logging.basicConfig(level=logging.WARNING)


def start_server(host: str, port: int, ready_event: threading.Event):
    """Start the FastAPI server in a background thread."""
    try:
        from mia.web.webui import app  # type: ignore[import-not-found]
        import uvicorn
        
        # Signal that imports are ready
        ready_event.set()
        
        # Configure uvicorn to be quiet
        config = uvicorn.Config(
            app, 
            host=host, 
            port=port, 
            log_level="error",
            access_log=False
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        print(f"Server error: {e}")
        ready_event.set()  # Signal anyway to avoid hanging


def main():
    """Main entry point for the desktop application."""
    try:
        import webview
    except ImportError:
        print("\n" + "=" * 60)
        print("  pywebview is not installed!")
        print("=" * 60)
        print("\nTo run M.I.A as a standalone desktop app, install pywebview:")
        print("\n  pip install pywebview")
        print("\nOn Windows, you may also need:")
        print("  pip install pywebview[cef]  # For CEF backend (recommended)")
        print("  # OR")
        print("  pip install pythonnet  # For EdgeChromium backend")
        print("\n" + "=" * 60)
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    port = 8080
    host = '127.0.0.1'
    url = f'http://{host}:{port}'
    
    print("""
+------------------------------------------------------------------------------+
|                                                                              |
|     M.I.A - Multimodal Intelligent Assistant                                 |
|     Desktop Application                                                      |
|                                                                              |
|     Starting application...                                                  |
|                                                                              |
+------------------------------------------------------------------------------+
    """)
    
    # Event to signal when server is ready
    ready_event = threading.Event()
    
    # Start server in background thread
    server_thread = threading.Thread(
        target=start_server, 
        args=(host, port, ready_event),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to be ready
    print("    [*] Starting backend server...")
    ready_event.wait(timeout=10)
    
    # Give uvicorn a moment to actually start listening
    time.sleep(1)
    
    print("    [*] Opening M.I.A window...")
    print("    [i] Close the window to exit")
    print()
    
    # Create native window
    window = webview.create_window(
        title='M.I.A - Multimodal Intelligent Assistant',
        url=url,
        width=1400,
        height=900,
        min_size=(800, 600),
        resizable=True,
        frameless=False,
        easy_drag=False,
        text_select=True,
        confirm_close=False,
        background_color='#1a1a2e'
    )
    
    # Start the webview (this blocks until window is closed)
    webview.start(
        debug=False,
        http_server=False,
        private_mode=False
    )
    
    print("\n    [*] M.I.A closed. Goodbye!")


if __name__ == "__main__":
    main()
