#!/usr/bin/env python3
"""
M.I.A Web UI Launcher - Standalone executable entry point
This file is used by PyInstaller to create a standalone .exe
"""

import os
import sys
import webbrowser
import threading
import time

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

def open_browser(port: int):
    """Open browser after a short delay."""
    time.sleep(2)
    webbrowser.open(f'http://127.0.0.1:{port}')

def main():
    """Main entry point for the executable."""
    port = 8080
    host = '127.0.0.1'
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     🧠  M.I.A - Multimodal Intelligent Assistant                            ║
║         AGI-Focused Web Interface                                            ║
║                                                                              ║
║     Starting server...                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Import after path setup
        from mia.web.webui import app, MIAAgent
        import uvicorn
        
        # Open browser in background thread
        browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
        browser_thread.start()
        
        print(f"    🌐 Opening browser at http://{host}:{port}")
        print(f"    📝 Press Ctrl+C to stop the server")
        print()
        
        # Run server
        uvicorn.run(app, host=host, port=port, log_level="warning")
        
    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies - {e}")
        print("\nPlease ensure all requirements are installed:")
        print("  pip install fastapi uvicorn")
        input("\nPress Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
