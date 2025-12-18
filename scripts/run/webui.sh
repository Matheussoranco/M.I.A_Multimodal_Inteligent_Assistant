#!/bin/bash
# M.I.A Web UI Launcher for Linux/macOS
# Starts the Ollama-style Web Interface

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║     🧠  M.I.A - Multimodal Intelligent Assistant            ║"
echo "║         Ollama-Style Web UI Launcher                         ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
    exit 1
fi

# Check for dependencies
if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
    echo "Installing required dependencies..."
    pip3 install fastapi 'uvicorn[standard]'
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Start the web UI
echo "Starting M.I.A Web UI on http://localhost:8080"
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m mia --web --port 8080
