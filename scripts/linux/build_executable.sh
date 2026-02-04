#!/usr/bin/env bash
set -euo pipefail

# Build M.I.A Web UI executable for Linux

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found"
  exit 1
fi

python3 -m pip show pyinstaller >/dev/null 2>&1 || python3 -m pip install pyinstaller

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src"

python3 -m PyInstaller --noconfirm --onefile \
  --name mia \
  --add-data "src/mia:mia" \
  --add-data "config:config" \
  --hidden-import uvicorn \
  --hidden-import fastapi \
  --hidden-import starlette \
  --hidden-import pydantic \
  --hidden-import anyio \
  mia_launcher.py

echo "Build complete. Check dist/mia"
