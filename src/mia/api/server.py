"""Minimal API server exposing health and readiness endpoints.

Install with extras: pip install .[api]
Run: python -m mia.api.server or mia-api
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any, Optional, List

try:
    from fastapi import FastAPI
    import uvicorn
except Exception as e:  # pragma: no cover
    # Allow import of this module without FastAPI for docs or inspection
    FastAPI = None  # type: ignore
    uvicorn = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

try:
    from ..__version__ import __version__
except Exception:
    __version__ = "0.0.0"


def create_app() -> Any:
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError(
            f"FastAPI/uvicorn not installed. Install API extras: pip install .[api].\nOriginal error: {_IMPORT_ERROR}"
        )
    app = FastAPI(title="M.I.A API", version=__version__)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> Dict[str, str]:
        # In future, verify model/LLM connectivity here
        return {"status": "ready"}

    return app


def main(argv: Optional[List[str]] = None) -> int:
    """Run the API server.

    Environment variables:
    - MIA_API_HOST (default: 0.0.0.0)
    - MIA_API_PORT (default: 8080)
    """
    print("Starting M.I.A API server...")
    if FastAPI is None or uvicorn is None:  # pragma: no cover
        sys.stderr.write(
            f"FastAPI/uvicorn not installed. Install API extras: pip install .[api].\nOriginal error: {_IMPORT_ERROR}\n"
        )
        return 1

    host = os.getenv("MIA_API_HOST", "0.0.0.0")
    port_str = os.getenv("MIA_API_PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        port = 8080

    print(f"Server will run on {host}:{port}")
    app = create_app()
    print("App created successfully")
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
