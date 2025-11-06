"""Minimal API server exposing health, readiness, and streaming endpoints.

Install with extras: pip install .[api]
Run: python -m mia.api.server or mia-api
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse
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

from ..providers import ProviderLookupError, provider_registry


def create_app() -> Any:
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError(
            f"FastAPI/uvicorn not installed. Install API extras: pip install .[api].\nOriginal error: {_IMPORT_ERROR}"
        )
    app = FastAPI(title="M.I.A API", version=__version__)

    # Initialize components
    components = {}
    try:
        # Initialize with default args for API server
        import argparse

        from ..main import initialize_components

        args = argparse.Namespace(
            model_id="default",
            debug=False,
            language=None,
            image_input=None,
            mode="api",
        )
        components = initialize_components(args)
    except Exception as exc:
        # Continue without components for basic endpoints
        pass

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> Dict[str, Any]:
        # Check if core components are available
        llm_available = "llm" in components and components["llm"] is not None
        return {
            "status": "ready" if llm_available else "partial",
            "llm_available": llm_available,
        }

    @app.get("/chat/stream")
    async def chat_stream(
        prompt: str = Query(..., min_length=1)
    ) -> StreamingResponse:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        async def event_generator():
            try:
                llm = components.get("llm")
                if not llm:
                    yield f"event: error\ndata: LLM provider unavailable\n\n"
                    return
            except Exception as exc:
                yield f"event: error\ndata: Failed to get LLM: {exc}\n\n"
                return

            try:
                # Try streaming first, fallback to regular query
                if hasattr(llm, "stream") and callable(getattr(llm, "stream")):
                    async for token in llm.stream(prompt):
                        if token:
                            yield f"data: {token}\n\n"
                else:
                    response_text = await asyncio.to_thread(llm.query, prompt)
                    for token in response_text.split():
                        yield f"data: {token}\n\n"

                yield "event: done\ndata: [DONE]\n\n"
            except Exception as exc:
                yield f"event: error\ndata: LLM error: {exc}\n\n"
                return

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )

    @app.post("/chat")
    async def chat(request: Dict[str, Any]) -> Dict[str, Any]:
        """Non-streaming chat endpoint."""
        prompt = request.get("prompt", "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        try:
            llm = components.get("llm")
            if not llm:
                raise HTTPException(
                    status_code=503, detail="LLM not available"
                )

            response = await asyncio.to_thread(llm.query, prompt)
            return {"response": response, "status": "success"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    @app.get("/actions")
    async def list_actions() -> Dict[str, Any]:
        """List available actions."""
        try:
            action_executor = components.get("action_executor")
            if action_executor and hasattr(action_executor, "ACTION_SCOPES"):
                actions = list(action_executor.ACTION_SCOPES.keys())
                return {"actions": actions, "status": "success"}
            else:
                return {
                    "actions": [
                        "create_file",
                        "send_email",
                        "web_search",
                        "analyze_code",
                    ],
                    "status": "success",
                }
        except Exception as exc:
            return {"actions": [], "error": str(exc), "status": "error"}

    @app.post("/actions/{action_name}")
    async def execute_action(
        action_name: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific action."""
        try:
            action_executor = components.get("action_executor")
            if not action_executor:
                raise HTTPException(
                    status_code=503, detail="Action executor not available"
                )

            result = action_executor.execute(action_name, request)
            return {"result": result, "status": "success"}
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Action execution error: {exc}"
            )

    @app.get("/memory")
    async def get_memory(query: Optional[str] = Query(None)) -> Dict[str, Any]:
        """Get memory status and search memory."""
        try:
            if query:
                # Search memory
                action_executor = components.get("action_executor")
                if action_executor and hasattr(
                    action_executor, "search_memory"
                ):
                    result = action_executor.search_memory({"query": query})
                    return {"results": result, "status": "success"}
                else:
                    return {
                        "results": "Memory search not available",
                        "status": "error",
                    }
            else:
                # Get memory status
                return {"memory_items": [], "status": "success"}
        except Exception as exc:
            return {"error": str(exc), "status": "error"}

    @app.post("/memory")
    async def add_memory(request: Dict[str, Any]) -> Dict[str, Any]:
        """Add item to memory."""
        try:
            text = request.get("text", "").strip()
            if not text:
                raise HTTPException(status_code=400, detail="Text is required")

            action_executor = components.get("action_executor")
            if action_executor and hasattr(action_executor, "store_memory"):
                result = action_executor.store_memory(
                    {"text": text, "metadata": request.get("metadata", {})}
                )
                return {"result": result, "status": "success"}
            else:
                return {
                    "error": "Memory storage not available",
                    "status": "error",
                }
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Memory storage error: {exc}"
            )

    @app.get("/status")
    async def system_status() -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "status": "operational",
            "version": __version__,
            "llm_available": "llm" in components
            and components["llm"] is not None,
            "memory_enabled": "rag_pipeline" in components
            and components["rag_pipeline"] is not None,
            "actions_available": "action_executor" in components
            and components["action_executor"] is not None,
            "speech_enabled": "speech_generator" in components
            and components["speech_generator"] is not None,
            "web_agent_available": "web_agent" in components
            and components["web_agent"] is not None,
        }

        # Add component details
        if status["llm_available"]:
            llm = components["llm"]
            status["llm_provider"] = getattr(llm, "provider_name", "unknown")
            status["llm_streaming"] = hasattr(llm, "stream") and callable(
                getattr(llm, "stream")
            )

        if status["memory_enabled"]:
            rag = components["rag_pipeline"]
            status["rag_chunks"] = getattr(rag, "_documents", None)
            status["rag_chunks"] = (
                len(status["rag_chunks"]) if status["rag_chunks"] else 0
            )

        return status

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
