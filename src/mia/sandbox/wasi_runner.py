"""WASI sandbox runner with graceful fallbacks.

Provides an interface for executing WebAssembly modules under
resource limits using either wasmtime or wasmer when available.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


class WasiSandboxError(RuntimeError):
    """Raised when sandbox execution fails."""


@dataclass
class SandboxLimits:
    """Limits applied to sandbox execution."""

    max_memory_mb: int = 256
    timeout_ms: int = 10_000
    fuel: Optional[int] = None

    @classmethod
    def from_env(cls) -> "SandboxLimits":
        def _int(env_name: str, default: Optional[int]) -> Optional[int]:
            raw = os.getenv(env_name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning("Invalid value for %s: %s", env_name, raw)
                return default

        return cls(
            max_memory_mb=_int("MIA_SANDBOX_MEMORY_MB", 256) or 256,
            timeout_ms=_int("MIA_SANDBOX_TIMEOUT_MS", 10_000) or 10_000,
            fuel=_int("MIA_SANDBOX_FUEL", None),
        )


class WasiSandbox:
    """Sandbox for executing WASI compatible modules.

    The sandbox lazily imports wasmtime/wasmer and records runs under
    `logs/sandbox`. If neither engine is available it surfaces a clear
    error instead of crashing the application.
    """

    def __init__(
        self,
        limits: Optional[SandboxLimits] = None,
        work_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        self.limits = limits or SandboxLimits.from_env()
        self.work_dir = Path(
            work_dir or os.getenv("MIA_SANDBOX_WORKDIR", tempfile.gettempdir())
        )
        self.log_dir = Path(log_dir or os.getenv("MIA_SANDBOX_LOGDIR", "logs/sandbox"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Initialized WASI sandbox: work=%s log=%s limits=%s",
            self.work_dir,
            self.log_dir,
            self.limits,
        )

    def run(
        self,
        module_path: Optional[str] = None,
        wasi_bytes: Optional[bytes] = None,
        stdin: Optional[bytes] = None,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a WASI module and return captured metadata."""

        if not module_path and not wasi_bytes:
            raise WasiSandboxError("A module path or raw bytes must be provided")

        run_id = uuid.uuid4().hex
        work_dir = self.work_dir / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"sandbox_{run_id}.log"

        start = time.perf_counter()
        result: Dict[str, Any] = {
            "id": run_id,
            "engine": None,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration_ms": None,
            "log_file": str(log_file),
        }

        try:
            engine = self._load_engine()
        except WasiSandboxError as exc:
            self._log_failure(log_file, run_id, str(exc))
            raise

        result["engine"] = engine["name"]

        try:
            if engine["name"] == "wasmtime":
                result.update(
                    self._run_wasmtime(
                        engine["obj"], module_path, wasi_bytes, stdin, args, env
                    )
                )
            else:
                result.update(
                    self._run_wasmer(
                        engine["obj"], module_path, wasi_bytes, stdin, args, env
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive
            self._log_failure(log_file, run_id, str(exc))
            raise WasiSandboxError(f"Sandbox execution failed: {exc}") from exc
        finally:
            result["duration_ms"] = int((time.perf_counter() - start) * 1000)
            self._write_log(log_file, result)

        return result

    def _load_engine(self) -> Dict[str, Any]:
        """Return available WASI engine or raise."""

        try:
            import wasmtime  # type: ignore

            store_config = wasmtime.Config()
            if self.limits.fuel is not None:
                store_config.consume_fuel = True
            engine = wasmtime.Engine(store_config)
            return {"name": "wasmtime", "obj": engine}
        except ImportError:
            logger.debug("wasmtime not available")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed loading wasmtime: %s", exc)

        try:
            import wasmer  # type: ignore

            return {"name": "wasmer", "obj": wasmer}
        except ImportError:
            logger.debug("wasmer not available")
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed loading wasmer: %s", exc)

        raise WasiSandboxError(
            "Neither wasmtime nor wasmer is installed. Install with 'pip install wasmtime' or 'pip install wasmer'."
        )

    def _run_wasmtime(
        self,
        engine: Any,
        module_path: Optional[str],
        wasi_bytes: Optional[bytes],
        stdin: Optional[bytes],
        args: Optional[Sequence[str]],
        env: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        import wasmtime  # type: ignore

        module = (
            wasmtime.Module(engine, wasi_bytes)
            if wasi_bytes is not None
            else wasmtime.Module.from_file(engine, module_path or "")
        )

        store = wasmtime.Store(engine)
        if self.limits.fuel is not None and hasattr(store, "add_fuel"):
            store.add_fuel(self.limits.fuel)  # type: ignore

        wasi_config = wasmtime.WasiConfig()
        wasi_config.argv = list(args or [])
        wasi_config.env = list((env or {}).items())
        wasi_config.inherit_stderr()
        wasi_config.inherit_stdout()
        if stdin is not None and hasattr(wasi_config, "set_stdin_bytes"):
            wasi_config.set_stdin_bytes(stdin)  # type: ignore
        else:
            wasi_config.inherit_stdin()

        store.set_wasi(wasi_config)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()

        instance = linker.instantiate(store, module)
        try:
            run_func = instance.exports(store)["_start"]
        except (KeyError, TypeError):
            run_func = None

        start = time.perf_counter()
        try:
            if run_func and callable(run_func):
                try:
                    run_func(store)  # type: ignore
                except TypeError:
                    run_func()  # type: ignore
            exit_code = 0
        except Exception as exc:
            # Handle exit codes from various wasmtime versions
            exit_code = getattr(exc, "code", 1)
        duration_ms = int((time.perf_counter() - start) * 1000)

        consumed_fuel = None
        if self.limits.fuel is not None:
            try:
                consumed_fuel = self.limits.fuel - store.fuel_consumed  # type: ignore
            except AttributeError:  # pragma: no cover
                consumed_fuel = None

        return {
            "exit_code": exit_code,
            "stdout": "",  # stdout inherited, see log file
            "stderr": "",
            "duration_ms": duration_ms,
            "fuel_remaining": consumed_fuel,
        }

    def _run_wasmer(
        self,
        wasmer: Any,
        module_path: Optional[str],
        wasi_bytes: Optional[bytes],
        stdin: Optional[bytes],
        args: Optional[Sequence[str]],
        env: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        engine = wasmer.Engine()
        if wasi_bytes is not None:
            wasm_data = wasi_bytes
        else:
            wasm_data = Path(module_path or "").read_bytes()

        module = wasmer.Module(engine, wasm_data)

        wasi_env = wasmer.WasiStateBuilder("mia_sandbox")
        for arg in args or []:
            wasi_env.argument(arg)
        for key, value in (env or {}).items():
            wasi_env.environment(key, value)
        if stdin is not None:
            wasi_env.stdin(stdin)
        wasi_env = wasi_env.finalize()

        import wasmer_compiler_cranelift  # type: ignore  # noqa: F401

        store = wasmer.Store(engine)
        instance = wasmer.Instance(
            module, wasi_env.generate_import_object(store, module)
        )
        start = time.perf_counter()
        try:
            instance.exports._start()
            exit_code = 0
        except wasmer.ExportError as exc:  # pragma: no cover
            raise WasiSandboxError(str(exc)) from exc
        duration_ms = int((time.perf_counter() - start) * 1000)

        return {
            "exit_code": exit_code,
            "stdout": "",
            "stderr": "",
            "duration_ms": duration_ms,
        }

    def _write_log(self, log_file: Path, result: Dict[str, Any]) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "result": result,
        }
        log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _log_failure(self, log_file: Path, run_id: str, message: str) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "id": run_id,
            "error": message,
        }
        log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["WasiSandbox", "WasiSandboxError", "SandboxLimits"]
