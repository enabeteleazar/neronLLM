"""neron_llm/api/routes.py
API routes for neron_llm — fully async.

v2.0: POST /llm/generate added as the primary bus endpoint.
      GET  /llm/metrics added.
      POST /llm/stream  added (SSE, future-ready).
      All existing routes (/chat, /health, /reload) preserved.

Correlation ID (x-neron-request-id) is read from headers and forwarded
to all log entries for end-to-end tracing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from llm.core.manager import LLMManager
from llm.core.types   import (
    GenerateRequest,
    GenerateResponse,
    LLMRequest,
    LLMResponse,
)

logger  = logging.getLogger("llm.routes")
router  = APIRouter()
manager = LLMManager()

# Protège le remplacement atomique du manager lors d'un /reload.
# asyncio.Lock() est suffisant ici — uvicorn tourne en single-worker.
# Avec plusieurs workers, chacun a son propre espace mémoire : le lock
# protège contre les appels /reload concurrents dans le même worker.
_reload_lock = asyncio.Lock()

# ── Authentication ─────────────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-Neron-API-Key", auto_error=False)
from llm.config import load_config as _load_config
def _current_api_key() -> str:
    return os.getenv("NERON_API_KEY") or _load_config().get("neron", {}).get("api_key", "")

if not _current_api_key():
    logger.warning(
        json.dumps({
            "event":   "auth_disabled",
            "reason":  "NERON_API_KEY not set — all endpoints are unprotected",
            "action":  "set NERON_API_KEY env var to enable authentication",
        })
    )


async def _require_api_key(
    key: str | None = Security(_API_KEY_HEADER),
) -> None:
    """FastAPI dependency — enforces API key on protected routes.

    If NERON_API_KEY is not set (dev/local mode), auth is disabled and
    all requests pass through with a warning logged at startup.
    If set, the X-Neron-API-Key header must match exactly.
    """
    current_key = _current_api_key()
    if not current_key:
        return  # auth disabled — dev mode, warning already logged at import
    if key != current_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# ── Prometheus metrics (registered once) ──────────────────────────────────────

def _safe_counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    from prometheus_client import REGISTRY
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    return Counter(name, doc, labels or [])


def _safe_histogram(name: str, doc: str, labels: list[str] | None = None) -> Histogram:
    from prometheus_client import REGISTRY
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    return Histogram(name, doc, labels or [])


_metric_requests  = _safe_counter(  "neron_llm_requests_total",  "Total generate requests", ["task_type"])
_metric_errors    = _safe_counter(  "neron_llm_errors_total",    "Total errors",            ["task_type"])
_metric_latency   = _safe_histogram("neron_llm_latency_ms",      "Latency ms",              ["task_type"])
_metric_fallbacks = _safe_counter(  "neron_llm_fallbacks_total", "Model fallbacks fired",   ["reason"])


# ── Helper: extract correlation ID ────────────────────────────────────────────

def _request_id(request: Request | None = None, body_rid: str = "") -> str:
    if body_rid:
        return body_rid
    if request is not None:
        return request.headers.get("x-neron-request-id", str(uuid.uuid4()))
    return str(uuid.uuid4())


# ── POST /llm/generate — PRIMARY BUS ENDPOINT ─────────────────────────────────

@router.post("/llm/generate", response_model=GenerateResponse, dependencies=[Depends(_require_api_key)])
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Primary REST bus endpoint.  server/ calls ONLY this route.

    Translates GenerateRequest → internal LLMRequest, runs the manager
    pipeline, then translates back to GenerateResponse.
    """
    rid = _request_id(request, req.request_id)

    logger.info(
        json.dumps({
            "event":      "generate_request",
            "request_id": rid,
            "task_type":  req.task_type,
            "prompt_len": len(req.prompt),
        })
    )
    _metric_requests.labels(task_type=req.task_type).inc()

    t0 = time.monotonic()

    # Translate public request → internal pipeline
    internal = LLMRequest(
        message  = req.prompt,
        task     = req.task_type,
        model    = None if req.model_preference == "auto" else req.model_preference,
        metadata = {k: str(v) for k, v in req.context.items()} if req.context else None,
    )

    result: LLMResponse = await manager.handle(internal)
    latency_ms = int((time.monotonic() - t0) * 1000)

    _metric_latency.labels(task_type=req.task_type).observe(latency_ms)

    # All providers failed
    if result.provider == "none":
        _metric_errors.labels(task_type=req.task_type).inc()
        logger.error(
            json.dumps({
                "event":      "generate_all_failed",
                "request_id": rid,
                "task_type":  req.task_type,
                "error":      result.error,
            })
        )
        raise HTTPException(status_code=502, detail=result.error or "All LLM providers failed")

    warning: str | None = result.error if result.error else None
    if warning:
        _metric_fallbacks.labels(reason="partial_error").inc()

    logger.info(
        json.dumps({
            "event":      "generate_ok",
            "request_id": rid,
            "task_type":  req.task_type,
            "model":      result.model,
            "provider":   result.provider,
            "latency_ms": latency_ms,
            "warning":    warning,
        })
    )

    return GenerateResponse(
        result     = result.response,
        model_used = result.model,
        latency_ms = latency_ms,
        warning    = warning,
    )


# ── POST /llm/stream — SSE streaming (future-ready) ──────────────────────────

@router.post("/llm/stream", dependencies=[Depends(_require_api_key)])
async def stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Streaming endpoint — Server-Sent Events.

    Currently implemented as a single non-streamed generate wrapped in SSE
    format for compatibility.  Replace the inner generator with true token
    streaming once the Ollama streaming path is wired up.
    """
    rid = _request_id(request, req.request_id)

    async def event_generator() -> AsyncIterator[str]:
        try:
            internal = LLMRequest(
                message  = req.prompt,
                task     = req.task_type,
                model    = None if req.model_preference == "auto" else req.model_preference,
                metadata = {k: str(v) for k, v in req.context.items()} if req.context else None,
            )
            result = await manager.handle(internal)

            if result.provider == "none":
                yield f"data: {json.dumps({'token': '', 'done': True, 'error': result.error})}\n\n"
                return

            # Emit entire response as a single token event
            # TODO: replace with token-by-token streaming via OllamaProvider.stream()
            yield f"data: {json.dumps({'token': result.response, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'model_used': result.model})}\n\n"

        except Exception as exc:
            logger.exception(
                json.dumps({"event": "stream_error", "request_id": rid, "error": str(exc)})
            )
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /llm/health ───────────────────────────────────────────────────────────

@router.get("/llm/health")
async def health() -> dict:
    """Health check — reports per-provider status."""
    ollama_up: bool = False
    try:
        # Reuse the shared OllamaProvider client (no new TCP connection).
        # Override timeout for this call only: health checks must be fast.
        ollama_provider = manager.providers.get("ollama")
        if ollama_provider is not None:
            r = await ollama_provider._client.get(
                "/api/tags", timeout=3.0,
            )
            ollama_up = r.status_code == 200
    except Exception:
        pass

    return {
        "status":    "ok" if ollama_up else "degraded",
        "service":   "neron_llm",
        "version":   "2.1.0",
        "providers": {
            "ollama": "up" if ollama_up else "down",
            "claude": "configured",
        },
    }


# ── POST /llm/reload ─────────────────────────────────────────────────────────

@router.post("/llm/reload", dependencies=[Depends(_require_api_key)])
async def reload() -> dict:
    """Hot-reload YAML config without restarting the service.

    Safe mode:
      1. Lock prevents concurrent reloads.
      2. New manager is fully constructed before swapping.
      3. Old manager's HTTP clients are closed after the swap.
    """
    global manager
    async with _reload_lock:
        try:
            from llm.config import reload_config
            reload_config()
            new_manager = LLMManager()       # raises if config is broken — old manager kept
            old_manager, manager = manager, new_manager
            logger.info(json.dumps({"event": "config_reloaded"}))
        except Exception as exc:
            logger.error(json.dumps({"event": "config_reload_failed", "error": str(exc)}))
            raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")

    # Close old manager's HTTP clients outside the lock — no need to block
    # incoming requests while waiting for TCP connections to drain.
    try:
        await old_manager.aclose()
    except Exception as exc:
        logger.warning(json.dumps({"event": "old_manager_close_error", "error": str(exc)}))

    return {"status": "ok", "message": "Configuration reloaded"}


# ── GET /llm/metrics ─────────────────────────────────────────────────────────

@router.get("/llm/metrics", dependencies=[Depends(_require_api_key)])
async def metrics() -> StreamingResponse:
    """Prometheus metrics endpoint."""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── POST /chat — backward compat (kept from v1) ───────────────────────────────

@router.post("/chat", response_model=LLMResponse, dependencies=[Depends(_require_api_key)])
async def chat(request: LLMRequest) -> LLMResponse:
    """Legacy endpoint — preserved for backward compat.  Prefer /llm/generate."""
    logger.info(
        json.dumps({
            "event": "chat_request_legacy",
            "task":  request.task,
        })
    )
    result = await manager.handle(request)
    if result.error and result.provider == "none":
        raise HTTPException(status_code=502, detail=result.error)
    return result
