"""neron_llm/api/routes.py
API routes for neron_llm — fully async.

v3.0: dependency injection via app.state — no module-level manager.
      Health split:
        GET /llm/health/live   → liveness  (process up, no I/O)
        GET /llm/health/ready  → readiness (providers reachable, 503 si non)
        GET /llm/health        → compat (payload historique, utilisé par doctor)

Correlation ID (x-neron-request-id) is read from headers and forwarded
to all log entries for end-to-end tracing.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CollectorRegistry, Counter, Histogram

from llm.core.manager import LLMManager
from llm.core.types import (
    GenerateRequest,
    GenerateResponse,
    LLMRequest,
    LLMResponse,
)
from llm.settings import LLMSettings, get_settings

logger = logging.getLogger("llm.routes")
router = APIRouter()


# ── Dependency injection ──────────────────────────────────────────────────────

def get_manager(request: Request) -> LLMManager:
    return request.app.state.manager


def get_app_settings(request: Request) -> LLMSettings:
    # app.state d'abord (posé par lifespan) ; repli sur le cache module
    # pour les contextes de test qui montent le router sans lifespan.
    return getattr(request.app.state, "settings", None) or get_settings()


# ── Authentication ────────────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)


async def _require_api_key(
    request: Request,
    key: str | None = Security(_API_KEY_HEADER),
) -> None:
    """FastAPI dependency — enforces API key on protected routes.

    If NERON_API_KEY is not set (dev/local mode), auth is disabled
    (warning logged once at startup by lifespan).
    If set, the Authorization header must be `Bearer <token>` (ou le token
    brut, compat legacy) matching exactly.
    """
    settings = get_app_settings(request)
    if not settings.auth_enabled:
        return

    if key is None:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    token = key
    if key.lower().startswith("bearer "):
        token = key[7:].strip()

    if token != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ── Prometheus metrics ────────────────────────────────────────────────────────
# Créées une fois à l'import. En cas de ré-import (tests), repli sur un
# registre jetable — aucune API privée de prometheus_client n'est utilisée.

def _build_metrics(registry=None) -> dict:
    kw = {"registry": registry} if registry is not None else {}
    return {
        "requests":  Counter("neron_llm_requests_total", "Total generate requests", ["task_type"], **kw),
        "errors":    Counter("neron_llm_errors_total", "Total errors", ["task_type"], **kw),
        "latency":   Histogram("neron_llm_latency_ms", "Latency ms", ["task_type"], **kw),
        "fallbacks": Counter("neron_llm_fallbacks_total", "Model fallbacks fired", ["reason"], **kw),
    }


try:
    _metrics = _build_metrics()
except ValueError:  # already registered (module re-imported in tests)
    _metrics = _build_metrics(registry=CollectorRegistry())

_metric_requests = _metrics["requests"]
_metric_errors = _metrics["errors"]
_metric_latency = _metrics["latency"]
_metric_fallbacks = _metrics["fallbacks"]


# ── Helper: extract correlation ID ────────────────────────────────────────────

def _request_id(request: Request | None = None, body_rid: str = "") -> str:
    if body_rid:
        return body_rid
    if request is not None:
        return request.headers.get("x-neron-request-id", str(uuid.uuid4()))
    return str(uuid.uuid4())


# ── POST /llm/generate — PRIMARY BUS ENDPOINT ─────────────────────────────────

@router.post("/llm/generate", response_model=GenerateResponse, dependencies=[Depends(_require_api_key)])
async def generate(
    req: GenerateRequest,
    request: Request,
    manager: LLMManager = Depends(get_manager),
) -> GenerateResponse:
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

    internal = LLMRequest(
        message  = req.prompt,
        task     = req.task_type,
        model    = None if req.model_preference == "auto" else req.model_preference,
        metadata = {k: str(v) for k, v in req.context.items()} if req.context else None,
    )

    result: LLMResponse = await manager.handle(internal)
    latency_ms = int((time.monotonic() - t0) * 1000)

    _metric_latency.labels(task_type=req.task_type).observe(latency_ms)

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
async def stream(
    req: GenerateRequest,
    request: Request,
    manager: LLMManager = Depends(get_manager),
) -> StreamingResponse:
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

            # TODO: replace with token-by-token streaming via OllamaProvider.stream()
            yield f"data: {json.dumps({'token': result.response, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'model_used': result.model})}\n\n"

        except Exception as exc:
            logger.exception(
                json.dumps({"event": "stream_error", "request_id": rid, "error": str(exc)})
            )
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Health: liveness / readiness (split Kubernetes-compatible) ────────────────

async def _ollama_reachable(manager: LLMManager) -> bool:
    try:
        provider = manager.providers.get("ollama")
        if provider is None:
            return False
        r = await provider._client.get("/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@router.get("/llm/health/live")
async def health_live(request: Request) -> dict:
    """Liveness — process vivant, aucune I/O. Toujours 200 si on répond."""
    return {
        "status": "alive",
        "service": "neron_llm",
        "version": getattr(request.app.state, "version", "unknown"),
    }


@router.get("/llm/health/ready")
async def health_ready(
    request: Request,
    response: Response,
    manager: LLMManager = Depends(get_manager),
) -> dict:
    """Readiness — providers joignables. 503 sinon (retiré du LB/registry)."""
    ollama_up = await _ollama_reachable(manager)
    if not ollama_up:
        response.status_code = 503
    return {
        "status": "ready" if ollama_up else "not_ready",
        "service": "neron_llm",
        "version": getattr(request.app.state, "version", "unknown"),
        "providers": {
            "ollama": "up" if ollama_up else "down",
            "claude": "configured",
        },
    }


@router.get("/llm/health")
async def health(
    request: Request,
    manager: LLMManager = Depends(get_manager),
) -> dict:
    """Compat historique (doctor, scripts) — payload inchangé, version dynamique."""
    ollama_up = await _ollama_reachable(manager)
    return {
        "status":    "ok" if ollama_up else "degraded",
        "service":   "neron_llm",
        "version":   getattr(request.app.state, "version", "unknown"),
        "providers": {
            "ollama": "up" if ollama_up else "down",
            "claude": "configured",
        },
    }


# ── POST /llm/reload ─────────────────────────────────────────────────────────

@router.post("/llm/reload", dependencies=[Depends(_require_api_key)])
async def reload(request: Request) -> dict:
    """Hot-reload YAML config without restarting the service.

    Safe mode:
      1. app.state.reload_lock prevents concurrent reloads.
      2. New manager is fully constructed before swapping.
      3. Old manager's HTTP clients are closed after the swap.
    """
    state = request.app.state
    async with state.reload_lock:
        try:
            from llm.config import reload_config
            reload_config()
            new_manager = LLMManager()   # raises if config is broken — old manager kept
            old_manager, state.manager = state.manager, new_manager
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
