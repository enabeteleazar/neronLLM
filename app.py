# llm/app.py
# Neron LLM microservice — main entry point.
#
# v3.0 refactoring:
#   - pydantic-settings (llm.settings) for infra values
#   - squelette commun (server/common/service.py) : health, metrics, registry
#   - dependency injection via app.state (no module-level singleton)
#   - split health endpoints (/llm/health/live, /llm/health/ready)

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm.api.routes import router
from llm.core.manager import LLMManager
from llm.settings import get_settings
from server.common.paths import service_version
from server.common.service import create_service_app

VERSION = service_version(__file__)


# ── Structured JSON logging ───────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        try:
            json.loads(msg)
            return msg
        except (json.JSONDecodeError, TypeError):
            pass

        return json.dumps({
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        })


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())

logging.basicConfig(level=logging.INFO, handlers=[_handler])
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("llm")


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def _setup(app: FastAPI):
    settings = get_settings()

    if not settings.auth_enabled:
        logger.warning(json.dumps({
            "event": "auth_disabled",
            "reason": "NERON_API_KEY not set — all endpoints are unprotected",
            "action": "set NERON_API_KEY env var to enable authentication",
        }))

    app.state.settings = settings
    app.state.manager = LLMManager()
    app.state.reload_lock = asyncio.Lock()

    logger.info(json.dumps({
        "event": "llm_started",
        "version": VERSION,
        "host": settings.service_host,
        "port": settings.service_port,
    }))

    try:
        yield
    finally:
        try:
            await app.state.manager.aclose()
        except Exception as exc:
            logger.warning(json.dumps({
                "event": "llm_shutdown_error",
                "error": str(exc),
            }))
        logger.info(json.dumps({"event": "llm_stopped"}))


app = create_service_app(
    name="llm",
    title="neronOS_LLM",
    description="Microservice IA — routing modèles, abstraction providers",
    version=VERSION,
    capabilities=["text_generation", "chat", "completion"],
    setup=_setup,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    _s = get_settings()
    uvicorn.run(
        "llm.app:app",
        host=_s.service_host,
        port=_s.service_port,
        workers=1,
    )
