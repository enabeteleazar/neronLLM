# llm/app.py
# Neron LLM microservice — main entry point.

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm.api.routes import router

VERSION = "2.1.2"
PORT = 8765

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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(json.dumps({
        "event": "llm_started",
        "version": VERSION,
        "port": PORT,
    }))

    try:
        yield
    finally:
        try:
            from llm.api.routes import manager

            if manager and hasattr(manager, "aclose"):
                await manager.aclose()

        except Exception as exc:
            logger.warning(json.dumps({
                "event": "llm_shutdown_error",
                "error": str(exc),
            }))

        logger.info(json.dumps({"event": "llm_stopped"}))


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="neronOS_LLM",
    description="Microservice IA — routing modèles, abstraction providers",
    version=VERSION,
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=PORT,
        workers=1,
    )
