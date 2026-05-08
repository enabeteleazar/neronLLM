# llm/app.py
# Neron LLM microservice — main entry point.

from __future__ import annotations

import json
import logging

from fastapi import FastAPI

from llm.api.routes import router

# ── Structured JSON logging ───────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # If message is already JSON-parseable, forward as-is
        msg = record.getMessage()
        try:
            json.loads(msg)
            return msg
        except (json.JSONDecodeError, TypeError):
            pass
        return json.dumps({
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": msg,
        })


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "neronOS_LLM",
    description = "Microservice IA — routing modèles, abstraction providers",
    version     = "2.1.1",
)

app.include_router(router)


@app.on_event("startup")
async def on_startup() -> None:
    logging.getLogger("llm").info(
        json.dumps({"event": "llm_started", "version": "2.1.1", "port": 8765})
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    from llm.api.routes import manager
    await manager.aclose()
    logging.getLogger("llm").info(
        json.dumps({"event": "llm_stopped"})
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host    = "127.0.0.1",
        port    = 8765,
        workers = 1,
    )
