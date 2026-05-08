# Ollama provider — async HTTP via a shared httpx.AsyncClient.

from __future__ import annotations

import logging

import httpx

from llm.config import get_llm_config
from llm.providers.base import BaseProvider

logger = logging.getLogger("neron_llm.ollama")


class OllamaProvider(BaseProvider):
    """Async provider for local Ollama instances.

    The underlying httpx.AsyncClient is shared for the lifetime of this
    object.  Call aclose() (or use LLMManager.aclose()) to release the
    connection pool gracefully on shutdown.
    """

    def __init__(self) -> None:
        cfg = get_llm_config()
        host    = cfg.get("host", "http://localhost:11434").rstrip("/")
        limits  = httpx.Limits(
            max_connections          = int(cfg.get("ollama_max_connections", 50)),
            max_keepalive_connections= int(cfg.get("ollama_max_keepalive_connections", 10)),
        )
        # Default timeout for single/parallel modes (long generations)
        self._timeout_default: float = float(cfg.get("timeout", 300))
        # Shorter timeout for race mode — fail fast, let the other provider win
        self._timeout_race:    float = float(cfg.get("race_timeout", 240))

        self._client = httpx.AsyncClient(
            base_url = host,
            # No client-level timeout: we set it per-request to support mode-aware values
            timeout  = None,
            limits   = limits,
        )
        logger.debug(
            "OllamaProvider initialised — base_url=%s default_timeout=%s race_timeout=%s",
            host, self._timeout_default, self._timeout_race,
        )

    async def generate(self, message: str, model: str, timeout: float | None = None) -> str:
        """Generate a response via Ollama's /api/generate endpoint.

        Args:
            timeout: Override the default timeout for this call.
                     Pass a shorter value for race mode.

        Raises:
            httpx.TimeoutException: On timeout.
            httpx.HTTPStatusError: On HTTP 4xx/5xx.
            ValueError: On unexpected response format.
        """
        effective_timeout = timeout if timeout is not None else self._timeout_default
        payload = {
            "model":  model,
            "prompt": message,
            "stream": False,
        }

        logger.debug("ollama | POST /api/generate model=%s timeout=%s", model, effective_timeout)

        r = await self._client.post("/api/generate", json=payload, timeout=effective_timeout)
        r.raise_for_status()
        data = r.json()

        if "response" not in data:
            raise ValueError(f"Ollama unexpected format: {list(data.keys())}")

        return data["response"]

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        logger.debug("OllamaProvider: HTTP client closed") 
