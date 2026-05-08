#Claude (Anthropic) provider — async HTTP via a shared httpx.AsyncClient.

from __future__ import annotations

import logging
import os

import httpx

from llm.config import get_llm_config
from llm.providers.base import BaseProvider

logger = logging.getLogger("neron_llm.claude")

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION  = "2023-06-01"


class ClaudeProvider(BaseProvider):
    """Async provider for the Anthropic Claude API.

    The underlying httpx.AsyncClient is shared for the lifetime of this
    object.  Call aclose() (or use LLMManager.aclose()) to release the
    connection pool gracefully on shutdown.
    """

    def __init__(self) -> None:
        cfg = get_llm_config()

        self.api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "ClaudeProvider: ANTHROPIC_API_KEY not set — calls will raise."
            )

        self.max_tokens:  int   = int(cfg.get("claude_max_tokens", 1024))
        self.temperature: float = float(cfg.get("temperature", 0.7))
        self._timeout_default: float = float(cfg.get("timeout", 300))
        limits  = httpx.Limits(
            max_connections          = int(cfg.get("claude_max_connections", 20)),
            max_keepalive_connections= int(cfg.get("claude_max_keepalive_connections", 5)),
        )

        # Authentication headers are baked into the client once at startup.
        # No need to rebuild them on every generate() call.
        self._client = httpx.AsyncClient(
            base_url = ANTHROPIC_BASE_URL,
            timeout  = self._timeout_default,
            limits   = limits,
            headers  = {
                "x-api-key":          self.api_key,
                "anthropic-version":  ANTHROPIC_VERSION,
                "content-type":       "application/json",
            },
        )
        logger.debug("ClaudeProvider initialised — timeout=%s max_tokens=%s", self._timeout_default, self.max_tokens)

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(self, message: str, model: str, timeout: float | None = None) -> str:
        """Generate a response via the Anthropic Messages API.

        Args:
            timeout: Per-call timeout override. Uses configured default if None.

        Raises:
            ValueError: If API key is missing or response is empty.
            httpx.TimeoutException: On timeout.
            httpx.HTTPStatusError: On HTTP 4xx/5xx.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        effective_timeout = timeout if timeout is not None else self._timeout_default

        payload = {
            "model":      model,
            "max_tokens": self.max_tokens,
            "temperature":self.temperature,
            "messages":   [{"role": "user", "content": message}],
        }

        logger.debug("claude | POST /v1/messages model=%s timeout=%s", model, effective_timeout)

        r = await self._client.post("/v1/messages", json=payload, timeout=effective_timeout)
        r.raise_for_status()
        data = r.json()

        content = data.get("content", [])
        if not content:
            raise ValueError("Claude returned empty content")

        return content[0].get("text", "")

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        logger.debug("ClaudeProvider: HTTP client closed")
