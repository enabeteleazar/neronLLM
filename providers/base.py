# Base provider interface for all LLM backends.
# Every provider MUST be async to enable parallel and race execution

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, message: str, model: str, timeout: float | None = None,
                       json_mode: bool = False) -> str:
        """Send a message to the LLM and return the text response.

        Args:
            timeout: Per-call timeout override in seconds.
                     Providers should use their configured default if None.

        Raises:
            Exception: On timeout, HTTP error, or any failure.
                      The manager catches exceptions and formats them
                      into LLMResponse(error=...).
        """
        ...

    async def aclose(self) -> None:
        """Release long-lived resources (e.g. shared HTTP client).

        Called by LLMManager.aclose() at application shutdown.
        Default is a no-op — override in providers that hold connections.
        """
        pass
