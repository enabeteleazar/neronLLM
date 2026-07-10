from __future__ import annotations

import time
from typing import Any

from llm.client.types import TaskType
from llm.core.manager import LLMManager
from llm.core.types import GenerateResponse, LLMRequest


class NéronLLMClient:
    """Compatibility client for Core over the current LLM manager contract."""

    def __init__(self) -> None:
        self.manager = LLMManager()

    async def generate(
        self,
        *,
        prompt: str,
        task_type: TaskType = "chat",
        context: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> GenerateResponse:
        started_at = time.monotonic()
        response = await self.manager.handle(
            LLMRequest(
                message=prompt,
                task=task_type,
                metadata=(
                    {key: str(value) for key, value in context.items()}
                    if context
                    else None
                ),
            )
        )
        return GenerateResponse(
            result=response.response,
            model_used=response.model if not response.error else "degraded",
            latency_ms=int((time.monotonic() - started_at) * 1000),
            warning=response.error,
        )

    async def aclose(self) -> None:
        await self.manager.aclose()
