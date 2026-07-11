from __future__ import annotations

import time
from typing import Any

from llm.client.types import TaskType
from llm.core.manager import LLMManager
from llm.core.types import GenerateResponse, LLMRequest

# Message renvoyé quand toutes les tentatives ont échoué et qu'aucun texte
# n'a pu être généré — évite de laisser remonter une chaîne vide jusqu'au
# TTS/chat (qui échouerait silencieusement ou de façon confuse pour
# l'utilisateur, sans lui dire ce qui s'est passé).
_DEGRADED_FALLBACK_MESSAGE = (
    "Désolé, je suis encore en train de traiter une demande précédente — "
    "réessaie dans quelques instants."
)


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
        result_text = response.response
        if not result_text and response.error:
            # Toutes les tentatives/fallbacks ont échoué (ex : Ollama
            # saturé, 429 persistant) — on ne laisse jamais une chaîne
            # vide remonter en silence.
            result_text = _DEGRADED_FALLBACK_MESSAGE
        return GenerateResponse(
            result=result_text,
            model_used=response.model if not response.error else "degraded",
            latency_ms=int((time.monotonic() - started_at) * 1000),
            warning=response.error,
        )

    async def aclose(self) -> None:
        await self.manager.aclose()
