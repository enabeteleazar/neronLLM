from __future__ import annotations

import os
from typing import Any

import httpx

from core.providers.models import ProviderRequest, ProviderResponse, ProviderStatus, ProviderType
from core.providers.protocol import ProviderProtocol


class LLMProvider(ProviderProtocol):
    """Provider bridge to the Néron LLM microservice."""

    def __init__(
        self,
        client: object | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._client = client
        self._base_url = (base_url or os.getenv("NERON_LLM_URL") or "http://localhost:8765").rstrip("/")
        self._api_key = api_key if api_key is not None else os.getenv("NERON_API_KEY", "")
        self._timeout = float(timeout or os.getenv("NERON_LLM_TIMEOUT") or 30.0)
        self._status: ProviderStatus = "unknown"

    @property
    def name(self) -> str:
        return "llm"

    @property
    def type(self) -> ProviderType:
        return "llm"

    @property
    def status(self) -> ProviderStatus:
        return self._status

    @property
    def capabilities(self) -> list[str]:
        return ["health", "generate", "status"]

    async def health(self) -> ProviderResponse:
        return await self.execute(ProviderRequest(action="health"))

    async def execute(self, request: ProviderRequest) -> ProviderResponse:
        action = request.action.strip().lower()
        try:
            if action in {"health", "status"}:
                health = await self._health()
                self._status = self._status_from_health(health)
                return self._response(request, result=health)

            if action == "generate":
                payload = request.payload
                prompt = str(payload.get("prompt") or payload.get("text") or "")
                if not prompt:
                    return self._response(
                        request,
                        status="degraded",
                        error="prompt is required",
                    )
                result = await self._generate(
                    task_type=str(payload.get("task_type") or payload.get("task") or "chat"),
                    prompt=prompt,
                    context=payload.get("context") or {},
                    request_id=request.trace_id,
                    model_preference=str(payload.get("model_preference") or "auto"),
                )
                self._status = "healthy" if not result.get("warning") else "degraded"
                return self._response(
                    request,
                    result=result,
                )

            return self._response(
                request,
                status="unhealthy",
                error=f"unsupported llm provider action: {request.action}",
            )
        except Exception as exc:
            self._status = "unavailable"
            return self._response(request, status="unavailable", error=str(exc))

    async def _health(self) -> dict[str, Any]:
        if self._client is not None:
            return await self._client.health()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(min(self._timeout, 5.0)),
            headers=self._headers(),
        ) as client:
            response = await client.get("/llm/health")
            response.raise_for_status()
            return response.json()

    async def _generate(
        self,
        *,
        task_type: str,
        prompt: str,
        context: dict[str, Any],
        request_id: str | None,
        model_preference: str,
    ) -> dict[str, Any]:
        if self._client is not None:
            result = await self._client.generate(
                task_type=task_type,
                prompt=prompt,
                context=context,
                request_id=request_id,
            )
            return {
                "text": getattr(result, "result", ""),
                "model": getattr(result, "model_used", ""),
                "latency_ms": getattr(result, "latency_ms", 0),
                "warning": getattr(result, "warning", None),
            }

        payload = {
            "task_type": task_type if task_type in {"code", "reasoning", "chat", "agent"} else "chat",
            "prompt": prompt,
            "context": {str(key): str(value) for key, value in context.items()},
            "model_preference": model_preference,
            "request_id": request_id or "",
        }
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                max(self._timeout, 360.0)
                if task_type in {"code", "agent"}
                else self._timeout
            ),
            headers=self._headers(),
        ) as client:
            response = await client.post("/llm/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return {
                "text": data.get("result", ""),
                "model": data.get("model_used", ""),
                "latency_ms": data.get("latency_ms", 0),
                "warning": data.get("warning"),
            }

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    @staticmethod
    def _status_from_health(health: dict[str, Any]) -> ProviderStatus:
        status = str(health.get("status") or "").lower()
        if status in {"ok", "healthy"}:
            return "healthy"
        if status == "degraded":
            return "degraded"
        return "unavailable"

    def _response(
        self,
        request: ProviderRequest,
        *,
        result: Any = None,
        status: ProviderStatus | None = None,
        error: str | None = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            provider=self.name,
            action=request.action,
            status=status or self._status,
            result=result,
            error=error,
            trace_id=request.trace_id,
        )
