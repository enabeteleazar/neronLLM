from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import logging
import os
from typing import Any

import httpx


logger = logging.getLogger("llm.registry")

SERVICE_NAME = "llm"
SERVICE_VERSION = "0.1.0"
SERVICE_CAPABILITIES = ["text_generation", "chat", "completion"]


@dataclass(frozen=True)
class RegistryClientConfig:
    core_url: str = "http://localhost:8010"
    api_key: str = ""
    service_host: str = "localhost"
    service_port: int = 8765
    heartbeat_interval: float = 30.0
    timeout: float = 5.0

    @classmethod
    def from_env(cls) -> RegistryClientConfig:
        return cls(
            core_url=os.getenv("NERON_CORE_URL", "http://localhost:8010").rstrip("/"),
            api_key=os.getenv("NERON_API_KEY", ""),
            service_host=os.getenv("NERON_SERVICE_HOST", "localhost"),
            service_port=int(os.getenv("NERON_SERVICE_PORT", "8765")),
        )


class RegistryClient:
    """Non-blocking LLM bootstrap client for the Core Service Registry."""

    def __init__(
        self,
        config: RegistryClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.core_url,
            timeout=config.timeout,
            transport=transport,
            headers=self._headers(),
        )
        self._task: asyncio.Task[None] | None = None
        self._registered = False

    @classmethod
    def from_env(cls) -> RegistryClient:
        return cls(RegistryClientConfig.from_env())

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(),
                name="llm-registry-heartbeat",
            )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._client.aclose()

    async def register(self) -> bool:
        payload: dict[str, Any] = {
            "service_name": SERVICE_NAME,
            "host": self.config.service_host,
            "port": self.config.service_port,
            "version": SERVICE_VERSION,
            "status": "healthy",
            "capabilities": SERVICE_CAPABILITIES,
            "metadata": {},
        }
        try:
            response = await self._client.post("/registry/register", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Core Registry registration unavailable: %s", exc)
            return False

        logger.info("Registered LLM with Core Registry at %s", self.config.core_url)
        return True

    async def heartbeat(self) -> bool:
        try:
            response = await self._client.post(
                "/registry/heartbeat",
                json={"service_name": SERVICE_NAME, "status": "healthy"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Core Registry heartbeat failed: %s", exc)
            return False

        logger.debug("LLM Registry heartbeat sent")
        return True

    async def _run(self) -> None:
        self._registered = await self.register()
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            if self._registered:
                self._registered = await self.heartbeat()
            else:
                self._registered = await self.register()

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            return {}
        return {"X-Neron-API-Key": self.config.api_key}
