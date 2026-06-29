from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from llm import app as llm_app
from llm.registry_client import RegistryClient, RegistryClientConfig


@pytest.mark.asyncio
async def test_register_uses_official_header_and_expected_payload():
    captured: dict = {}

    async def core(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "healthy"})

    client = RegistryClient(
        RegistryClientConfig(
            core_url="http://core.test",
            api_key="secret",
            service_host="llm.internal",
            service_port=8765,
        ),
        transport=httpx.MockTransport(core),
    )
    try:
        registered = await client.register()
    finally:
        await client.stop()

    assert registered is True
    assert captured["request"].headers["X-Neron-API-Key"] == "secret"
    assert "X-API-Key" not in captured["request"].headers
    assert captured["payload"] == {
        "service_name": "llm",
        "host": "llm.internal",
        "port": 8765,
        "version": "0.1.0",
        "status": "healthy",
        "capabilities": ["text_generation", "chat", "completion"],
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_start_without_core_does_not_crash():
    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Core unavailable", request=request)

    client = RegistryClient(
        RegistryClientConfig(core_url="http://core.test"),
        transport=httpx.MockTransport(unavailable),
    )
    try:
        await client.start()
        await asyncio.sleep(0.01)
        assert client._registered is False
        assert client._task is not None
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_registration_retries_then_heartbeat_runs():
    paths: list[str] = []
    registration_attempts = 0

    async def core(request: httpx.Request) -> httpx.Response:
        nonlocal registration_attempts
        paths.append(request.url.path)
        if request.url.path == "/registry/register":
            registration_attempts += 1
            if registration_attempts == 1:
                return httpx.Response(503, json={"detail": "Core starting"})
        return httpx.Response(200, json={"status": "healthy"})

    client = RegistryClient(
        RegistryClientConfig(
            core_url="http://core.test",
            heartbeat_interval=0.01,
        ),
        transport=httpx.MockTransport(core),
    )
    try:
        await client.start()
        await asyncio.sleep(0.045)
    finally:
        await client.stop()

    assert registration_attempts == 2
    assert "/registry/heartbeat" in paths


def test_environment_configuration():
    env = {
        "NERON_CORE_URL": "http://core.internal:8010/",
        "NERON_API_KEY": "key",
        "NERON_SERVICE_HOST": "llm.internal",
        "NERON_SERVICE_PORT": "9765",
    }
    with patch.dict(os.environ, env, clear=False):
        config = RegistryClientConfig.from_env()

    assert config.core_url == "http://core.internal:8010"
    assert config.api_key == "key"
    assert config.service_host == "llm.internal"
    assert config.service_port == 9765


@pytest.mark.asyncio
async def test_llm_startup_starts_registry_client():
    fake_client = Mock()
    fake_client.start = AsyncMock()
    fake_client.stop = AsyncMock()

    with patch.object(
        llm_app.RegistryClient,
        "from_env",
        return_value=fake_client,
    ):
        await llm_app.on_startup()

    fake_client.start.assert_awaited_once()
    assert llm_app.app.state.registry_client is fake_client
    await fake_client.stop()
