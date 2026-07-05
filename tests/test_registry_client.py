from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from llm import app as llm_app
from server.common.registry.client import RegistryClient


def _client(**kwargs):
    defaults = {
        "service_name": "llm",
        "version": "0.1.0",
        "host": "localhost",
        "port": 8765,
        "capabilities": ["text_generation", "chat", "completion"],
        "metadata": {},
    }
    defaults.update(kwargs)
    return RegistryClient(**defaults)


@pytest.mark.asyncio
async def test_register_uses_official_header_and_expected_payload():
    captured: list[tuple[httpx.Request, dict | None]] = []

    async def core(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        captured.append((request, payload))
        return httpx.Response(200, json={"status": "healthy"})

    client = _client(
        core_url="http://core.test",
        api_key="secret",
        host="llm.internal",
        transport=httpx.MockTransport(core),
    )
    try:
        registered = await client.register()
    finally:
        await client.stop()

    request, payload = captured[0]
    assert registered is True
    assert request.headers["Authorization"] == "secret"
    assert "X-API-Key" not in request.headers
    assert payload == {
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

    client = _client(
        core_url="http://core.test",
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

    client = _client(
        core_url="http://core.test",
        heartbeat_interval=0.01,
        transport=httpx.MockTransport(core),
    )
    try:
        await client.start()
        await asyncio.sleep(0.045)
    finally:
        await client.stop()

    assert registration_attempts == 2
    assert "/registry/heartbeat" in paths


@pytest.mark.asyncio
async def test_environment_configuration():
    env = {
        "NERON_CORE_URL": "http://core.internal:8010/",
        "NERON_API_KEY": "key",
        "NERON_SERVICE_HOST": "llm.internal",
        "NERON_SERVICE_PORT": "9765",
    }
    with patch.dict(os.environ, env, clear=False):
        client = _client()

    assert client.settings.core_url == "http://core.internal:8010"
    assert client.settings.api_key == "key"
    assert client.service.host == "llm.internal"
    assert client.service.port == 9765
    await client.stop()


@pytest.mark.asyncio
async def test_llm_startup_starts_registry_client():
    fake_client = Mock()
    fake_client.start = AsyncMock()
    fake_client.stop = AsyncMock()

    with patch.object(llm_app, "RegistryClient", return_value=fake_client):
        await llm_app.on_startup()

    fake_client.start.assert_awaited_once()
    assert llm_app.app.state.registry_client is fake_client
    await fake_client.stop()
