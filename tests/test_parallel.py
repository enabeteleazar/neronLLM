"""Tests for neron_llm v1.0 — parallel, race, single, strategy, fallback, retry.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import time

import pytest

from fastapi import HTTPException
from unittest.mock import patch

from neron_llm.core.manager import LLMManager, MAX_RETRIES
from neron_llm.core.router import LLMRouter
from neron_llm.core.strategy import StrategyEngine
from core.types import LLMRequest, LLMResponse
from neron_llm.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Fake providers (async — conform to BaseProvider)
# ---------------------------------------------------------------------------


class SlowProvider(BaseProvider):
    """Fake provider simulating a network call of known duration."""

    def __init__(self, name: str, delay: float):
        self.name = name
        self.delay = delay

    async def generate(self, message: str, model: str) -> str:
        await asyncio.sleep(self.delay)
        return f"[{self.name}] response after {self.delay}s"


class FailingProvider(BaseProvider):
    """Fake provider that always raises an exception."""

    def __init__(self, fail_count: int = 999):
        self.fail_count = fail_count
        self.call_count = 0

    async def generate(self, message: str, model: str) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise RuntimeError(f"Provider error (attempt {self.call_count})")
        return "recovered"


class RecoveringProvider(BaseProvider):
    """Fails on first attempt, succeeds on second (tests retry)."""

    def __init__(self):
        self.call_count = 0

    async def generate(self, message: str, model: str) -> str:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("First attempt failed")
        return f"Success on attempt {self.call_count}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_manager_with_slow_providers() -> LLMManager:
    mgr = LLMManager()
    mgr.providers = {
        "fast": SlowProvider("fast", 0.2),
        "medium": SlowProvider("medium", 0.3),
        "slow": SlowProvider("slow", 0.4),
    }
    return mgr


def make_request(
    message: str = "test",
    task: str = "default",
    mode: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> LLMRequest:
    return LLMRequest(message=message, task=task, mode=mode, provider=provider, model=model)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


def test_strategy_explicit_mode():
    """Explicit mode takes priority over task-based strategy."""
    engine = StrategyEngine()
    assert engine.decide(task="code", mode="single") == "single"
    assert engine.decide(task="chat", mode="parallel") == "parallel"


def test_strategy_task_based():
    """Task determines mode when no explicit mode is set."""
    engine = StrategyEngine()
    assert engine.decide(task="code") == "parallel"
    assert engine.decide(task="chat") == "race"
    assert engine.decide(task="fast") == "single"


def test_strategy_default():
    """Unknown task falls back to 'single'."""
    engine = StrategyEngine()
    assert engine.decide(task="unknown") == "single"
    assert engine.decide() == "single"


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def test_router_select_model():
    """Router selects model based on task from config."""
    router = LLMRouter()
    # These depend on neron.yaml being present
    model = router.select_model("default")
    assert isinstance(model, str)
    assert len(model) > 0


def test_router_select_provider():
    """Explicit provider takes priority."""
    router = LLMRouter()
    assert router.select_provider("claude") == "claude"
    assert router.select_provider() == "ollama"  # default


def test_router_fallback_chain():
    """Fallback provider chain works correctly."""
    router = LLMRouter()
    assert router.get_fallback_provider("ollama") == "claude"
    assert router.get_fallback_provider("claude") is None
    assert router.get_fallback_provider("unknown") is None


# ---------------------------------------------------------------------------
# Execution mode tests
# ---------------------------------------------------------------------------


def test_parallel_execution():
    """Providers run in parallel — total time ≈ max(delay), not sum(delay)."""
    mgr = make_manager_with_slow_providers()
    req = make_request(mode="parallel")

    start = time.perf_counter()
    result = asyncio.run(mgr.handle(req))
    elapsed = time.perf_counter() - start

    # Parallel: ≈ 0.4s | Sequential: ≈ 0.9s
    assert elapsed < 0.6, (
        f"NOT PARALLEL! Time={elapsed:.2f}s, expected < 0.6s. "
        f"Sequential would be ~0.9s."
    )
    assert result.error is None
    assert len(result.response) > 0

    print(f"\n  PARALLEL confirmed: {elapsed:.3f}s (sequential ~0.9s)")


def test_race_execution():
    """In race mode, the fastest provider wins and others are cancelled."""
    mgr = make_manager_with_slow_providers()
    req = make_request(mode="race")

    start = time.perf_counter()
    result = asyncio.run(mgr.handle(req))
    elapsed = time.perf_counter() - start

    assert elapsed < 0.35, (
        f"RACE too slow: {elapsed:.2f}s. 'fast' (0.2s) should have won."
    )
    assert result.error is None
    assert "fast" in result.response

    print(f"\n  RACE confirmed: {elapsed:.3f}s — winner={result.provider}")


def test_single_execution():
    """Single mode returns structured LLMResponse."""
    mgr = LLMManager()
    mgr.providers = {"ollama": SlowProvider("ollama", 0.05)}
    mgr.router.select_provider = lambda provider=None: "ollama"
    mgr.router.select_model = lambda task=None: "test-model"

    req = make_request(mode="single")
    result = asyncio.run(mgr.handle(req))

    assert isinstance(result, LLMResponse)
    assert result.provider == "ollama"
    assert result.model == "test-model"
    assert result.error is None
    assert "ollama" in result.response

    print(f"\n  SINGLE structured: {result}")


def test_sequential_baseline():
    """Sequential baseline — 3 calls × 0.2s should take > 0.4s."""
    mgr = LLMManager()
    mgr.providers = {"fast": SlowProvider("fast", 0.2)}
    mgr.router.select_provider = lambda provider=None: "fast"
    mgr.router.select_model = lambda task=None: "test"

    req = make_request(mode="single")

    start = time.perf_counter()
    asyncio.run(mgr.handle(req))
    asyncio.run(mgr.handle(req))
    asyncio.run(mgr.handle(req))
    elapsed = time.perf_counter() - start

    assert elapsed > 0.4, f"Sequential too fast: {elapsed:.2f}s, expected > 0.4s"

    print(f"\n  SEQUENTIAL baseline: {elapsed:.3f}s (3 × 0.2s)")


# ---------------------------------------------------------------------------
# Fallback + retry tests
# ---------------------------------------------------------------------------


def test_fallback_on_failure():
    """If primary provider fails, fallback provider is used."""
    mgr = LLMManager()
    mgr.providers = {
        "ollama": FailingProvider(),  # always fails
        "claude": SlowProvider("claude", 0.05),  # succeeds
    }

    req = make_request(mode="single", provider="ollama")
    result = asyncio.run(mgr.handle(req))

    # Should fallback to claude
    assert result.provider == "claude"
    assert result.error is None
    assert "claude" in result.response

    print(f"\n  FALLBACK confirmed: ollama → claude, result={result.provider}")


def test_retry_then_success():
    """Provider that fails once but succeeds on retry.

    The backoff sleep is patched to zero so the test stays fast.
    """
    mgr = LLMManager()
    recovering = RecoveringProvider()
    mgr.providers = {
        "ollama": recovering,
        "claude": SlowProvider("claude", 0.05),
    }
    mgr.router.select_provider = lambda provider=None: "ollama"
    mgr.router.select_model = lambda task=None: "test"

    req = make_request(mode="single")

    with patch("neron_llm.core.manager.asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None  # instant
        result = asyncio.run(mgr.handle(req))

    assert result.error is None
    assert result.provider == "ollama"
    assert recovering.call_count == 2       # failed once, succeeded on retry
    assert mock_sleep.call_count == 1       # slept exactly once (between attempts)
    wait_called = mock_sleep.call_args[0][0]
    assert wait_called > 0                  # backoff is non-zero

    print(f"\n  RETRY confirmed: 2 attempts, backoff={wait_called:.2f}s (patched to 0)")


def test_retry_no_sleep_after_last_attempt():
    """Sleep must NOT be called after the final failed attempt (wasted wait).

    Calls _call_with_retry directly to isolate retry logic from the
    model/provider fallback chain in _execute_single.
    """
    mgr = LLMManager()
    mgr.providers = {"ollama": FailingProvider()}

    with patch("neron_llm.core.manager.asyncio.sleep") as mock_sleep:
        mock_sleep.return_value = None
        asyncio.run(mgr._call_with_retry("ollama", "test message", "test-model"))

    # MAX_RETRIES=2 → 2 attempts, 1 sleep between them, 0 sleep after last
    assert mock_sleep.call_count == MAX_RETRIES - 1

    print(f"\n  NO POST-LAST-ATTEMPT SLEEP confirmed: sleep called {mock_sleep.call_count}x")


def test_all_providers_fail():
    """When all providers fail, error is returned."""
    mgr = LLMManager()
    mgr.providers = {
        "ollama": FailingProvider(),
        "claude": FailingProvider(),
    }

    req = make_request(mode="single", provider="ollama")

    with patch("neron_llm.core.manager.asyncio.sleep"):
        result = asyncio.run(mgr.handle(req))

    assert result.error is not None
    assert "error" in result.error.lower() or "failed" in result.error.lower()

    print(f"\n  ALL-FAIL handled: error={result.error}")


def test_parallel_tolerates_failing_provider():
    """In parallel mode, a failing provider doesn't crash others."""
    mgr = LLMManager()
    mgr.providers = {
        "good": SlowProvider("good", 0.1),
        "failing": FailingProvider(),
    }

    req = make_request(mode="parallel")
    result = asyncio.run(mgr.handle(req))

    assert result.error is None
    assert "good" in result.response

    print(f"\n  PARALLEL tolerance: good provider result returned")


# ---------------------------------------------------------------------------
# Standardized response format tests
# ---------------------------------------------------------------------------


def test_response_format():
    """Every response follows the LLMResponse format."""
    mgr = LLMManager()
    mgr.providers = {"ollama": SlowProvider("ollama", 0.05)}
    mgr.router.select_provider = lambda provider=None: "ollama"
    mgr.router.select_model = lambda task=None: "test-model"

    for mode in ["single", "parallel", "race"]:
        req = make_request(mode=mode)
        result = asyncio.run(mgr.handle(req))

        assert isinstance(result, LLMResponse)
        assert hasattr(result, "model")
        assert hasattr(result, "provider")
        assert hasattr(result, "response")
        assert hasattr(result, "error")

    print(f"\n  RESPONSE FORMAT consistent across all modes")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


def test_auth_disabled_when_no_env_var():
    """When NERON_API_KEY is not set, all requests pass through."""
    import importlib
    import os
    from unittest.mock import patch, AsyncMock

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("NERON_API_KEY", None)

        # Re-import routes with no key set
        import api.routes as routes_mod
        importlib.reload(routes_mod)

        assert routes_mod._NERON_API_KEY == ""

        # _require_api_key must not raise when no key is configured
        result = asyncio.run(routes_mod._require_api_key(key=None))
        assert result is None

    print("\n  AUTH DISABLED: no key set → requests pass through")


def test_auth_rejects_missing_key():
    """When NERON_API_KEY is set, a missing header returns 403."""
    import importlib
    import os

    with patch.dict(os.environ, {"NERON_API_KEY": "secret-test-key"}):
        import api.routes as routes_mod
        importlib.reload(routes_mod)

        try:
            asyncio.run(routes_mod._require_api_key(key=None))
            assert False, "Should have raised HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 403

    print("\n  AUTH: missing key → 403")


def test_auth_rejects_wrong_key():
    """Wrong key returns 403."""
    import importlib
    import os

    with patch.dict(os.environ, {"NERON_API_KEY": "secret-test-key"}):
        import api.routes as routes_mod
        importlib.reload(routes_mod)

        try:
            asyncio.run(routes_mod._require_api_key(key="wrong-key"))
            assert False, "Should have raised HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 403

    print("\n  AUTH: wrong key → 403")


def test_auth_accepts_correct_key():
    """Correct key passes through."""
    import importlib
    import os

    with patch.dict(os.environ, {"NERON_API_KEY": "secret-test-key"}):
        import api.routes as routes_mod
        importlib.reload(routes_mod)

        result = asyncio.run(routes_mod._require_api_key(key="secret-test-key"))
        assert result is None

    print("\n  AUTH: correct key → 200")


# ---------------------------------------------------------------------------
# Input validation tests (P1.4)
# ---------------------------------------------------------------------------


def test_generate_request_rejects_empty_prompt():
    """Empty prompt must be rejected by Pydantic."""
    from pydantic import ValidationError
    from core.types import GenerateRequest

    try:
        GenerateRequest(task_type="chat", prompt="")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print("\n  VALIDATION: empty prompt → ValidationError")


def test_generate_request_rejects_oversized_prompt():
    """Prompt exceeding PROMPT_MAX_LEN must be rejected."""
    from pydantic import ValidationError
    from core.types import GenerateRequest, PROMPT_MAX_LEN

    try:
        GenerateRequest(task_type="chat", prompt="x" * (PROMPT_MAX_LEN + 1))
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print(f"\n  VALIDATION: prompt > {PROMPT_MAX_LEN} chars → ValidationError")


def test_generate_request_accepts_max_prompt():
    """Prompt exactly at the limit must be accepted."""
    from core.types import GenerateRequest, PROMPT_MAX_LEN

    req = GenerateRequest(task_type="chat", prompt="x" * PROMPT_MAX_LEN)
    assert len(req.prompt) == PROMPT_MAX_LEN

    print(f"\n  VALIDATION: prompt = {PROMPT_MAX_LEN} chars → accepted")


def test_generate_request_rejects_oversized_context():
    """Context dict exceeding CONTEXT_MAX_KEYS must be rejected."""
    from pydantic import ValidationError
    from core.types import GenerateRequest, CONTEXT_MAX_KEYS

    try:
        big_context = {str(i): "v" for i in range(CONTEXT_MAX_KEYS + 1)}
        GenerateRequest(task_type="chat", prompt="test", context=big_context)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print(f"\n  VALIDATION: context > {CONTEXT_MAX_KEYS} keys → ValidationError")


def test_llm_request_rejects_empty_message():
    """Legacy LLMRequest also validates message length."""
    from pydantic import ValidationError
    from core.types import LLMRequest

    try:
        LLMRequest(message="")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    print("\n  VALIDATION: empty LLMRequest.message → ValidationError")


# ---------------------------------------------------------------------------
# Reload lock tests (P1.6)
# ---------------------------------------------------------------------------


def test_reload_closes_old_manager():
    """Old manager's aclose() must be called after a successful reload."""
    import importlib
    from unittest.mock import AsyncMock, patch

    import api.routes as routes_mod
    importlib.reload(routes_mod)

    old_manager = routes_mod.manager
    old_manager.aclose = AsyncMock()

    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as client:
        resp = client.post("/llm/reload")

    assert resp.status_code == 200
    old_manager.aclose.assert_awaited_once()

    print("\n  RELOAD: old manager.aclose() called ✓")


def test_reload_keeps_old_manager_on_failure():
    """If new manager construction fails, old manager must be preserved."""
    import importlib
    from unittest.mock import patch

    import api.routes as routes_mod
    importlib.reload(routes_mod)

    original_manager = routes_mod.manager

    with patch("neron_llm.api.routes.LLMManager", side_effect=RuntimeError("bad config")):
        from fastapi.testclient import TestClient
        from app import app

        with TestClient(app) as client:
            resp = client.post("/llm/reload")

    assert resp.status_code == 500
    # Manager must be unchanged after failed reload
    assert routes_mod.manager is original_manager

    print("\n  RELOAD: failed reload preserves old manager ✓")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_strategy_explicit_mode()
    test_strategy_task_based()
    test_strategy_default()
    test_router_select_model()
    test_router_select_provider()
    test_router_fallback_chain()
    test_parallel_execution()
    test_race_execution()
    test_single_execution()
    test_sequential_baseline()
    test_fallback_on_failure()
    test_retry_then_success()
    test_retry_no_sleep_after_last_attempt()
    test_all_providers_fail()
    test_parallel_tolerates_failing_provider()
    test_response_format()
    test_auth_disabled_when_no_env_var()
    test_auth_rejects_missing_key()
    test_auth_rejects_wrong_key()
    test_auth_accepts_correct_key()
    test_generate_request_rejects_empty_prompt()
    test_generate_request_rejects_oversized_prompt()
    test_generate_request_accepts_max_prompt()
    test_generate_request_rejects_oversized_context()
    test_llm_request_rejects_empty_message()
    test_reload_closes_old_manager()
    test_reload_keeps_old_manager_on_failure()
    print("\n  All tests passed — neron_llm v2.0 is production-ready.")
