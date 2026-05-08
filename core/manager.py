# neron_llm/core/manager.py
# LLM Manager — orchestration engine with single/parallel/race modes.

from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from llm.core.router   import LLMRouter
from llm.core.strategy import StrategyEngine
from llm.core.types    import LLMRequest, LLMResponse
from llm.providers.base   import BaseProvider
from llm.providers.claude import ClaudeProvider
from llm.providers.ollama import OllamaProvider

logger      = logging.getLogger("llm.manager")
MAX_RETRIES      = 2
RETRY_BASE_DELAY = 1.0   # secondes — délai avant la 2e tentative
RETRY_MAX_DELAY  = 10.0  # secondes — plafond (croissance exponentielle)
RETRY_JITTER     = 0.3   # secondes — aléatoire ajouté pour éviter les rafales


class LLMManager:
    """Orchestrates LLM calls across providers with strategy and fallback."""

    def __init__(self) -> None:
        self.router   = LLMRouter()
        self.strategy = StrategyEngine()
        self.providers: dict[str, BaseProvider] = {
            "ollama": OllamaProvider(),
            "claude": ClaudeProvider(),
        }

    async def aclose(self) -> None:
        """Close all provider HTTP clients gracefully on shutdown."""
        for name, provider in self.providers.items():
            try:
                await provider.aclose()
                logger.debug(json.dumps({"event": "provider_closed", "provider": name}))
            except Exception as exc:
                logger.warning(
                    json.dumps({"event": "provider_close_error", "provider": name, "error": str(exc)})
                )

    # ── Main entry point ──────────────────────────────────────────────────────

    async def handle(self, request: LLMRequest) -> LLMResponse:
        """Route, strategize, and execute the request."""
        mode          = self.strategy.decide(task=request.task, mode=request.mode)
        model         = request.model or self.router.select_model(task=request.task)
        provider_name = self.router.select_provider(provider=request.provider)

        logger.info(
            json.dumps({
                "event":    "llm_handle",
                "task":     request.task,
                "mode":     mode,
                "model":    model,
                "provider": provider_name,
            })
        )

        if mode == "parallel":
            return await self._execute_parallel(request, model)
        elif mode == "race":
            return await self._execute_race(request, model)
        else:
            return await self._execute_single(request, model, provider_name)

    # ── Mode SINGLE — model fallback then provider fallback ───────────────────

    async def _execute_single(
        self, request: LLMRequest, model: str, provider_name: str,
    ) -> LLMResponse:
        """Execute with one provider.

        Fallback order:
          1. retry (×MAX_RETRIES) on primary model
          2. switch to next model in chain (deepseek-coder:6.7b)
          3. try next provider in chain (ollama → claude)
        """
        # Attempt primary model
        result = await self._call_with_retry(provider_name, request.message, model)
        if result.error is None:
            return result

        # Model-level fallback
        fallback_model = self.router.get_fallback_model(model)
        if fallback_model:
            logger.warning(
                json.dumps({
                    "event":          "model_fallback",
                    "primary_model":  model,
                    "fallback_model": fallback_model,
                    "provider":       provider_name,
                })
            )
            result = await self._call_with_retry(provider_name, request.message, fallback_model)
            if result.error is None:
                return result

        # Provider-level fallback
        fallback_provider = self.router.get_fallback_provider(provider_name)
        if fallback_provider and fallback_provider in self.providers:
            logger.warning(
                json.dumps({
                    "event":             "provider_fallback",
                    "primary_provider":  provider_name,
                    "fallback_provider": fallback_provider,
                })
            )
            result = await self._call_with_retry(
                fallback_provider, request.message, fallback_model or model
            )

        return result

    # ── Mode PARALLEL — all providers, pick best ──────────────────────────────

    async def _execute_parallel(
        self, request: LLMRequest, model: str,
    ) -> LLMResponse:
        """Execute on all providers in parallel; return longest valid response."""
        tasks = [
            self._call_provider(name, provider, request.message, model)
            for name, provider in self.providers.items()
        ]
        results = await asyncio.gather(*tasks)

        valid = [r for r in results if r.error is None]
        if not valid:
            logger.error(json.dumps({"event": "parallel_all_failed"}))
            return results[0] if results else LLMResponse(
                model=model, provider="none", response="", error="All providers failed",
            )

        best = max(valid, key=lambda r: len(r.response))
        logger.info(
            json.dumps({
                "event":    "parallel_best",
                "provider": best.provider,
                "model":    best.model,
                "length":   len(best.response),
            })
        )
        return best

    # ── Mode RACE — first completed wins ─────────────────────────────────────

    async def _execute_race(
        self, request: LLMRequest, model: str,
    ) -> LLMResponse:
        """Execute on all providers; return first to complete successfully.

        Uses race_timeout (default 30s) instead of the full generation timeout
        to ensure fast failure and avoid blocking the event loop.
        """
        tasks: dict[asyncio.Task, str] = {}
        for name, provider in self.providers.items():
            # Skip providers that aren't configured
            if hasattr(provider, 'is_available') and not provider.is_available():
                continue
            # Pass race_timeout so providers fail fast instead of blocking for 300s
            race_timeout = getattr(provider, "_timeout_race", None)
            task = asyncio.create_task(
                self._call_provider(name, provider, request.message, model,
                                    timeout=race_timeout)
            )
            tasks[task] = name

        done, pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
        logger.debug(
            json.dumps({
                "event":     "race_done",
                "cancelled": [tasks[t] for t in pending],
            })
        )

        for task in done:
            result = task.result()
            if isinstance(result, LLMResponse) and result.error is None:
                logger.info(json.dumps({"event": "race_winner", "provider": result.provider}))
                return result

        # All done tasks had errors
        for task in done:
            result = task.result()
            if isinstance(result, LLMResponse):
                return result

        return LLMResponse(model=model, provider="none", response="", error="Race: all providers failed")

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    async def _call_with_retry(
        self, provider_name: str, message: str, model: str,
    ) -> LLMResponse:
        provider = self.providers.get(provider_name)
        if not provider:
            return LLMResponse(
                model=model, provider=provider_name, response="",
                error=f"Unknown provider: {provider_name}",
            )

        last_result: LLMResponse | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            t0     = time.monotonic()
            result = await self._call_provider(provider_name, provider, message, model)
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            if result.error is None:
                return result

            # Délai avant la prochaine tentative.
            # Formule : min(base × 2^(attempt-1) + jitter, max)
            #   attempt 1 → ~1.0s + jitter
            #   attempt 2 → ~2.0s + jitter
            # Pas de sleep après la dernière tentative — inutile d'attendre
            # avant de passer au fallback ou de retourner l'erreur.
            will_retry = attempt < MAX_RETRIES
            wait_s = (
                min(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER),
                    RETRY_MAX_DELAY)
                if will_retry else 0.0
            )

            logger.warning(
                json.dumps({
                    "event":      "provider_retry",
                    "provider":   provider_name,
                    "model":      model,
                    "attempt":    attempt,
                    "max":        MAX_RETRIES,
                    "error":      result.error,
                    "elapsed_ms": elapsed_ms,
                    "wait_ms":    int(wait_s * 1000),
                })
            )
            last_result = result

            if will_retry:
                await asyncio.sleep(wait_s)

        return last_result or LLMResponse(
            model=model, provider=provider_name, response="",
            error=f"Provider '{provider_name}' failed after {MAX_RETRIES} attempts",
        )

    async def _call_provider(
        self, name: str, provider: BaseProvider, message: str, model: str,
        timeout: float | None = None,
    ) -> LLMResponse:
        try:
            response = await provider.generate(message, model, timeout=timeout)
            return LLMResponse(model=model, provider=name, response=response, error=None)
        except Exception as exc:
            exc_type = type(exc).__name__
            # str(exc) is empty for httpx timeout/network exceptions — use repr fallback
            exc_msg  = str(exc) or repr(exc)
            # Include request URL for httpx exceptions (have a .request attribute)
            request_url = str(getattr(getattr(exc, "request", None), "url", ""))

            logger.error(
                json.dumps({
                    "event":    "provider_error",
                    "provider": name,
                    "model":    model,
                    "exc_type": exc_type,
                    "error":    exc_msg,
                    **({"request_url": request_url} if request_url else {}),
                })
            )
            error_str = f"{exc_type}: {exc_msg}" if exc_msg else exc_type
            return LLMResponse(model=model, provider=name, response="", error=error_str)
