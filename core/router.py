# neron_llm/core/router.py
# Intelligent LLM router — selects model and provider based on config.

from __future__ import annotations
import logging
from config import get_llm_config, get_routing_config

logger = logging.getLogger("neron_llm.router")

# ── Defaults ──────────────────────────────────────────────────────────────────

# Used when task is unknown AND no 'default' key exists in config
FALLBACK_MODEL = "deepseek-coder:6.7b"

# Ordered fallback chain — first model in list is preferred
MODEL_FALLBACK_CHAIN: list[str] = [
    "qwen2.5-coder:14b",
    "deepseek-coder:6.7b",
    "llama3.2:1b",
]

# Provider fallback chain (unchanged from v1)
PROVIDER_CHAIN: list[str] = ["ollama", "claude"]

# Built-in task → model defaults (overridable via neron.yaml → routing:)
_DEFAULT_TASK_ROUTING: dict[str, str] = {
    "code":      "qwen2.5-coder:14b",
    "reasoning": "qwen2.5-coder:14b",
    "agent":     "qwen2.5-coder:14b",
    "chat":      "llama3.2:1b",
    "fast":      "llama3.2:1b",
    "summary":   "llama3.2:1b",
    "default":   "llama3.2:1b",
}


class LLMRouter:

    def __init__(self) -> None:
        # Config-based routing overrides built-in defaults
        self._config_routing = get_routing_config()   # dict from neron.yaml
        self._llm_config     = get_llm_config()

        # Merge: config wins over built-in defaults
        self._routing: dict[str, str] = {**_DEFAULT_TASK_ROUTING, **self._config_routing}
        logger.debug("Router initialized — routing table: %s", self._routing)

    # ── Model selection ───────────────────────────────────────────────────────

    def select_model(self, task: str | None = None) -> str:
        if task and task in self._routing:
            model = self._routing[task]
            logger.debug("Router: task='%s' → model='%s'", task, model)
            return model

        model = self._routing.get("default", FALLBACK_MODEL)
        logger.debug("Router: unknown task='%s' → default model='%s'", task, model)
        return model

    def get_fallback_model(self, current_model: str) -> str | None:
        """Return the next model in the fallback chain after current_model.

        Returns None if current_model is already the last in the chain.
        """
        try:
            idx = MODEL_FALLBACK_CHAIN.index(current_model)
            if idx + 1 < len(MODEL_FALLBACK_CHAIN):
                next_model = MODEL_FALLBACK_CHAIN[idx + 1]
                logger.debug("Router: model fallback %s → %s", current_model, next_model)
                return next_model
        except ValueError:
            # current_model not in chain — return first fallback
            if MODEL_FALLBACK_CHAIN:
                return MODEL_FALLBACK_CHAIN[-1]
        return None

    # ── Provider selection ────────────────────────────────────────────────────

    def select_provider(self, provider: str | None = None) -> str:
        """Select provider. Priority: explicit > config default > 'ollama'."""
        if provider:
            logger.debug("Router: explicit provider='%s'", provider)
            return provider
        default = self._llm_config.get("default_provider", "ollama")
        logger.debug("Router: default provider='%s'", default)
        return default

    def get_fallback_provider(self, current: str) -> str | None:
        """Return the next provider in the fallback chain after current."""
        try:
            idx = PROVIDER_CHAIN.index(current)
            if idx + 1 < len(PROVIDER_CHAIN):
                next_provider = PROVIDER_CHAIN[idx + 1]
                logger.debug("Router: provider fallback %s → %s", current, next_provider)
                return next_provider
        except ValueError:
            pass
        return None
