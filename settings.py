# neron_llm/settings.py
# Infrastructure settings — pydantic-settings, hydrated from environment.
#
# Split of responsibilities:
#   - settings.py  → infra (host, port, core_url, api_key) — env vars, systemd
#   - config.py    → comportement (routing, strategy, providers) — neron.yaml
#
# Env vars (all optional, sane defaults matching neron.server.yaml topology):
#   NERON_SERVICE_HOST, NERON_SERVICE_PORT, NERON_CORE_URL, NERON_API_KEY

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NERON_",
        extra="ignore",
        frozen=True,
    )

    service_host: str = Field(default="127.0.1.2")
    service_port: int = Field(default=8765, ge=1, le=65535)
    core_url: str = Field(default="http://127.0.1.1:8010")
    api_key: str = Field(default="")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> LLMSettings:
    """Cached settings instance. Frozen — reload requires process restart,
    which is correct for infra values (systemd owns them)."""
    return LLMSettings()
