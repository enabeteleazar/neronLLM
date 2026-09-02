# llama.cpp provider — async HTTP via a shared httpx.AsyncClient.
#
# Cible : llama.cpp server (--server mode), API OpenAI-compatible
# /v1/chat/completions.

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from llm.config import get_llm_config
from llm.providers.base import BaseProvider


logger = logging.getLogger("neron_llm.llama_cpp")

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


class LlamaCppProvider(BaseProvider):
    """Provider asynchrone pour un serveur llama.cpp local."""

    def __init__(self) -> None:
        cfg = get_llm_config()

        host = str(cfg.get("llama_cpp_host", "")).rstrip("/")

        if not host:
            logger.warning(
                "LlamaCppProvider: 'llama_cpp_host' absent de la config "
                "— le provider sera marqué indisponible."
            )

        self._host = host

        # Paramètres de génération.
        self._n_predict = int(
            cfg.get("llama_cpp_n_predict", 512)
        )

        self._temperature = float(
            cfg.get("llama_cpp_temperature", 0.7)
        )

        self._stop = cfg.get(
            "llama_cpp_stop",
            ["</s>", "User:", "Assistant:"],
        )

        # Timeouts.
        self._timeout_default = float(
            cfg.get("timeout", 300)
        )

        self._timeout_race = float(
            cfg.get("race_timeout", 60)
        )

        limits = httpx.Limits(
            max_connections=int(
                cfg.get("llama_cpp_max_connections", 4)
            ),
            max_keepalive_connections=int(
                cfg.get(
                    "llama_cpp_max_keepalive_connections",
                    2,
                )
            ),
        )

        self._client = httpx.AsyncClient(
            base_url=host or "http://localhost:8080",
            timeout=None,
            limits=limits,
        )

        logger.debug(
            "LlamaCppProvider initialisé — "
            "host=%s n_predict=%s temperature=%s "
            "timeout=%s race_timeout=%s",
            host or "<non configuré>",
            self._n_predict,
            self._temperature,
            self._timeout_default,
            self._timeout_race,
        )

    def is_available(self) -> bool:
        """Retourne True si llama.cpp est configuré."""
        return bool(self._host)

    async def generate(
        self,
        message: str,
        model: str | None = None,
        timeout: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Génère une réponse via llama.cpp
        /v1/chat/completions.
        """

        if not self._host:
            raise ValueError(
                "LlamaCppProvider: llama_cpp_host "
                "non configuré dans neron.yaml"
            )

        effective_timeout = (
            timeout
            if timeout is not None
            else self._timeout_default
        )

        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "max_tokens": self._n_predict,
            "temperature": self._temperature,
            "stream": False,
        }

        if self._stop:
            payload["stop"] = self._stop

        if json_mode:
            payload["response_format"] = {
                "type": "json_object"
            }

        logger.warning(
            "llama_cpp TRACE | json_mode=%s "
            "n_predict=%s timeout=%s payload_keys=%s",
            json_mode,
            self._n_predict,
            effective_timeout,
            list(payload.keys()),
        )

        response = await self._client.post(
            _CHAT_COMPLETIONS_ENDPOINT,
            json=payload,
            timeout=effective_timeout,
        )

        response.raise_for_status()

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "LlamaCpp réponse inattendue — format "
                "/v1/chat/completions invalide"
            ) from exc

        if not isinstance(content, str):
            raise ValueError(
                "LlamaCpp contenu inattendu — "
                f"type={type(content).__name__}"
            )

        if not json_mode:
            return content

        raw = content.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            parsed = json.loads(cleaned)
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            parsed = None

        if parsed is None:
            start_obj = cleaned.find("{")
            end_obj = cleaned.rfind("}")

            if start_obj >= 0 and end_obj > start_obj:
                candidate = cleaned[
                    start_obj:end_obj + 1
                ]

                try:
                    parsed = json.loads(candidate)
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    parsed = None

        if parsed is None:
            start_arr = cleaned.find("[")
            end_arr = cleaned.rfind("]")

            if start_arr >= 0 and end_arr > start_arr:
                candidate = cleaned[
                    start_arr:end_arr + 1
                ]

                try:
                    parsed = json.loads(candidate)
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    parsed = None

        if parsed is None:
            logger.warning(
                "llama_cpp | json_mode réponse invalide: %r",
                content[:500],
            )

            raise ValueError(
                "LlamaCpp JSON mode — "
                "réponse JSON invalide"
            )

        return json.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def aclose(self) -> None:
        """Ferme le client HTTP partagé."""
        await self._client.aclose()

        logger.debug(
            "LlamaCppProvider: HTTP client fermé"
        )
