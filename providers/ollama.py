# Ollama provider — async HTTP via a shared httpx.AsyncClient.

from __future__ import annotations

import logging

import httpx

from llm.config import get_llm_config
from llm.providers.base import BaseProvider

logger = logging.getLogger("neron_llm.ollama")


class ProviderBusyError(RuntimeError):
    """Levée quand Ollama refuse la requête (HTTP 429) car son unique
    créneau de génération est déjà occupé par une autre requête en cours.
    Distinguée des autres erreurs pour permettre un backoff bien plus long
    côté LLMManager (une génération peut durer 60-100s+ sur ce matériel)."""


class OllamaProvider(BaseProvider):
    """Async provider for local Ollama instances.

    Fonctionnalités :
    - appel HTTP async vers Ollama
    - résolution automatique du modèle
    - fallback vers un modèle local disponible si le modèle demandé est absent
    - option auto_pull désactivée par défaut
    """

    DEFAULT_FALLBACK_MODELS: list[str] = [
        "tinyllama:latest",
        "phi3:mini",
        "Qwen2.5-Coder:latest",
    ]

    def __init__(self) -> None:
        cfg = get_llm_config()

        host = cfg.get("host", "http://localhost:11434").rstrip("/")

        limits = httpx.Limits(
            max_connections=int(cfg.get("ollama_max_connections", 50)),
            max_keepalive_connections=int(cfg.get("ollama_max_keepalive_connections", 10)),
        )

        self._timeout_default: float = float(cfg.get("timeout", 300))
        self._timeout_race: float = float(cfg.get("race_timeout", 240))
        self._auto_pull: bool = str(cfg.get("auto_pull", "false")).lower() == "true"
        # Cf. commentaire neron.yaml → llm.keep_alive : évite le rechargement
        # à froid d'un modèle inactif depuis >5 min (défaut Ollama), tout en
        # restant modeste vu la RAM limitée de Homebox (7.2 Gi, cf. `free -h`).
        self._keep_alive: str = str(cfg.get("keep_alive", "10m"))

        raw_fallbacks = cfg.get("fallback_models", None)
        if isinstance(raw_fallbacks, list):
            self._fallback_models = [str(m) for m in raw_fallbacks if str(m).strip()]
        else:
            self._fallback_models = self.DEFAULT_FALLBACK_MODELS

        self._client = httpx.AsyncClient(
            base_url=host,
            timeout=None,
            limits=limits,
        )

        logger.debug(
            "OllamaProvider initialised — base_url=%s default_timeout=%s race_timeout=%s auto_pull=%s fallbacks=%s",
            host,
            self._timeout_default,
            self._timeout_race,
            self._auto_pull,
            self._fallback_models,
        )

    async def _list_models(self) -> list[str]:
        """Retourne la liste des modèles installés dans Ollama."""
        r = await self._client.get("/api/tags", timeout=30)
        r.raise_for_status()

        data = r.json()
        models: list[str] = []

        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                models.append(str(name))

        return models

    async def _pull_model(self, model: str) -> bool:
        """Télécharge un modèle Ollama si auto_pull est activé."""
        logger.warning("ollama | pulling missing model=%s", model)

        r = await self._client.post(
            "/api/pull",
            json={"name": model, "stream": False},
            timeout=1800,
        )

        if r.status_code >= 400:
            logger.warning("ollama | pull failed model=%s status=%s body=%s", model, r.status_code, r.text[:300])
            return False

        return True

    async def _resolve_model(self, requested_model: str) -> str:
        """Résout le modèle à utiliser.

        Priorité :
        1. modèle demandé s'il est installé
        2. pull automatique si auto_pull=true
        3. fallback configuré disponible
        4. premier modèle disponible dans Ollama
        5. erreur claire si aucun modèle
        """
        requested_model = (requested_model or "").strip()

        available = await self._list_models()

        if requested_model and requested_model in available:
            return requested_model

        if requested_model and self._auto_pull:
            pulled = await self._pull_model(requested_model)
            if pulled:
                available = await self._list_models()
                if requested_model in available:
                    logger.info("ollama | model pulled and resolved requested=%s", requested_model)
                    return requested_model

        for fallback in self._fallback_models:
            if fallback in available:
                logger.warning(
                    "ollama | requested model unavailable requested=%s fallback=%s available=%s",
                    requested_model,
                    fallback,
                    available,
                )
                return fallback

        if available:
            logger.warning(
                "ollama | no configured fallback available requested=%s using_first_available=%s",
                requested_model,
                available[0],
            )
            return available[0]

        raise RuntimeError(
            f"Aucun modèle Ollama disponible. Modèle demandé : {requested_model or 'non défini'}"
        )

    async def generate(self, message: str, model: str, timeout: float | None = None) -> str:
        """Generate a response via Ollama's /api/generate endpoint."""
        effective_timeout = timeout if timeout is not None else self._timeout_default
        resolved_model = await self._resolve_model(model)

        payload = {
            "model": resolved_model,
            "prompt": message,
            "stream": False,
            "keep_alive": self._keep_alive,
        }

        logger.debug(
            "ollama | POST /api/generate requested_model=%s resolved_model=%s timeout=%s keep_alive=%s",
            model,
            resolved_model,
            effective_timeout,
            self._keep_alive,
        )

        r = await self._client.post(
            "/api/generate",
            json=payload,
            timeout=effective_timeout,
        )
        if r.status_code == 429:
            raise ProviderBusyError(
                "Ollama a répondu 429 — une autre génération occupe déjà "
                "son unique créneau de calcul."
            )
        r.raise_for_status()

        data = r.json()

        if "response" not in data:
            raise ValueError(f"Ollama unexpected format: {list(data.keys())}")

        return data["response"]

    async def aclose(self) -> None:
        """Close the shared HTTP client and release connections."""
        await self._client.aclose()
        logger.debug("OllamaProvider: HTTP client closed")
