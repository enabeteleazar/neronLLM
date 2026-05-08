# llama.cpp provider — async HTTP via a shared httpx.AsyncClient.
#
# Cible : llama.cpp server (--server mode), endpoint natif /completion.
# Le serveur charge UN seul modèle au démarrage ; le paramètre 
# est transmis à titre informatif mais ignoré par l'API.
#
# Config attendue dans neron.yaml → llm:
#
#   llm:
#     llama_cpp_host:                     http://localhost:8080
#     llama_cpp_max_connections:          4       # default
#     llama_cpp_max_keepalive_connections: 2      # default
#     llama_cpp_n_predict:                512     # tokens max (-1 = illimité)
#     llama_cpp_temperature:              0.7
#     llama_cpp_stop:                     ["</s>", "User:", "Assistant:"]
#     timeout:                            300     # partagé avec les autres providers
#     race_timeout:                       60

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_llm_config
from providers.base import BaseProvider

logger = logging.getLogger("neron_llm.llama_cpp")

# Endpoint natif llama.cpp server
_COMPLETION_ENDPOINT = "/completion"


class LlamaCppProvider(BaseProvider):
    """Async provider for a local llama.cpp server instance.

    llama.cpp server expose une API HTTP légère sur /completion.
    Compatible avec des modèles GGUF quantifiés (Q4_K_M, Q5_K_M, etc.),
    idéal pour les machines à ressources limitées.

    Le client httpx est partagé sur toute la durée de vie de l'objet.
    Appeler aclose() (via LLMManager.aclose()) pour libérer les connexions.
    """

    def __init__(self) -> None:
        cfg = get_llm_config()

        host = cfg.get("llama_cpp_host", "").rstrip("/")
        if not host:
            logger.warning(
                "LlamaCppProvider: 'llama_cpp_host' absent de la config "
                "— le provider sera marqué indisponible."
            )

        self._host = host

        # Génération
        self._n_predict:   int   = int(cfg.get("llama_cpp_n_predict", 512))
        self._temperature: float = float(cfg.get("llama_cpp_temperature", 0.7))
        # Stop tokens par défaut — évite les sorties infinies sur des modèles bavards
        self._stop: list[str] = cfg.get("llama_cpp_stop", ["</s>", "User:", "Assistant:"])

        # Timeouts (partagés avec les autres providers via les clés globales)
        self._timeout_default: float = float(cfg.get("timeout", 300))
        self._timeout_race:    float = float(cfg.get("race_timeout", 60))

        limits = httpx.Limits(
            max_connections           = int(cfg.get("llama_cpp_max_connections", 4)),
            max_keepalive_connections = int(cfg.get("llama_cpp_max_keepalive_connections", 2)),
        )

        # Client créé même si host est vide — les appels échoueront proprement
        # avec une ConnectError que le manager transforme en LLMResponse(error=...).
        self._client = httpx.AsyncClient(
            base_url = host or "http://localhost:8080",
            timeout  = None,   # timeout défini par requête
            limits   = limits,
        )

        logger.debug(
            "LlamaCppProvider initialisé — host=%s n_predict=%s temperature=%s timeout=%s race_timeout=%s",
            host or "<non configuré>",
            self._n_predict,
            self._temperature,
            self._timeout_default,
            self._timeout_race,
        )

    def is_available(self) -> bool:
        """Retourne False si llama_cpp_host n'est pas configuré."""
        return bool(self._host)

    async def generate(self, message: str, model: str, timeout: float | None = None) -> str:
        """Génère une réponse via l'endpoint /completion de llama.cpp server.

        Args:
            message: Le prompt texte brut à envoyer.
            model:   Nom du modèle (informatif uniquement — le serveur
                     charge son modèle au démarrage, ce paramètre est logué
                     mais non transmis à l'API).
            timeout: Override du timeout pour cet appel.
                     Utilisé par le mode race pour un fail-fast rapide.

        Raises:
            ValueError:              Si le provider n'est pas configuré.
            httpx.TimeoutException:  Sur timeout.
            httpx.HTTPStatusError:   Sur HTTP 4xx/5xx.
            ValueError:              Si la réponse n'a pas le format attendu.
        """
        if not self._host:
            raise ValueError(
                "LlamaCppProvider: llama_cpp_host non configuré dans neron.yaml"
            )

        effective_timeout = timeout if timeout is not None else self._timeout_default

        payload: dict[str, Any] = {
            "prompt":      message,
            "n_predict":   self._n_predict,
            "temperature": self._temperature,
            "stop":        self._stop,
            "stream":      False,
        }

        logger.debug(
            "llama_cpp | POST %s model=%s (informatif) timeout=%s",
            _COMPLETION_ENDPOINT, model, effective_timeout,
        )

        r = await self._client.post(
            _COMPLETION_ENDPOINT,
            json    = payload,
            timeout = effective_timeout,
        )
        r.raise_for_status()
        data = r.json()

        # L'endpoint /completion retourne {"content": "...", "stop": true, ...}
        if "content" not in data:
            raise ValueError(
                f"LlamaCpp réponse inattendue — clés reçues : {list(data.keys())}"
            )

        return data["content"]

    async def aclose(self) -> None:
        """Ferme le client HTTP partagé et libère les connexions."""
        await self._client.aclose()
        logger.debug("LlamaCppProvider: HTTP client fermé")
