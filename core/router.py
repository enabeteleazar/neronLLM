# neron_llm/core/router.py
# Intelligent LLM router — sélectionne modèle/provider(s) à partir de la
# config par tâche (neron.yaml → tasks:), avec plancher de sécurité.

from __future__ import annotations
import logging
from llm.config import (
    get_tasks_config,
    get_providers_allowed,
    get_safety_floor_provider,
)

logger = logging.getLogger("neron_llm.router")

# Providers réellement instanciés dans LLMManager (voir manager.py).
# "codex" peut être présent dans providers_allowed (préparé pour le futur)
# sans qu'aucune classe Provider n'existe encore — le sélectionner retombe
# donc systématiquement sur le plancher, jamais une KeyError silencieuse.
IMPLEMENTED_PROVIDERS = {"ollama", "llama_cpp", "claude"}

# Chaîne de repli sur le MODÈLE (pas le provider) en cas d'échec — inchangée,
# ne traverse jamais de frontière de provider à elle seule.
MODEL_FALLBACK_CHAIN: list[str] = [
    "qwen2.5-coder:14b",
    "deepseek-coder:6.7b",
    "llama3.2:1b",
]
FALLBACK_MODEL = "deepseek-coder:6.7b"

_DEFAULT_TASK: dict = {"providers": ["ollama"], "model": FALLBACK_MODEL, "mode": "single"}


class LLMRouter:
    def __init__(self) -> None:
        self._tasks = get_tasks_config()
        self._allowed = set(get_providers_allowed())

        floor = get_safety_floor_provider()
        if floor != "ollama":
            logger.error(
                "safety_floor_provider=%r n'est pas 'ollama' — repli forcé "
                "sur 'ollama' quoi qu'il arrive. Le plancher de sécurité "
                "n'est PAS configurable vers un provider externe, même "
                "valide/implémenté (décision explicite : le plancher est "
                "toujours local).",
                floor,
            )
            floor = "ollama"
        self._floor = floor

        logger.debug(
            "Router initialized — tasks=%s allowed=%s floor=%s",
            sorted(self._tasks), self._allowed, self._floor,
        )

    # ── Résolution de la config d'une tâche ───────────────────────────────

    def _task_cfg(self, task: str | None) -> dict:
        if task and task in self._tasks and isinstance(self._tasks[task], dict):
            return self._tasks[task]
        default = self._tasks.get("default")
        return default if isinstance(default, dict) else _DEFAULT_TASK

    # ── Sélection du modèle ────────────────────────────────────────────────

    def select_model(self, task: str | None = None) -> str:
        model = self._task_cfg(task).get("model")
        if model:
            return str(model)
        logger.warning("Aucun modèle configuré pour task=%r — repli %s", task, FALLBACK_MODEL)
        return FALLBACK_MODEL

    # ── Sélection du/des provider(s) ───────────────────────────────────────

    def providers_for(self, task: str | None = None, explicit: str | None = None) -> list[str]:
        """Liste ordonnée des providers utilisables pour cette tâche.

        Chaque candidat doit passer DEUX filtres : présent dans
        providers_allowed (liste blanche config) ET dans
        IMPLEMENTED_PROVIDERS (réellement câblé dans le code). Un candidat
        qui échoue est retiré (avec log), jamais bloquant pour les autres.
        Si la liste résultante est vide, repli sur [safety_floor_provider]
        SANS repasser par ces filtres (le plancher est déjà validé à l'init).
        """
        cfg = self._task_cfg(task)
        raw = list(cfg.get("providers") or [])
        if not raw and cfg.get("provider"):
            raw = [cfg["provider"]]  # compat forme singulière

        if explicit:
            raw = [explicit] + [p for p in raw if p != explicit]

        result: list[str] = []
        for candidate in raw:
            if candidate not in self._allowed:
                logger.warning(
                    "provider=%r hors de providers_allowed (task=%r) — ignoré",
                    candidate, task,
                )
                continue
            if candidate not in IMPLEMENTED_PROVIDERS:
                logger.warning(
                    "provider=%r autorisé en config mais pas encore implémenté "
                    "(task=%r) — ignoré, en attente d'une classe Provider",
                    candidate, task,
                )
                continue
            if candidate not in result:
                result.append(candidate)

        if not result:
            logger.warning(
                "Aucun provider valide pour task=%r (config brute=%s) — "
                "repli sur le plancher de sécurité %r",
                task, raw, self._floor,
            )
            result = [self._floor]

        return result

    def select_provider(self, task: str | None = None, provider: str | None = None) -> str:
        """Compat usage 'single' : premier provider de la liste résolue."""
        return self.providers_for(task=task, explicit=provider)[0]

    def get_fallback_provider(self, current: str, task: str | None = None) -> str | None:
        """Provider suivant dans la chaîne PROPRE À LA TÂCHE — plus de
        chaîne globale : chaque tâche ne peut retomber que sur un provider
        qu'elle a elle-même autorisé."""
        chain = self.providers_for(task=task)
        try:
            idx = chain.index(current)
        except ValueError:
            return None
        return chain[idx + 1] if idx + 1 < len(chain) else None

    def get_fallback_model(self, current_model: str) -> str | None:
        try:
            idx = MODEL_FALLBACK_CHAIN.index(current_model)
            if idx + 1 < len(MODEL_FALLBACK_CHAIN):
                return MODEL_FALLBACK_CHAIN[idx + 1]
        except ValueError:
            if MODEL_FALLBACK_CHAIN:
                return MODEL_FALLBACK_CHAIN[-1]
        return None
