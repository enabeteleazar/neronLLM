# neron_llm/core/strategy.py
# Strategy layer — décide du mode d'exécution à partir de tasks.<task>.mode.

from __future__ import annotations
import logging
from llm.config import get_tasks_config

logger = logging.getLogger("neron_llm.strategy")

_VALID_MODES = {"single", "parallel", "race"}
_FALLBACK_MODE = "single"


class StrategyEngine:
    def __init__(self) -> None:
        self._tasks = get_tasks_config()
        logger.debug("StrategyEngine initialized — tasks=%s", sorted(self._tasks))

    def decide(self, task: str | None = None, mode: str | None = None) -> str:
        """Priorité : mode explicite (si valide) > mode configuré pour la
        tâche (si valide) > 'single'. Un mode invalide, où qu'il vienne,
        ne fait jamais planter — il retombe sur 'single'."""
        if mode:
            if mode in _VALID_MODES:
                logger.debug("Strategy: mode explicite=%r", mode)
                return mode
            logger.warning("Strategy: mode explicite invalide=%r — repli %r", mode, _FALLBACK_MODE)
            return _FALLBACK_MODE

        cfg = self._tasks.get(task) if task else None
        configured = (cfg or {}).get("mode") if isinstance(cfg, dict) else None
        if configured in _VALID_MODES:
            logger.debug("Strategy: task=%r → mode=%r (config)", task, configured)
            return configured

        logger.debug("Strategy: task=%r sans mode valide configuré → %r", task, _FALLBACK_MODE)
        return _FALLBACK_MODE
