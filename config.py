# Configuration loader with memory cache for llm.

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from common.paths import NERON_CONFIG

logger = logging.getLogger("llm.config")

CONFIG_PATH = NERON_CONFIG

def load_config() -> dict:
    # Load and cache the full YAML configuration.# 
    return _load_config_cached()


@lru_cache(maxsize=1)
def _load_config_cached() -> dict:
    # Load YAML config once and cache in memory.# 
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Configuration loaded from %s", CONFIG_PATH)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", CONFIG_PATH)
        return {}
    except yaml.YAMLError as e:
        logger.error("YAML parse error: %s", e)
        return {}


def get_llm_config() -> dict:
    # Get the 'llm' section of the config.# 
    return load_config().get("llm", {})


def get_tasks_config() -> dict[str, dict]:
    """Section 'tasks:' de neron.yaml — un bloc {provider(s)/model/mode} par
    tâche. Remplace les anciennes sections model_map/routing/strategy
    (fusionnées ici, un seul endroit à lire par tâche)."""
    tasks = load_config().get("tasks", {})
    return tasks if isinstance(tasks, dict) else {}


def get_providers_allowed() -> list[str]:
    """Liste blanche des providers utilisables, TOUS usages confondus.
    Un provider absent de cette liste est ignoré par le router, quelle
    que soit la config de la tâche — c'est la première barrière avant
    le plancher de sécurité."""
    allowed = get_llm_config().get("providers_allowed")
    if isinstance(allowed, list) and allowed:
        return [str(p) for p in allowed]
    return ["ollama"]


def get_safety_floor_provider() -> str:
    """Provider utilisé quand une tâche n'a pas de config valide, ou que
    son provider configuré est invalide/non implémenté. Ne doit JAMAIS
    être un provider externe — validé une seconde fois dans router.py."""
    return str(get_llm_config().get("safety_floor_provider") or "ollama")


def reload_config() -> dict:
    # Force reload the configuration (clears LRU cache).# 
    _load_config_cached.cache_clear()
    new_config = load_config()
    logger.info("Configuration reloaded from %s", CONFIG_PATH)
    return new_config


def get_core_url() -> str:
    """Core registry URL — deprecated shim, delegates to llm.settings."""
    from llm.settings import get_settings
    return get_settings().core_url


