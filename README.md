## neronOS_LLM - v2.1.0


Microservice LLM unifié pour Néron. Abstrait les providers (Ollama, Claude) derrière une API FastAPI unique avec trois modes d'exécution.

## Modes

| Mode       	| Comportement 								|
|---         	|---									|
| `single`   	| Un seul provider, sélectionné par le router 				|
| `parallel` 	| Tous les providers en `asyncio.gather` — tous les résultats 		|
| `race`	| `asyncio.FIRST_COMPLETED` — le plus rapide gagne, les autres annulés 	|

## Structure

```
llm/
├── app.py                # Point d'entrée uvicorn
├── config.py             # Lecture /etc/neron/neron.yaml (lru_cache)
├── core/
│   ├── types.py          # LLMRequest / LLMResponse (Pydantic)
│   ├── router.py         # Routage tâche → modèle/provider
│   ├── strategy.py       # StrategyEngine (single/parallel/race par task_type)
│   └── manager.py        # Orchestrateur async
├── providers/
│   ├── base.py           # ABC async BaseProvider
│   ├── ollama.py         # Ollama via httpx async
│   └── claude.py         # Anthropic Claude via httpx async
└── api/
    └── routes.py         # Routeur FastAPI — /llm/*
cli/
└── neronctl.py           # CLI Typer
tests/
└── test_parallel.py      # Tests parallélisme / race / fallback
```

## Config neron.yaml

```yaml
neron:
  api_key: <clé auth X-Neron-API-Key>   # auth activée si présente

llm:
  model: llama3.2:1b
  fallback_model: llama3.2:1b
  host: http://localhost:11434
  timeout: 120
  default_provider: ollama
  claude_max_tokens: 1024
  model_map:
    default: llama3.2:1b
    code: deepseek-coder:latest
    summary: llama3.2:1b

strategy:
  chat: single       # single | parallel | race
  code: single
  summary: parallel
```

## Lancement

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8765
```

Depuis `/etc/neron/` avec le venv activé. `PYTHONPATH=/etc/neron/llm` est défini dans le service systemd.

## Authentification

Clé lue depuis `neron.api_key` dans `neron.yaml`, avec fallback sur la variable d'env `NERON_API_KEY`. Si aucune clé n'est définie, l'auth est désactivée (mode dev).

```bash
curl -X POST http://localhost:8765/llm/generate \
  -H "Content-Type: application/json" \
  -H "X-Neron-API-Key: <clé>" \
  -d '{"prompt": "Dis bonjour", "task_type": "chat"}'
```

## API

```
POST /llm/generate
{
  "prompt":    "Explique asyncio",
  "task_type": "chat",             # détermine le mode via StrategyEngine
  "mode":      "single"            # override optionnel : single | parallel | race
}

GET  /llm/health
GET  /llm/metrics
POST /llm/reload    # recharge neron.yaml sans redémarrage
```

## CLI

```bash
python -m cli.neronctl "Qui es-tu ?" --mode single
python -m cli.neronctl "Compare ces approches" --mode parallel --pretty
python -m cli.neronctl "Réponds vite" --mode race
```

## Tests

```bash
pytest tests/ -v
```
