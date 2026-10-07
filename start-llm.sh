#!/usr/bin/env bash
# Lance le service llm (nœud "llm" de neron.server.yaml) comme start-core.sh lance le Core.
set -euo pipefail

CORE="${NERON_CORE:-/srv/neron/neronCore}"
PY="$CORE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "venv introuvable dans $CORE/.venv" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r system/requirements/core.txt" >&2
  exit 1
fi

# llm doit être résolvable (shim .pth ou dossier server/llm)
if ! "$PY" -c "import llm" 2>/dev/null; then
  echo "module 'llm' introuvable : lance /srv/neron/setup-shim.sh" >&2
  exit 1
fi

export NERON_ROOT="$CORE"
export PYTHONPATH="$CORE:$CORE/server"

cd "$CORE/server"
exec "$PY" -m common.serve llm
