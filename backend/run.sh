#!/usr/bin/env bash
# Launch the SumisoraOMR jianpu API in the sumisora conda env.
set -euo pipefail
export SUMISORA_ROOT="${SUMISORA_ROOT:-/liuyutian/SumisoraOMR-main/SumisoraOMR-main}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:5173,http://127.0.0.1:5173}"
exec /opt/conda/envs/sumisora/bin/python -m uvicorn \
  --app-dir "$(dirname "$0")" app:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
