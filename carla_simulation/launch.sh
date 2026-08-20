#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "CARLA 0.9.16 server requires Linux x86_64 (or a separate Windows machine)."
  echo "This computer is $(uname -s) $(uname -m). See README_zh.md for the Mac remote workflow."
  exit 2
fi

for command_name in docker python3 nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing requirement: $command_name"
    exit 2
  fi
done

PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)')"
if [[ "$PY_MINOR" -lt 10 || "$PY_MINOR" -gt 12 ]]; then
  echo "CARLA 0.9.16 needs Python 3.10, 3.11, or 3.12; found $(python3 --version)."
  exit 2
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi
set -a
source .env
set +a

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

cleanup() {
  if [[ -n "${WEB_PID:-}" ]]; then kill "$WEB_PID" 2>/dev/null || true; fi
  docker compose down >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker compose up -d carla
echo "Waiting for CARLA RPC on ${CARLA_HOST}:${CARLA_PORT} ..."
.venv/bin/python - "$CARLA_HOST" "$CARLA_PORT" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=1):
            print("CARLA server is ready.")
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("CARLA did not become ready in 120 seconds. Run: docker compose logs carla")
PY

.venv/bin/python web_simulator.py &
WEB_PID=$!

for _ in {1..40}; do
  if .venv/bin/python - "$WEB_PORT" <<'PY'
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5): pass
PY
  then break; fi
  sleep 0.5
done

URL="http://127.0.0.1:${WEB_PORT}"
echo "CARLA is ready: $URL"
if [[ "${1:-}" != "--no-open" ]] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi
wait "$WEB_PID"

