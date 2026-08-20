#!/usr/bin/env bash
set -u
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
URL="${CARLA_WEB_URL:-http://127.0.0.1:8080}"

if curl --silent --fail --max-time 2 "$URL/api/status" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

echo "CARLA browser UI is not reachable at: $URL"
echo
echo "Start ./launch.sh on an Ubuntu x86_64 computer with an NVIDIA GPU."
echo "Then set CARLA_WEB_URL=http://THAT_COMPUTER_IP:8080 in .env and double-click this file again."
echo
read -r -p "Press Enter to close..." _
exit 1

