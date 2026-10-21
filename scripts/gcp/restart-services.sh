#!/usr/bin/env bash
set -euo pipefail

RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
BACKEND_SERVICE_NAME="${BACKEND_SERVICE_NAME:-ragnarok-backend}"
RAG_HEALTH_URL="${RAG_HEALTH_URL:-http://127.0.0.1:8090/health}"
BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-http://127.0.0.1:8787/health}"
RAG_STARTUP_TIMEOUT_SECONDS="${RAG_STARTUP_TIMEOUT_SECONDS:-480}"
BACKEND_STARTUP_TIMEOUT_SECONDS="${BACKEND_STARTUP_TIMEOUT_SECONDS:-60}"
RAG_WATCHDOG_LOCK_FILE="${RAG_WATCHDOG_LOCK_FILE:-/tmp/ragnarok-rag-watchdog.lock}"

# The existing cron watchdog uses this same lock. Hold it through readiness so
# a failed probe cannot restart a service that the deploy is still warming up.
exec 9>"${RAG_WATCHDOG_LOCK_FILE}"
if ! flock -w 300 9; then
  echo "Could not acquire the RAG lifecycle lock; no services restarted."
  exit 1
fi
started_at="$(date --iso-8601=seconds)"

diagnostics() {
  local service="$1"
  systemctl show "${service}.service" --property=ActiveState,SubState,Result,ExecMainStatus,NRestarts,MemoryCurrent,MemoryPeak || true
  journalctl -u "${service}.service" --since "$started_at" -n 100 --no-pager || true
}

wait_ready() {
  local service="$1" url="$2" timeout="$3" kind="$4"
  local deadline=$((SECONDS + timeout)) state
  while [ "$SECONDS" -lt "$deadline" ]; do
    state="$(systemctl is-active "${service}.service" 2>/dev/null || true)"
    if [ "$state" = "failed" ]; then
      echo "${service}.service failed during startup:"
      diagnostics "$service"
      return 1
    fi
    if curl -fsS --max-time 5 "$url" 2>/dev/null | python3 -c '
import json, sys
try:
    body = json.load(sys.stdin)
except (ValueError, OSError):
    sys.exit(1)
ready = isinstance(body, dict) and (body.get("initialized") in (True, "true") if sys.argv[1] == "rag" else body.get("ok") is True)
sys.exit(0 if ready else 1)
' "$kind"; then
      echo "${service}.service is ready."
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for ${service}.service readiness after ${timeout}s:"
  diagnostics "$service"
  return 1
}

echo "Restarting RAG with watchdog recovery locked out during startup..."
if ! systemctl restart "${RAG_SERVICE_NAME}.service"; then
  diagnostics "$RAG_SERVICE_NAME"
  exit 1
fi
wait_ready "$RAG_SERVICE_NAME" "$RAG_HEALTH_URL" "$RAG_STARTUP_TIMEOUT_SECONDS" rag
echo "RAG is ready; restarting backend..."
if ! systemctl restart "${BACKEND_SERVICE_NAME}.service"; then
  diagnostics "$BACKEND_SERVICE_NAME"
  exit 1
fi
wait_ready "$BACKEND_SERVICE_NAME" "$BACKEND_HEALTH_URL" "$BACKEND_STARTUP_TIMEOUT_SECONDS" backend
