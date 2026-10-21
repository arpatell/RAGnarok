#!/usr/bin/env bash
set -euo pipefail

RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
RAG_SEARCH_URL="${RAG_SEARCH_URL:-http://127.0.0.1:8090/rag/search}"
RAG_HEALTH_URL="${RAG_HEALTH_URL:-${RAG_SEARCH_URL%/rag/search}/health}"
RAG_WATCHDOG_TIMEOUT_SECONDS="${RAG_WATCHDOG_TIMEOUT_SECONDS:-90}"
RAG_WATCHDOG_RETRIES="${RAG_WATCHDOG_RETRIES:-2}"
RAG_WATCHDOG_RETRY_SLEEP_SECONDS="${RAG_WATCHDOG_RETRY_SLEEP_SECONDS:-2}"
RAG_WATCHDOG_STARTUP_GRACE_SECONDS="${RAG_WATCHDOG_STARTUP_GRACE_SECONDS:-480}"
RAG_WATCHDOG_LOCK_FILE="${RAG_WATCHDOG_LOCK_FILE:-/tmp/ragnarok-rag-watchdog.lock}"
RAG_WATCHDOG_LOG_FILE="${RAG_WATCHDOG_LOG_FILE:-/tmp/ragnarok-rag-watchdog.log}"
RAG_WATCHDOG_LOG_HEALTHY="${RAG_WATCHDOG_LOG_HEALTHY:-1}"

if [ "${EUID}" -eq 0 ]; then
  SUDO=()
else
  SUDO=(sudo)
fi

exec 9>"${RAG_WATCHDOG_LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >>"${RAG_WATCHDOG_LOG_FILE}"
}

startup_in_progress() {
  local state started started_seconds now
  state="$(systemctl is-active "${RAG_SERVICE_NAME}.service" 2>/dev/null || true)"
  if [ "$state" = "activating" ] || [ "$state" = "deactivating" ]; then
    return 0
  fi
  if [ "$state" != "active" ]; then
    return 1
  fi
  started="$(LC_ALL=C systemctl show "${RAG_SERVICE_NAME}.service" --property=ExecMainStartTimestamp --value 2>/dev/null || true)"
  if [ -z "$started" ]; then
    return 1
  fi
  started_seconds="$(date -d "$started" +%s 2>/dev/null || echo 0)"
  now="$(date +%s)"
  [ "$started_seconds" -gt 0 ] && [ "$((now - started_seconds))" -lt "$RAG_WATCHDOG_STARTUP_GRACE_SECONDS" ]
}

rag_ready() {
  curl -fsS --max-time "${RAG_WATCHDOG_TIMEOUT_SECONDS}" \
    "${RAG_HEALTH_URL}" 2>/dev/null | python3 -c '
import json, sys
try:
    body = json.load(sys.stdin)
except (ValueError, OSError):
    sys.exit(1)
sys.exit(0 if isinstance(body, dict) and body.get("initialized") in (True, "true") else 1)
'
}

if startup_in_progress; then
  log "startup grace active; leaving ${RAG_SERVICE_NAME}.service alone"
  exit 0
fi

attempt=1
while [ "${attempt}" -le "${RAG_WATCHDOG_RETRIES}" ]; do
  if rag_ready; then
    if [ "${RAG_WATCHDOG_LOG_HEALTHY}" = "1" ]; then
      log "readiness ok for ${RAG_HEALTH_URL}"
    fi
    exit 0
  fi
  sleep "${RAG_WATCHDOG_RETRY_SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

# A deploy or systemd restart may have begun while the probes were running.
if startup_in_progress; then
  log "service restarted during probes; honoring startup grace"
  exit 0
fi

active_state="$(systemctl is-active "${RAG_SERVICE_NAME}.service" 2>/dev/null || true)"
log "readiness failed for ${RAG_HEALTH_URL}; service_state=${active_state}; restarting ${RAG_SERVICE_NAME}.service"

"${SUDO[@]}" systemctl reset-failed "${RAG_SERVICE_NAME}.service" || true
"${SUDO[@]}" systemctl restart "${RAG_SERVICE_NAME}.service"

deadline=$((SECONDS + RAG_WATCHDOG_STARTUP_GRACE_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if rag_ready; then
    log "restart recovered ${RAG_SERVICE_NAME}.service"
    exit 0
  fi
  sleep 2
done

log "restart did not recover ${RAG_SERVICE_NAME}.service within ${RAG_WATCHDOG_STARTUP_GRACE_SECONDS}s"
exit 1
