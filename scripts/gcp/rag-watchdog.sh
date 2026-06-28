#!/usr/bin/env bash
set -euo pipefail

RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
RAG_HEALTH_URL="${RAG_HEALTH_URL:-http://127.0.0.1:8090/health}"
RAG_WATCHDOG_TIMEOUT_SECONDS="${RAG_WATCHDOG_TIMEOUT_SECONDS:-6}"
RAG_WATCHDOG_RETRIES="${RAG_WATCHDOG_RETRIES:-2}"
RAG_WATCHDOG_RETRY_SLEEP_SECONDS="${RAG_WATCHDOG_RETRY_SLEEP_SECONDS:-2}"
RAG_WATCHDOG_STARTUP_GRACE_SECONDS="${RAG_WATCHDOG_STARTUP_GRACE_SECONDS:-45}"
RAG_WATCHDOG_LOCK_FILE="${RAG_WATCHDOG_LOCK_FILE:-/tmp/ragnarok-rag-watchdog.lock}"
RAG_WATCHDOG_LOG_FILE="${RAG_WATCHDOG_LOG_FILE:-/tmp/ragnarok-rag-watchdog.log}"

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

health_ok() {
  curl -fsS --max-time "${RAG_WATCHDOG_TIMEOUT_SECONDS}" "${RAG_HEALTH_URL}" 2>/dev/null \
    | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'
}

attempt=1
while [ "${attempt}" -le "${RAG_WATCHDOG_RETRIES}" ]; do
  if health_ok; then
    exit 0
  fi
  sleep "${RAG_WATCHDOG_RETRY_SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

active_state="$(systemctl is-active "${RAG_SERVICE_NAME}.service" 2>/dev/null || true)"
log "health failed for ${RAG_HEALTH_URL}; service_state=${active_state}; restarting ${RAG_SERVICE_NAME}.service"

"${SUDO[@]}" systemctl reset-failed "${RAG_SERVICE_NAME}.service" || true
"${SUDO[@]}" systemctl restart "${RAG_SERVICE_NAME}.service"

deadline=$((SECONDS + RAG_WATCHDOG_STARTUP_GRACE_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if health_ok; then
    log "restart recovered ${RAG_SERVICE_NAME}.service"
    exit 0
  fi
  sleep 2
done

log "restart did not recover ${RAG_SERVICE_NAME}.service within ${RAG_WATCHDOG_STARTUP_GRACE_SECONDS}s"
exit 1
