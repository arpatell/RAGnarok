#!/usr/bin/env bash
set -euo pipefail

RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
RAG_SEARCH_URL="${RAG_SEARCH_URL:-http://127.0.0.1:8090/rag/search}"
RAG_WATCHDOG_QUERY="${RAG_WATCHDOG_QUERY:-death note}"
RAG_WATCHDOG_TIMEOUT_SECONDS="${RAG_WATCHDOG_TIMEOUT_SECONDS:-90}"
RAG_WATCHDOG_RETRIES="${RAG_WATCHDOG_RETRIES:-2}"
RAG_WATCHDOG_RETRY_SLEEP_SECONDS="${RAG_WATCHDOG_RETRY_SLEEP_SECONDS:-2}"
RAG_WATCHDOG_STARTUP_GRACE_SECONDS="${RAG_WATCHDOG_STARTUP_GRACE_SECONDS:-240}"
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

rag_search_ok() {
  curl -fsS --max-time "${RAG_WATCHDOG_TIMEOUT_SECONDS}" \
    -H "content-type: application/json" \
    -d "{\"query\":\"${RAG_WATCHDOG_QUERY}\"}" \
    "${RAG_SEARCH_URL}" 2>/dev/null \
    | grep -q '"top_results"[[:space:]]*:'
}

attempt=1
while [ "${attempt}" -le "${RAG_WATCHDOG_RETRIES}" ]; do
  if rag_search_ok; then
    if [ "${RAG_WATCHDOG_LOG_HEALTHY}" = "1" ]; then
      log "search ok for ${RAG_SEARCH_URL} query='${RAG_WATCHDOG_QUERY}'"
    fi
    exit 0
  fi
  sleep "${RAG_WATCHDOG_RETRY_SLEEP_SECONDS}"
  attempt=$((attempt + 1))
done

active_state="$(systemctl is-active "${RAG_SERVICE_NAME}.service" 2>/dev/null || true)"
log "search failed for ${RAG_SEARCH_URL} query='${RAG_WATCHDOG_QUERY}'; service_state=${active_state}; restarting ${RAG_SERVICE_NAME}.service"

"${SUDO[@]}" systemctl reset-failed "${RAG_SERVICE_NAME}.service" || true
"${SUDO[@]}" systemctl restart "${RAG_SERVICE_NAME}.service"

deadline=$((SECONDS + RAG_WATCHDOG_STARTUP_GRACE_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if rag_search_ok; then
    log "restart recovered ${RAG_SERVICE_NAME}.service"
    exit 0
  fi
  sleep 2
done

log "restart did not recover ${RAG_SERVICE_NAME}.service within ${RAG_WATCHDOG_STARTUP_GRACE_SECONDS}s"
exit 1
