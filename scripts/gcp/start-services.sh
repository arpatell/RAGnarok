#!/usr/bin/env bash
set -euo pipefail

BACKEND_SERVICE_NAME="${BACKEND_SERVICE_NAME:-ragnarok-backend}"
RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
ENV_DIR="${ENV_DIR:-/etc/ragnarok}"
BACKEND_ENV_FILE="${ENV_DIR}/backend.env"
RAG_ENV_FILE="${ENV_DIR}/rag.env"

require_value() {
  local file="$1"
  local key="$2"
  local value
  value="$(grep -E "^${key}=" "${file}" 2>/dev/null | tail -1 | cut -d'=' -f2- | tr -d '[:space:]')"
  if [ -z "${value}" ]; then
    echo "Missing required ${key} in ${file}"
    return 1
  fi
  return 0
}

require_any_value() {
  local file="$1"
  shift
  local key
  for key in "$@"; do
    local value
    value="$(grep -E "^${key}=" "${file}" 2>/dev/null | tail -1 | cut -d'=' -f2- | tr -d '[:space:]')"
    if [ -n "${value}" ]; then
      return 0
    fi
  done
  echo "Missing required key set in ${file}. Need one of: $*"
  return 1
}

if [ ! -f "${BACKEND_ENV_FILE}" ] || [ ! -f "${RAG_ENV_FILE}" ]; then
  echo "Missing env file(s). Run scripts/gcp/prepare-instance.sh first."
  exit 1
fi

echo "Validating required env vars..."
require_value "${BACKEND_ENV_FILE}" "RAG_API_URL"
require_any_value "${BACKEND_ENV_FILE}" "CEREBRAS_API_KEY" "OPENAI_API_KEY"
require_value "${RAG_ENV_FILE}" "PINECONE_API_KEY"
require_value "${RAG_ENV_FILE}" "PINECONE_INDEX"
require_value "${RAG_ENV_FILE}" "PINECONE_NAMESPACE"
require_value "${RAG_ENV_FILE}" "CEREBRAS_API_KEY"

echo "Restarting services..."
sudo systemctl daemon-reload
sudo systemctl restart "${RAG_SERVICE_NAME}.service"
sudo systemctl restart "${BACKEND_SERVICE_NAME}.service"

echo
sudo systemctl --no-pager --full status "${RAG_SERVICE_NAME}.service" | sed -n '1,20p'
echo
sudo systemctl --no-pager --full status "${BACKEND_SERVICE_NAME}.service" | sed -n '1,20p'
echo
curl -fsS "http://127.0.0.1:8090/health" && echo
curl -fsS "http://127.0.0.1:8787/health" && echo

echo "Services started."
