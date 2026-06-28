#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SWAP_SIZE_GB="${SWAP_SIZE_GB:-4}"
SWAP_FILE="${SWAP_FILE:-/swapfile}"
INSTALL_SWAP="${INSTALL_SWAP:-1}"
BACKEND_SERVICE_NAME="${BACKEND_SERVICE_NAME:-ragnarok-backend}"
RAG_SERVICE_NAME="${RAG_SERVICE_NAME:-ragnarok-rag}"
ENV_DIR="${ENV_DIR:-/etc/ragnarok}"
RUN_USER="${RUN_USER:-$USER}"

if ! [[ "${SWAP_SIZE_GB}" =~ ^[0-9]+$ ]] || [ "${SWAP_SIZE_GB}" -lt 1 ]; then
  echo "SWAP_SIZE_GB must be a positive integer (current: ${SWAP_SIZE_GB})."
  exit 1
fi

if [ ! -f "${REPO_ROOT}/package.json" ] || [ ! -d "${REPO_ROOT}/apps/backend" ] || [ ! -d "${REPO_ROOT}/rag" ]; then
  echo "Run this script from inside the manga repo (or keep it at scripts/gcp)."
  exit 1
fi

if [ "${EUID}" -eq 0 ]; then
  echo "Run as a normal sudo-capable user, not root."
  exit 1
fi

echo "[1/8] Installing base packages"
sudo apt-get update -y
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  jq \
  build-essential \
  python3 \
  python3-venv \
  python3-pip

echo "[2/8] Ensuring Node.js >= 20"
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -p "process.versions.node.split('.')[0]")"
else
  NODE_MAJOR="0"
fi

if [ "${NODE_MAJOR}" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

echo "[3/8] Enabling swap (size=${SWAP_SIZE_GB}G)"
if [ "${INSTALL_SWAP}" = "1" ]; then
  SWAP_SIZE_GB="${SWAP_SIZE_GB}" SWAP_FILE="${SWAP_FILE}" bash "${SCRIPT_DIR}/setup-swap.sh"
else
  echo "INSTALL_SWAP=0 -> skipped swap setup"
fi

echo "[4/8] Installing Node dependencies + building backend"
cd "${REPO_ROOT}"
npm ci
npm run build --workspace @panelflow/backend

echo "[5/8] Creating Python virtualenv + installing RAG dependencies"
if [ ! -x "${REPO_ROOT}/.venv/bin/python" ]; then
  python3 -m venv "${REPO_ROOT}/.venv"
fi
"${REPO_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${REPO_ROOT}/.venv/bin/python" -m pip install -r "${REPO_ROOT}/rag/requirements.txt"

echo "[6/8] Creating runtime env files in ${ENV_DIR}"
sudo mkdir -p "${ENV_DIR}"

BACKEND_ENV_FILE="${ENV_DIR}/backend.env"
RAG_ENV_FILE="${ENV_DIR}/rag.env"

if [ ! -f "${BACKEND_ENV_FILE}" ]; then
  cat <<'EOF' | sudo tee "${BACKEND_ENV_FILE}" >/dev/null
# Required backend settings
PORT=8787
RAG_API_URL=http://127.0.0.1:8090/rag/search
CORS_ALLOWED_ORIGINS=
CEREBRAS_API_KEY=
CEREBRAS_MODEL=gpt-oss-120b
RAG_REQUEST_TIMEOUT_MS=8000
RAG_RESPONSE_CACHE_TTL_MS=300000
READ_NOW_TIMEOUT_MS=4000
READ_NOW_SERIES_RESOLVE_TIMEOUT_MS=1500
READ_NOW_TITLE_VARIANT_LIMIT=4
WEEBCENTRAL_SEARCH_TIMEOUT_MS=2500
MANHWAZONE_SEARCH_TIMEOUT_MS=2500
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
EOF
fi

if [ ! -f "${RAG_ENV_FILE}" ]; then
  cat <<'EOF' | sudo tee "${RAG_ENV_FILE}" >/dev/null
# Required RAG settings
RAG_HOST=0.0.0.0
RAG_PORT=8090
PINECONE_API_KEY=
PINECONE_INDEX=
PINECONE_NAMESPACE=top-4000
CEREBRAS_API_KEY=
CEREBRAS_MODEL=gpt-oss-120b
RAG_LLM_RERANK_ENABLED=0
RAG_TITLE_JIKAN_FALLBACK=0
RAG_TOP_K=10
RAG_CANDIDATE_POOL_SIZE=30
RAG_CHARACTER_CANDIDATE_POOL=80
RAG_RESPONSE_CACHE_TTL_SECONDS=300
RAG_RESPONSE_CACHE_MAX=256
EOF
fi

sudo chown root:root "${BACKEND_ENV_FILE}" "${RAG_ENV_FILE}"
sudo chmod 600 "${BACKEND_ENV_FILE}" "${RAG_ENV_FILE}"

echo "[7/8] Writing systemd units"
BACKEND_UNIT_FILE="/etc/systemd/system/${BACKEND_SERVICE_NAME}.service"
RAG_UNIT_FILE="/etc/systemd/system/${RAG_SERVICE_NAME}.service"
NODE_BIN="$(command -v node)"

cat <<EOF | sudo tee "${BACKEND_UNIT_FILE}" >/dev/null
[Unit]
Description=RAGnarok Backend API
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=${BACKEND_ENV_FILE}
ExecStart=${NODE_BIN} ${REPO_ROOT}/apps/backend/dist/server.js
Restart=always
RestartSec=3
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

cat <<EOF | sudo tee "${RAG_UNIT_FILE}" >/dev/null
[Unit]
Description=RAGnarok RAG API
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=${RAG_ENV_FILE}
ExecStart=/bin/bash ${REPO_ROOT}/rag/start-rag.sh
Restart=always
RestartSec=3
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

echo "[8/8] Reloading systemd and enabling services (not starting yet)"
sudo systemctl daemon-reload
sudo systemctl enable "${BACKEND_SERVICE_NAME}.service" "${RAG_SERVICE_NAME}.service"

cat <<EOF

Preparation complete.

Next:
1) Edit required env vars:
   sudo nano ${BACKEND_ENV_FILE}
   sudo nano ${RAG_ENV_FILE}

2) Start both services:
   bash ${SCRIPT_DIR}/start-services.sh

EOF
