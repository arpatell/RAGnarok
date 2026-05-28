#!/usr/bin/env bash
set -euo pipefail

# Configures HTTPS reverse proxy:
#   https://BACKEND_DOMAIN -> http://127.0.0.1:8787
#
# If you do not own a domain, use sslip.io:
#   BACKEND_DOMAIN=<VM_EXTERNAL_IP>.sslip.io bash scripts/gcp/configure-caddy.sh

BACKEND_DOMAIN="${BACKEND_DOMAIN:-}"
BACKEND_UPSTREAM="${BACKEND_UPSTREAM:-127.0.0.1:8787}"
ACME_EMAIL="${ACME_EMAIL:-}"

if [ -z "${BACKEND_DOMAIN}" ]; then
  echo "Set BACKEND_DOMAIN first. Example:"
  echo "  BACKEND_DOMAIN=203.0.113.10.sslip.io bash scripts/gcp/configure-caddy.sh"
  exit 1
fi

if [ "${EUID}" -eq 0 ]; then
  echo "Run as a normal sudo-capable user, not root."
  exit 1
fi

echo "[1/4] Installing Caddy"
sudo apt-get update -y
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gpg

if [ ! -f /usr/share/keyrings/caddy-stable-archive-keyring.gpg ]; then
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
fi

curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null

sudo apt-get update -y
sudo apt-get install -y caddy

echo "[2/4] Writing Caddyfile"
GLOBAL_BLOCK=""
if [ -n "${ACME_EMAIL}" ]; then
  GLOBAL_BLOCK="{
  email ${ACME_EMAIL}
}

"
fi

cat <<EOF | sudo tee /etc/caddy/Caddyfile >/dev/null
${GLOBAL_BLOCK}${BACKEND_DOMAIN} {
  encode zstd gzip
  reverse_proxy ${BACKEND_UPSTREAM}
}
EOF

echo "[3/4] Validating Caddy config"
sudo caddy validate --config /etc/caddy/Caddyfile

echo "[4/4] Restarting Caddy"
sudo systemctl enable caddy
sudo systemctl restart caddy
sudo systemctl --no-pager --full status caddy | sed -n '1,18p'

cat <<EOF

Caddy is configured.

Backend public URL:
  https://${BACKEND_DOMAIN}

Use this as:
  VITE_API_BASE_URL=https://${BACKEND_DOMAIN}
  CORS_ALLOWED_ORIGINS=https://ragnarok-reader.vercel.app

EOF
