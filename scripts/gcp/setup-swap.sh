#!/usr/bin/env bash
set -euo pipefail

SWAP_SIZE_GB="${SWAP_SIZE_GB:-4}"
SWAP_FILE="${SWAP_FILE:-/swapfile}"
SWAPPINESS="${SWAPPINESS:-10}"
VFS_CACHE_PRESSURE="${VFS_CACHE_PRESSURE:-50}"

if ! [[ "${SWAP_SIZE_GB}" =~ ^[0-9]+$ ]] || [ "${SWAP_SIZE_GB}" -lt 1 ]; then
  echo "SWAP_SIZE_GB must be a positive integer (current: ${SWAP_SIZE_GB})."
  exit 1
fi

if swapon --show=NAME --noheadings | awk '{$1=$1; print}' | grep -Fxq "${SWAP_FILE}"; then
  echo "Swap already enabled at ${SWAP_FILE}."
  swapon --show
  exit 0
fi

if [ ! -f "${SWAP_FILE}" ]; then
  AVAIL_GB="$(df --output=avail -BG / | tail -1 | tr -dc '0-9')"
  REQUIRED_GB=$((SWAP_SIZE_GB + 2))

  if [ "${AVAIL_GB}" -lt "${REQUIRED_GB}" ]; then
    echo "Not enough free disk space for ${SWAP_SIZE_GB}G swap. Available: ${AVAIL_GB}G."
    exit 1
  fi

  if ! sudo fallocate -l "${SWAP_SIZE_GB}G" "${SWAP_FILE}" 2>/dev/null; then
    sudo dd if=/dev/zero of="${SWAP_FILE}" bs=1M count="$((SWAP_SIZE_GB * 1024))" status=progress
  fi

  sudo chmod 600 "${SWAP_FILE}"
  sudo mkswap "${SWAP_FILE}"
fi

sudo swapon "${SWAP_FILE}"

FSTAB_LINE="${SWAP_FILE} none swap sw 0 0"
if ! grep -qE "^[^#]*[[:space:]]${SWAP_FILE}[[:space:]]+none[[:space:]]+swap" /etc/fstab; then
  echo "${FSTAB_LINE}" | sudo tee -a /etc/fstab >/dev/null
fi

sudo sysctl "vm.swappiness=${SWAPPINESS}"
sudo sysctl "vm.vfs_cache_pressure=${VFS_CACHE_PRESSURE}"
{
  echo "vm.swappiness=${SWAPPINESS}"
  echo "vm.vfs_cache_pressure=${VFS_CACHE_PRESSURE}"
} | sudo tee /etc/sysctl.d/99-ragnarok-swap.conf >/dev/null

echo "Swap enabled."
swapon --show
free -h
