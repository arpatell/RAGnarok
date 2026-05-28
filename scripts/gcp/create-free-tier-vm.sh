#!/usr/bin/env bash
set -euo pipefail

# Run this on your local machine after `gcloud auth login`.
# It creates a Google Compute Engine VM constrained to the published Free Tier shape.

PROJECT_ID="${PROJECT_ID:-}"
INSTANCE_NAME="${INSTANCE_NAME:-ragnarok-reader-vm}"
ZONE="${ZONE:-us-west1-b}"
MACHINE_TYPE="e2-micro"
EXPECTED_MACHINE_DESCRIPTION="0.25-2 vCPU, 1 shared core + 1 GB memory"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-30}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-standard}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2404-lts-amd64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
NETWORK="${NETWORK:-default}"
TAGS="${TAGS:-http-server,https-server}"

FREE_TIER_ZONE_RE='^(us-west1|us-central1|us-east1)-[a-z]$'

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Missing gcloud CLI. Install it first: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if [ -z "${PROJECT_ID}" ]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
  echo "Set PROJECT_ID or run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

if ! [[ "${ZONE}" =~ ${FREE_TIER_ZONE_RE} ]]; then
  echo "ZONE must be in a Free Tier region: us-west1, us-central1, or us-east1. Current: ${ZONE}"
  exit 1
fi

if [ "${BOOT_DISK_SIZE_GB}" != "30" ]; then
  echo "BOOT_DISK_SIZE_GB must stay at 30 to fit the 30 GB-month standard persistent disk Free Tier allowance."
  exit 1
fi

if [ "${BOOT_DISK_TYPE}" != "pd-standard" ]; then
  echo "BOOT_DISK_TYPE must be pd-standard for the Free Tier disk allowance. Current: ${BOOT_DISK_TYPE}"
  exit 1
fi

echo "Project: ${PROJECT_ID}"
echo "Instance: ${INSTANCE_NAME}"
echo "Zone: ${ZONE}"
echo "Machine: ${MACHINE_TYPE} (${EXPECTED_MACHINE_DESCRIPTION})"
echo "Boot disk: ${BOOT_DISK_SIZE_GB}GB ${BOOT_DISK_TYPE}"
echo "Provisioning: STANDARD / non-preemptible"
echo "GPU/TPU: none"
echo "Firewall: HTTP 80 + HTTPS 443"
echo
cat <<'EOF'
Cost preview:
  gcloud does not provide a reliable dollar estimate before creating this VM.
  Before approving, verify in Google Cloud Console:
    Compute Engine -> VM instances -> Create instance
  Match these values:
    e2-micro
    eligible region: us-west1, us-central1, or us-east1
    30GB standard persistent disk
    no GPUs/TPUs

Free Tier can still be affected by account eligibility, static/external IPv4 pricing,
other project resources, and outbound transfer beyond 1GB/month.
EOF
echo

echo "Creating firewall rules for HTTP/HTTPS if missing..."

if ! gcloud compute firewall-rules describe ragnarok-allow-http --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create ragnarok-allow-http \
    --project="${PROJECT_ID}" \
    --network="${NETWORK}" \
    --allow=tcp:80 \
    --target-tags=http-server \
    --source-ranges=0.0.0.0/0 \
    --description="Allow HTTP traffic to tagged RAGnarok VM"
fi

if ! gcloud compute firewall-rules describe ragnarok-allow-https --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create ragnarok-allow-https \
    --project="${PROJECT_ID}" \
    --network="${NETWORK}" \
    --allow=tcp:443 \
    --target-tags=https-server \
    --source-ranges=0.0.0.0/0 \
    --description="Allow HTTPS traffic to tagged RAGnarok VM"
fi

echo
echo "Creating Free Tier constrained VM..."
gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --provisioning-model=STANDARD \
  --network-interface=network="${NETWORK}",network-tier=STANDARD,stack-type=IPV4_ONLY \
  --maintenance-policy=MIGRATE \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --tags="${TAGS}" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
  --boot-disk-type="${BOOT_DISK_TYPE}" \
  --boot-disk-device-name="${INSTANCE_NAME}"

echo
echo "VM created. External IP:"
gcloud compute instances describe "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

cat <<EOF

Next:
1) SSH into the VM:
   gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}

2) Clone the repo and run:
   bash scripts/gcp/prepare-instance.sh

3) Check Google Cloud Console Billing estimate before leaving services running.
EOF
