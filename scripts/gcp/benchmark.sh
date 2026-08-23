#!/usr/bin/env bash
# Baseline/benchmark timing capture for the e2-micro production VM.
#
# Run ON THE VM (it measures localhost, so numbers exclude network RTT):
#   bash scripts/gcp/benchmark.sh                     # print table
#   PERF_RECORD=1 bash scripts/gcp/benchmark.sh       # also append to PERF.md
#   BASE_URL=http://127.0.0.1:8787 bash scripts/gcp/benchmark.sh
#
# Run from your LOCAL machine against the public endpoint to include RTT:
#   BASE_URL=https://YOUR_VM.sslip.io bash scripts/gcp/benchmark.sh
#
# The VM-only sections (journalctl, free, vmstat) are skipped automatically
# when they are not available.

set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8787}"
RAG_URL="${RAG_URL:-http://127.0.0.1:8090}"
PERF_RECORD="${PERF_RECORD:-0}"
PERF_FILE="${PERF_FILE:-PERF.md}"
RUN_LABEL="${RUN_LABEL:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"

SEARCH_QUERIES=(
  "death note"
  "guy finds notebook that can kill people"
  "satoru gojo"
)
INGEST_SAMPLE_URL="${INGEST_SAMPLE_URL:-}"

timestamp() { date -Is; }

curl_ms() {
  # Prints "<http_code> <total_ms>" for a GET/POST request.
  local method="$1" url="$2" body="${3:-}"
  local args=(-sS -o /dev/null -w "%{http_code} %{time_total}" --max-time 60)
  if [ "$method" = "POST" ]; then
    args+=(-X POST -H "Content-Type: application/json" -d "$body")
  fi
  local out
  out="$(curl "${args[@]}" "$url" 2>/dev/null)" || { echo "ERR -"; return; }
  echo "$out"
}

run_case() {
  local name="$1" method="$2" url="$3" body="${4:-}"
  local result code ms
  result="$(curl_ms "$method" "$url" "$body")"
  code="${result%% *}"
  ms_raw="${result#* }"
  if [ "$ms_raw" = "-" ] || [ -z "$ms_raw" ]; then
    printf '| %s | %s | %s | %s |\n' "$name" "$code" "-" "$(timestamp)"
    return
  fi
  # curl time_total is fractional seconds -> ms
  ms="$(awk -v t="$ms_raw" 'BEGIN { printf "%.0f", t * 1000 }')"
  printf '| %s | %s | %sms | %s |\n' "$name" "$code" "$ms" "$(timestamp)"
}

TABLE=""
TABLE+="### $RUN_LABEL"$'\n'
TABLE+=$'| Case | HTTP | Total | At |\n|---|---|---|---|\n'

echo "== RAGnarok benchmark: $RUN_LABEL =="
echo "== base=$BASE_URL =="

TABLE+="$(run_case "health" GET "$BASE_URL/health")"$'\n'
TABLE+="$(run_case "adapters" GET "$BASE_URL/adapters")"$'\n'

for q in "${SEARCH_QUERIES[@]}"; do
  enc="$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q" 2>/dev/null || printf '%s' "$q" | sed 's/ /%20/g')"
  TABLE+="$(run_case "search: $q" GET "$BASE_URL/api/search?q=$enc")"$'\n'
  TABLE+="$(run_case "rag/search: $q" GET "$BASE_URL/api/rag/search?q=$enc")"$'\n'
done

if [ -n "$INGEST_SAMPLE_URL" ]; then
  body="$(printf '{"url":"%s"}' "$INGEST_SAMPLE_URL")"
  TABLE+="$(run_case "ingest sample" POST "$BASE_URL/api/ingest" "$body")"$'\n'
fi

TABLE+="$(run_case "meta: death note (6471 manga)" GET "$BASE_URL/api/meta?type=manga&id=6471&title=Death%20Note")"$'\n'
TABLE+="$(run_case "meta recommendations" GET "$BASE_URL/api/meta/recommendations?title=Naruto")"$'\n'
TABLE+="$(run_case "rag health" GET "$RAG_URL/health")"$'\n'

echo
echo "$TABLE"

if [ "$PERF_RECORD" = "1" ]; then
  {
    echo ""
    echo "$TABLE"
  } >>"$PERF_FILE"
  echo "[benchmark] appended to $PERF_FILE"
fi

# ---- VM-only evidence ------------------------------------------------------
echo
echo "== Memory / swap =="
if command -v free >/dev/null 2>&1; then free -m; fi

echo
echo "== CPU steal sample (st column) =="
if command -v vmstat >/dev/null 2>&1; then vmstat 1 3; fi

echo
echo "== Service restart counts =="
if command -v systemctl >/dev/null 2>&1; then
  for svc in ragnarok-backend ragnarok-rag; do
    systemctl show "$svc.service" --property=NRestarts,ActiveState,SubState 2>/dev/null || true
  done
fi

echo
echo "== OOM check (kernel log) =="
if command -v journalctl >/dev/null 2>&1; then
  journalctl -k -n 200 --no-pager 2>/dev/null | grep -Ei 'killed process|out of memory|oom' || echo "(no OOM events in recent kernel log)"
fi

echo
echo "== Recent RAG stage timings =="
if command -v journalctl >/dev/null 2>&1; then
  journalctl -u ragnarok-rag -n 300 --no-pager 2>/dev/null | grep -E "RAG search ms|RAG hybrid query ms|warm-start" | tail -20 || echo "(no stage timings found)"
fi

echo
echo "== Recent backend API latencies =="
if command -v journalctl >/dev/null 2>&1; then
  journalctl -u ragnarok-backend -n 300 --no-pager 2>/dev/null | grep "\[api\]" | tail -20 || echo "(no api logs found)"
fi
