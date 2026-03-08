#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_end_to_end.sh [options]

Runs the local benchmark with local keys, publishes a public-safe static dataset,
and can optionally serve the GitHub Pages-style viewer.

Options:
  --output-dir <dir>        Local run history directory (default: runs)
  --viewer-output-dir <dir> Published viewer dataset dir (default: data/latest)
  --run-id <id>             Explicit run id (default: auto timestamp)
  --database-url <url>      Override database URL for the run
  --publish-mode <mode>     auto|supplemental|replace (default: auto)
  --skip-run                Skip benchmark execution and only publish from database
  --serve                   Start a local HTTP server after publish
  --port <port>             Local server port for --serve (default: 8877)
  -h, --help                Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

OUTPUT_DIR="runs"
VIEWER_OUTPUT_DIR="data/latest"
RUN_ID="run_$(date -u +%Y%m%d_%H%M%S)"
DATABASE_URL_INPUT=""
PUBLISH_MODE="auto"
SKIP_RUN=0
SERVE=0
PORT=8877

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --viewer-output-dir)
      VIEWER_OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --database-url)
      DATABASE_URL_INPUT="${2:-}"
      shift 2
      ;;
    --publish-mode)
      PUBLISH_MODE="${2:-}"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    --serve)
      SERVE=1
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

if [[ -n "${DATABASE_URL_INPUT}" ]]; then
  export DATABASE_URL="${DATABASE_URL_INPUT}"
else
  export DATABASE_URL="sqlite:///${ROOT_DIR}/${RUN_DIR}/benchmark.db"
fi
export SCHEDULER_ENABLED=0

if [[ "${SKIP_RUN}" -ne 1 ]]; then
  echo "==> Running benchmark: ${RUN_ID}"
  python3 scripts/run_benchmark_once.py \
    --database-url "${DATABASE_URL}" \
    --run-id "${RUN_ID}" \
    --run-dir "${RUN_DIR}"
else
  echo "==> Skipping benchmark run"
fi

echo "==> Publishing static viewer dataset"
python3 scripts/publish_latest_to_viewer.py \
  --database-url "${DATABASE_URL}" \
  --output-dir "${VIEWER_OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --run-dir "${RUN_DIR}" \
  --publish-mode "${PUBLISH_MODE}"

echo ""
echo "Complete."
echo "Run ID: ${RUN_ID}"
echo "Run dir: ${ROOT_DIR}/${RUN_DIR}"
echo "Database: ${DATABASE_URL}"
echo "Published viewer data: ${ROOT_DIR}/${VIEWER_OUTPUT_DIR}"
echo "Viewer entry: ${ROOT_DIR}/viewer/index.html"

if [[ "${SERVE}" -eq 1 ]]; then
  echo ""
  echo "Starting local server on port ${PORT}..."
  python3 -m http.server "${PORT}"
fi
