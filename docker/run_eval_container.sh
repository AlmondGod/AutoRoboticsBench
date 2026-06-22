#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID is required" >&2
  exit 2
fi

if [[ -z "${TASK:-}" ]]; then
  echo "TASK is required" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-roboautoresearch:latest}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
SUBMISSION_DIR="${RUN_DIR}/output/final_submission"
EVAL_DIR="${RUN_DIR}/eval"
CONTAINER_NAME="robo_eval_${RUN_ID}"

if [[ ! -d "${SUBMISSION_DIR}" ]]; then
  echo "Final submission directory not found: ${SUBMISSION_DIR}" >&2
  exit 1
fi

mkdir -p "${EVAL_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network none \
  -v "${REPO_ROOT}/read_only:/workspace/read_only:ro" \
  -v "${SUBMISSION_DIR}:/workspace/final_submission:ro" \
  -v "${EVAL_DIR}:/workspace/eval" \
  "${IMAGE_NAME}" \
  python /workspace/read_only/eval/evaluate_submission.py \
    --task "${TASK}" \
    --submission /workspace/final_submission \
    --out /workspace/eval/results.json
