#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod_prepare_run.sh --agent AGENT --task TASK --base BASE [--model MODEL] [--scaffold SCAFFOLD] [--seed N] [--timeout-hours H] [--skip-setup] [--workspace-root PATH] [--no-git-branch]

Example:
  scripts/runpod_prepare_run.sh --agent codex --task toy_pickplace --base dummy --seed 0
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AGENT=""
MODEL=""
SCAFFOLD=""
TASK=""
BASE=""
SEED="0"
TIMEOUT_HOURS="10"
SKIP_SETUP="0"
WORKSPACE_ROOT="/workspace"
GIT_BRANCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --scaffold)
      SCAFFOLD="${2:-}"
      shift 2
      ;;
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --base)
      BASE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --timeout-hours)
      TIMEOUT_HOURS="${2:-}"
      shift 2
      ;;
    --skip-setup)
      SKIP_SETUP="1"
      shift
      ;;
    --workspace-root)
      WORKSPACE_ROOT="${2:-}"
      shift 2
      ;;
    --no-git-branch)
      GIT_BRANCH_ARGS+=(--no-git-branch)
      shift
      ;;
    --start-branch)
      GIT_BRANCH_ARGS+=(--start-branch "${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${AGENT}" || -z "${TASK}" || -z "${BASE}" ]]; then
  usage >&2
  exit 2
fi

cd "${REPO_ROOT}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  ./scripts/setup_runpod_env.sh
fi

LAUNCH_LOG="$(mktemp)"
python scripts/launch_runpod_run.py \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --scaffold "${SCAFFOLD}" \
  --task "${TASK}" \
  --base "${BASE}" \
  --seed "${SEED}" \
  --timeout-hours "${TIMEOUT_HOURS}" \
  --workspace-root "${WORKSPACE_ROOT}" \
  "${GIT_BRANCH_ARGS[@]}" | tee "${LAUNCH_LOG}"

RUN_ID="$(awk -F= '/^run_id=/{print $2}' "${LAUNCH_LOG}" | tail -n 1)"
rm -f "${LAUNCH_LOG}"

if [[ -z "${RUN_ID}" ]]; then
  echo "Failed to parse run_id from launch output" >&2
  exit 1
fi

RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
START_EPOCH="$(date +%s)"
TIMEOUT_SECONDS="$(python - "${TIMEOUT_HOURS}" <<'PY'
import sys

print(int(float(sys.argv[1]) * 3600))
PY
)"
DEADLINE_EPOCH="$((START_EPOCH + TIMEOUT_SECONDS))"
printf "%s\n" "${START_EPOCH}" > "${RUN_DIR}/started_epoch.txt"
printf "%s\n" "${DEADLINE_EPOCH}" > "${RUN_DIR}/deadline_epoch.txt"
python - "${START_EPOCH}" > "${RUN_DIR}/started_at.txt" <<'PY'
from datetime import datetime, timezone
import sys

print(datetime.fromtimestamp(int(sys.argv[1]), timezone.utc).isoformat().replace("+00:00", "Z"))
PY
cat > "${RUN_DIR}/runpod_environment.txt" <<EOF
prepared_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
repo_root=${REPO_ROOT}
agent=${AGENT}
model=${MODEL}
scaffold=${SCAFFOLD}
task=${TASK}
base=${BASE}
seed=${SEED}
timeout_hours=${TIMEOUT_HOURS}
timeout_seconds=${TIMEOUT_SECONDS}
deadline_epoch=${DEADLINE_EPOCH}
workspace_root=${WORKSPACE_ROOT}
hostname=$(hostname)
python=$(command -v python || true)
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "${RUN_DIR}/gpu.txt" || true
fi

echo
echo "Prepared RunPod run: ${RUN_ID}"
echo "Prompt: runs/${RUN_ID}/prompt.txt"
echo
echo "Agent command wrapper:"
echo "  runs/${RUN_ID}/run.sh \"<command>\""
echo
echo "Start message for Codex/Claude/Gemini:"
echo "  Read runs/${RUN_ID}/prompt.txt and follow it exactly."
echo
echo "Toy smoke command:"
echo "  runs/${RUN_ID}/run.sh \"cd /workspace/task && python train.py\""
echo
echo "After the agent finishes:"
echo "  python scripts/finalize_run.py --run-id ${RUN_ID} --task ${TASK} --mode runpod"
