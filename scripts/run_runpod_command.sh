#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 RUN_ID \"command\"" >&2
  exit 2
fi

RUN_ID="$1"
shift
COMMAND="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${REPO_ROOT}/runs/${RUN_ID}"
COMMAND_LOG="${RUN_DIR}/commands.jsonl"
COMMAND_TEXT_LOG="${RUN_DIR}/logs/commands.log"
DEADLINE_FILE="${RUN_DIR}/deadline_epoch.txt"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

export ROBOAUTORESEARCH_RUN_ID="${RUN_ID}"
export ROBOAUTORESEARCH_RUN_DIR="${RUN_DIR}"
export ROBOAUTORESEARCH_REPO_ROOT="${REPO_ROOT}"
export ROBOAUTORESEARCH_NO_DOCKER=1

mkdir -p "${RUN_DIR}/logs"

START_EPOCH="$(date +%s)"
START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
TIMEOUT_PREFIX=()

if [[ -f "${DEADLINE_FILE}" ]]; then
  DEADLINE_EPOCH="$(cat "${DEADLINE_FILE}")"
  REMAINING_SECONDS="$((DEADLINE_EPOCH - START_EPOCH))"
  if [[ "${REMAINING_SECONDS}" -le 0 ]]; then
    echo "Run ${RUN_ID} is past its deadline; refusing to execute command." >&2
    python - "$COMMAND_LOG" "$START_ISO" "$COMMAND" <<'PY'
import json
import sys

path, started_at, command = sys.argv[1:]
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "started_at": started_at,
        "finished_at": started_at,
        "duration_seconds": 0,
        "command": command,
        "exit_code": 124,
        "timed_out": True,
        "skipped": "deadline_elapsed",
    }, sort_keys=True) + "\n")
PY
    exit 124
  fi
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_PREFIX=(timeout "${REMAINING_SECONDS}s")
  fi
fi

{
  echo "[$START_ISO] $COMMAND"
} >> "${COMMAND_TEXT_LOG}"

set +e
if [[ "${#TIMEOUT_PREFIX[@]}" -gt 0 ]]; then
  "${TIMEOUT_PREFIX[@]}" bash -lc "${COMMAND}"
elif [[ -n "${REMAINING_SECONDS:-}" ]]; then
  python - "${REMAINING_SECONDS}" "${COMMAND}" <<'PY'
import subprocess
import sys

timeout_seconds = int(sys.argv[1])
command = sys.argv[2]
try:
    completed = subprocess.run(["bash", "-lc", command], timeout=timeout_seconds)
except subprocess.TimeoutExpired:
    sys.exit(124)
sys.exit(completed.returncode)
PY
else
  bash -lc "${COMMAND}"
fi
EXIT_CODE="$?"
set -e

END_EPOCH="$(date +%s)"
END_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
DURATION_SECONDS="$((END_EPOCH - START_EPOCH))"

python - "$COMMAND_LOG" "$START_ISO" "$END_ISO" "$DURATION_SECONDS" "$EXIT_CODE" "$COMMAND" <<'PY'
import json
import sys

path, started_at, finished_at, duration, exit_code, command = sys.argv[1:]
code = int(exit_code)
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": int(duration),
        "command": command,
        "exit_code": code,
        "timed_out": code == 124,
    }, sort_keys=True) + "\n")
PY

{
  echo "[$END_ISO] exit=${EXIT_CODE} duration_seconds=${DURATION_SECONDS}"
} >> "${COMMAND_TEXT_LOG}"

exit "${EXIT_CODE}"
