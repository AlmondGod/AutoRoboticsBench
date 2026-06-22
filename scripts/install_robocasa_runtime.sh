#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== RoboCasa runtime dependency install =="
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m pip install --upgrade pip

# RoboCasa in this benchmark checkout asserts these exact versions at import.
"${PYTHON_BIN}" -m pip install \
  "numpy==2.2.5" \
  "mujoco==3.3.1"

"${PYTHON_BIN}" -m pip install \
  "termcolor>=2.4" \
  "opencv-python-headless>=4.10" \
  "numba>=0.60" \
  "scipy>=1.14" \
  "h5py>=3.11" \
  "imageio[ffmpeg]>=2.34" \
  "tqdm>=4.66" \
  "pandas>=2.2" \
  "pyarrow>=15" \
  "gymnasium>=0.29" \
  "einops>=0.8" \
  "pillow>=10"

PY_MINOR="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
then
  "${PYTHON_BIN}" -m pip install "lerobot>=0.5,<0.6"
else
  echo "Python ${PY_MINOR} detected; installing LeRobot 0.4.4 because LeRobot 0.5 requires Python >=3.12."
  "${PYTHON_BIN}" -m pip install "lerobot==0.4.4"
fi

echo
echo "Verifying key imports..."
"${PYTHON_BIN}" - <<'PY'
import importlib

mods = [
    "numpy",
    "mujoco",
    "termcolor",
    "cv2",
    "numba",
    "scipy",
    "h5py",
    "imageio",
    "tqdm",
    "pandas",
    "pyarrow",
    "gymnasium",
    "einops",
    "PIL",
    "lerobot",
]
for mod in mods:
    importlib.import_module(mod)
    print(f"{mod}: ok")
PY
