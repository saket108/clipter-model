#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_TEXT=0

for arg in "$@"; do
  case "$arg" in
    --with-text)
      WITH_TEXT=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

echo "[1/4] Python runtime"
python - <<'PY'
import sys, torch, torchvision
print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

echo "[2/4] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing CLIPTER Colab requirements"
python -m pip install -r requirements_colab.txt

if [[ "$WITH_TEXT" == "1" ]]; then
  echo "[3b/4] Installing optional text/tokenizer dependencies"
  python -m pip install "transformers>=4.46,<5"
fi

echo "[4/4] Verifying runtime"
if [[ "$WITH_TEXT" == "1" ]]; then
  python scripts/verify_env.py --with-text
else
  python scripts/verify_env.py
fi

echo "CLIPTER Colab bootstrap complete."
