#!/usr/bin/env bash
# setup_env.sh  -  create the project's Python 3.14 environment (.venv) the same way on every machine.
# Run from the project root:   bash setup_env.sh
#
# Why a script instead of just "pip install -r requirements.txt":
#   tensorflow 2.22.0rc0 (the only build for Python 3.14) depends on 'keras-nightly' (3.16 dev).
#   Both keras-nightly and keras write into the same 'keras' folder, and the nightly one wins,
#   but the saved LSTM/GRU models need keras 3.15.1.  Step 3 swaps the nightly for 3.15.1.
set -euo pipefail

PY="${PYTHON:-python3.14}"                      # override with:  PYTHON=/path/to/python3.14 bash setup_env.sh

echo "1/4  creating .venv with $("$PY" --version)"
"$PY" -m venv .venv

echo "2/4  installing requirements.txt"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "3/4  replacing keras-nightly with keras 3.15.1"
.venv/bin/pip uninstall --quiet -y keras-nightly keras
.venv/bin/pip install --quiet --no-deps "keras==3.15.1"

echo "4/4  checking versions"
.venv/bin/python - <<'EOF'
import keras, sklearn, tensorflow as tf
print(f"keras {keras.__version__} | scikit-learn {sklearn.__version__} | tensorflow {tf.__version__}")
assert keras.__version__ == "3.15.1", "wrong keras version - saved models will not load"
EOF
echo "Done.  Activate with:  source .venv/bin/activate"
