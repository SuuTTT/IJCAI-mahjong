#!/bin/bash
# Build a Botzone-uploadable Python bot zip (flat layout, __main__.py at root).
# Model weights are NOT in the zip — upload them separately to Botzone Storage
# as  data/bc_v3_ft_weights.npz  (the bot reads them from ./data/ at runtime).
set -e
cd "$(dirname "$0")/.."

BUILD=deploy/build
rm -rf "$BUILD"; mkdir -p "$BUILD"

# Flatten module imports for a single-directory zip
sed -e 's/^from data\.feature_agent import/from feature_agent import/' \
    -e 's/^from train\.numpy_infer import/from numpy_infer import/' \
    bot/ml_bot.py > "$BUILD/ml_bot.py"

cp bot/mahjong_bot.py        "$BUILD/mahjong_bot.py"
cp data/feature_agent.py     "$BUILD/feature_agent.py"
cp train/numpy_infer.py      "$BUILD/numpy_infer.py"

# Entry point: Botzone runs `python __main__.py` in the bot dir.
cat > "$BUILD/__main__.py" <<'EOF'
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Find the model weights. Botzone Storage is mounted at ./data/ at runtime;
# also accept a copy bundled next to this script.
if "MODEL" not in os.environ:
    for cand in ("data/bc_v3_ft_weights.npz", "bc_v3_ft_weights.npz"):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            os.environ["MODEL"] = p
            break
    else:
        os.environ["MODEL"] = os.path.join(HERE, "data", "bc_v3_ft_weights.npz")
from ml_bot import run
run()
EOF

cd "$BUILD"
zip -q -r ../mahjong_ml_bot.zip __main__.py ml_bot.py mahjong_bot.py feature_agent.py numpy_infer.py
cd - >/dev/null
echo "Built deploy/mahjong_ml_bot.zip:"
unzip -l deploy/mahjong_ml_bot.zip
echo
echo "Weights to upload to Botzone Storage as data/bc_v3_ft_weights.npz:"
ls -lh train/checkpoints/bc_v3_ft_weights.npz
