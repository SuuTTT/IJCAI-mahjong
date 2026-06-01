#!/bin/bash
# Build a Botzone-uploadable Python bot zip (flat layout, __main__.py at root).
# Model weights are NOT in the zip — upload them separately to Botzone Storage
# as  data/bc_v3_ft_weights.npz  (the bot reads them from ./data/ at runtime).
set -e
cd "$(dirname "$0")/.."

BUILD=deploy/build
rm -rf "$BUILD"; mkdir -p "$BUILD"

# Flatten module imports for a single-directory zip.
# NOTE: match imports anywhere (incl. indented inside try:) — NOT anchored to ^.
sed -e 's/from data\.feature_agent import/from feature_agent import/g' \
    -e 's/from train\.numpy_infer import/from numpy_infer import/g' \
    -e 's/from train\.model import/from model import/g' \
    bot/ml_bot.py > "$BUILD/ml_bot.py"

cp bot/mahjong_bot.py        "$BUILD/mahjong_bot.py"
cp data/feature_agent.py     "$BUILD/feature_agent.py"
cp train/numpy_infer.py      "$BUILD/numpy_infer.py"

# Entry point: Botzone runs `python __main__.py` in the bot dir.
cat > "$BUILD/__main__.py" <<'EOF'
import os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Locate the model weights robustly. Botzone Storage is "data/ under the bot's
# runtime directory" — which may be the script dir (HERE) OR the working dir
# (CWD). Search both, in data/ and the root, and pick the LARGEST .npz (the full
# model, not the tiny test one). Record where we looked for the debug field.
if "MODEL" not in os.environ:
    bases = []
    for b in (HERE, os.getcwd()):
        if b and b not in bases:
            bases.append(b)
    cands = []
    for b in bases:
        for d in ("data", "."):
            cands += glob.glob(os.path.join(b, d, "*.npz"))
    cands = list(dict.fromkeys(os.path.abspath(c) for c in cands))  # dedup, keep order
    if cands:
        os.environ["MODEL"] = max(cands, key=lambda p: os.path.getsize(p))  # largest
    else:
        os.environ["MODEL"] = os.path.join(HERE, "data", "model.npz")
    os.environ["MODEL_SEARCH"] = f"bases={bases} found={[os.path.basename(c) for c in cands]}"
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
