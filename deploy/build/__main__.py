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
