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
