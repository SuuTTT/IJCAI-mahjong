import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Find the model weights. Botzone Storage is mounted at ./data/ at runtime.
# Auto-discover ANY *.npz so you can upload tiny/fp16/full without renaming.
import glob
if "MODEL" not in os.environ:
    cands = []
    for d in ("data", "."):
        cands += sorted(glob.glob(os.path.join(HERE, d, "*.npz")))
    os.environ["MODEL"] = cands[0] if cands else os.path.join(HERE, "data", "model.npz")
from ml_bot import run
run()
