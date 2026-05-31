"""
quantize.py — shrink a weights .npz by storing as float16 (halves size).
LayerNorm gains/biases are small; everything stays loadable by NumpyMLP
(which upcasts to float32 on load).

    python3 train/quantize.py IN.npz OUT.npz
"""
import sys, numpy as np

inp, out = sys.argv[1], sys.argv[2]
d = np.load(inp)
half = {k: d[k].astype(np.float16) for k in d.files}
np.savez_compressed(out, **half)

import os
print(f"{inp}  {os.path.getsize(inp)/1e6:.2f}MB  ->  {out}  {os.path.getsize(out)/1e6:.2f}MB")
