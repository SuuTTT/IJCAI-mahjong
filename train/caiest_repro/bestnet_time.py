"""bestnet_time.py — per-move single-forward inference latency, CPU single-thread (Botzone-like).
Usage: python3 bestnet_time.py CKPT CHANNELS BLOCKS  -> prints/records ms/move (single fwd)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import build
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ckpt = sys.argv[1]; ch = int(sys.argv[2]); bl = int(sys.argv[3])
m = build("resbn_fused", channels=ch, blocks=bl)
sd = torch.load(ckpt, map_location="cpu"); m.load_state_dict(sd); m.eval()
d = np.load(os.path.join(HERE, "data", "cooked_single.npz"))
rng = np.random.RandomState(0)
idx = np.sort(rng.choice(len(d["act"]), 2500, replace=False))
OBS = np.ascontiguousarray(d["obs"][idx]).astype(np.int8)
MASK = np.ascontiguousarray(d["mask"][idx])
def single(o, mk):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(o)).float()[None],
            "action_mask": torch.from_numpy(np.ascontiguousarray(mk)).float()[None]}}).numpy().flatten()
    return int(lg.argmax())
for i in range(200): single(OBS[i], MASK[i])   # warm
n=1500; t0=time.time()
for i in range(n): single(OBS[i%len(OBS)], MASK[i%len(MASK)])
ms = 1000.0*(time.time()-t0)/n
# also worst-case: max over 300 individual moves
worst=0.0
for i in range(300):
    t1=time.time(); single(OBS[i%len(OBS)], MASK[i%len(MASK)]); worst=max(worst,1000.0*(time.time()-t1))
print(json.dumps({"ckpt":ckpt,"channels":ch,"blocks":bl,"mean_ms_per_move":round(ms,2),"worst_ms_over_300":round(worst,2),"threads":torch.get_num_threads(),"budget_ms":1000}))
