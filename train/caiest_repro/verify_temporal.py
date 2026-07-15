"""
verify_temporal.py — argmax agreement of numpy_temporal vs torch TemporalNet on real states.
CPU only (good GPU neighbor). Loads cooked_single.npz (obs/mask) + cooked_seq.npz (seq),
same val split as training, samples N states, compares masked-argmax.
Usage: python3 verify_temporal.py <temporal.pkl> <temporal.npz> [N]
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_seq import build_seq
from numpy_temporal import NumpyTemporal

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")


def main():
    pkl, npz = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    torch.set_num_threads(4)

    d = np.load(os.path.join(DDIR, "cooked_single.npz"))
    o, m = d["obs"], d["mask"]
    ds = np.load(os.path.join(DDIR, "cooked_seq.npz"))
    seq = ds["seq"].astype(np.int64)
    Ntot = len(o)
    # same val split as seq_bc (rng 12345) so we test on HELD-OUT states
    rng = np.random.RandomState(12345); perm = rng.permutation(Ntot)
    nval = min(50000, Ntot // 20); vidx = np.sort(perm[:nval])
    pick = vidx[np.linspace(0, len(vidx) - 1, N).astype(int)]

    net = build_seq('temporal', channels=128, blocks=40, emb=64, gru=256)
    net.load_state_dict(torch.load(pkl, map_location='cpu')); net.eval()
    npm = NumpyTemporal(npz)

    agree = 0; maxdiff = 0.0; t0 = time.time()
    tmoves = []
    for c, idx in enumerate(pick):
        ob = o[idx].astype(np.float32); mk = m[idx].astype(np.float32); sq = seq[idx]
        with torch.no_grad():
            tl = net({'is_training': False,
                      'obs': {'observation': torch.from_numpy(ob[None]),
                              'action_mask': torch.from_numpy(mk[None])},
                      'seq': torch.from_numpy(sq[None].astype(np.int64))}).numpy()[0]
        tstart = time.time()
        nl = npm.logits(ob, mk, sq)
        tmoves.append((time.time() - tstart) * 1000)
        # compare over legal actions only (masked)
        legal = mk > 0
        ta = int(np.argmax(np.where(legal, tl, -1e30)))
        na = int(np.argmax(np.where(legal, nl, -1e30)))
        if ta == na:
            agree += 1
        dd = float(np.max(np.abs(tl[legal] - np.clip(nl[legal], -1e9, 1e9))))
        maxdiff = max(maxdiff, dd)
        if (c + 1) % 200 == 0:
            print("  %d/%d agree=%.2f%% maxdiff=%.4f" % (c + 1, N, 100.0 * agree / (c + 1), maxdiff), flush=True)
    tmoves = np.array(tmoves)
    print("AGREE %d/%d = %.3f%%  max|logit-diff|=%.5f  numpy_ms mean=%.1f p95=%.1f max=%.1f (single-thread, box CPU)" % (
        agree, N, 100.0 * agree / N, maxdiff, tmoves.mean(), np.percentile(tmoves, 95), tmoves.max()), flush=True)


if __name__ == '__main__':
    main()
