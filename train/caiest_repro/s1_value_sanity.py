"""
s1_value_sanity.py -- verify the VALUE_C_60K value head sign/scale on hand-built states.

Higher predicted value must mean better-for-that-seat (target = score/SC). So on a
14-tile hand that is tenpai after discarding an isolated tile, the value head must
prefer the post-discard state that KEEPS the near-complete hand (discard the isolated
tile) over the one that BREAKS it (discard from a completed set).

Also reports value across all 4 Final2 source ids to confirm the ordering is stable
w.r.t. the (arbitrary) deploy src choice.
"""
import os, sys, copy
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch
import models_explore
models_explore.IN_PLANES = 38
from f2_value_v2 import VNet
from feature import FeatureAgent

torch.set_num_threads(1)
DEV = "cuda:0"

def load_vnet():
    net = VNet(cond=True)
    net.load_state_dict(torch.load("results/VALUE_C_60K.pt", map_location="cpu"))
    net.eval().to(DEV)
    return net

def post_discard_obs(ag, tile):
    a = copy.deepcopy(ag)
    a.request2obs(f"Player 0 Play {tile}")
    return a.obs.reshape(38, 4, 9).astype(np.float32)

def value(net, obs_arr, src):
    ob = torch.from_numpy(np.ascontiguousarray(obs_arr)).float().to(DEV)
    sb = torch.full((len(obs_arr),), src, dtype=torch.long, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast():
        return net(ob, sb).float().cpu().numpy()

def main():
    net = load_vnet()
    # 13-tile hand: W1W1W1 (pung) W2W3W4 (seq) T5T5T5 (pung) B7B8B9 (seq) J1
    # then draw B1 (isolated). Discarding B1 -> tenpai (4 sets, pair-wait on J1).
    # Discarding W1 -> breaks the W1 pung (much worse).
    hand13 = ["W1","W1","W1","W2","W3","W4","T5","T5","T5","B7","B8","B9","J1"]
    ag = FeatureAgent(0)
    ag.request2obs("Wind 0")
    ag.request2obs("Deal " + " ".join(hand13))
    ag.request2obs("Draw B1")
    assert len(ag.hand) == 14, ag.hand

    print("hand(14):", sorted(ag.hand))
    cands = ["B1", "W1", "T5", "B7"]   # B1=keep near-complete; others break sets
    for src in range(4):
        vals = value(net, np.stack([post_discard_obs(ag, t) for t in cands]), src)
        row = "  ".join(f"{t}:{v:+.4f}" for t, v in zip(cands, vals))
        best = cands[int(np.argmax(vals))]
        print(f"src={src}  {row}   argmax={best}  {'OK' if best=='B1' else 'FAIL'}")

    # second state: a flat far-from-win hand vs one where we keep a pung
    hand13b = ["W1","W3","W5","T2","T4","T6","B1","B3","B5","F1","F2","J1","J2"]
    ag2 = FeatureAgent(0); ag2.request2obs("Wind 0")
    ag2.request2obs("Deal " + " ".join(hand13b)); ag2.request2obs("Draw W1")
    # now W1W1 pair formed by draw. Discard J2 (isolated honor) keeps pair; discard W1 breaks it.
    print("\nhand2(14):", sorted(ag2.hand))
    cands2 = ["J2", "W1", "F1"]
    vals2 = value(net, np.stack([post_discard_obs(ag2, t) for t in cands2]), 0)
    print("src=0 ", "  ".join(f"{t}:{v:+.4f}" for t,v in zip(cands2, vals2)),
          " argmax=", cands2[int(np.argmax(vals2))])

if __name__ == "__main__":
    main()
