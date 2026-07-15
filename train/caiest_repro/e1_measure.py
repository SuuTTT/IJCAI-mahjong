"""
e1_measure.py — per-model claim-rate + expert-gap for the over-claiming study.

For a trained model (fused or BN ResBNCNN), on the HELD-OUT claim-legal reference set
(data/teachers/claim_states.npz, 11175 states, 100% claim-legal, leaders' real decisions):
  1. claim-rate = claims / claim-legal states   (claim = act in [36,133))
  2. expert-gap on the experts' real decisions:
        agree              = pred == expert_act
        claim_when_pass    = expert passed (act<1) but model claims (pred in [36,133))  / #expert-pass
        pass_when_claim    = expert claimed (act in [36,133)) but model passes (pred<1)  / #expert-claim
Also reports the EXPERT reference's own claim-rate on the same set (the baseline).

Writes a single-record JSON (merged into E1_RESULTS.json by the harness).

  python3 e1_measure.py --model ckpt/e1/full_128x40_s0.pkl --kind resbn_fused \
      --channels 128 --blocks 40 --gpu 0 --out ckpt/e1/meas/full_128x40_s0.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import build

REF = "data/teachers/claim_states.npz"   # leaders' real decisions, all claim-legal


def is_claim(a):  return (a >= 36) & (a < 133)
def is_pass(a):   return a < 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="resbn_fused")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"

    d = np.load(REF)
    obs = torch.from_numpy(d["obs"]).float()
    mask = torch.from_numpy(d["mask"]).float()
    exp_act = d["act"].astype(int)
    N = len(exp_act)

    net = build(a.kind, channels=a.channels, blocks=a.blocks).eval().to(dev)
    sd = torch.load(a.model, map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith(("stem", "body", "foot")) for k in sd):
        sd = sd["state_dict"]
    net.load_state_dict(sd)

    pred = np.empty(N, dtype=int)
    with torch.no_grad():
        for i in range(0, N, 4096):
            o = obs[i:i + 4096].to(dev); mk = mask[i:i + 4096].to(dev)
            pred[i:i + 4096] = net({"is_training": False,
                "obs": {"observation": o, "action_mask": mk}}).argmax(1).cpu().numpy()

    # all N states are claim-legal (pre-filtered): claim-rate = claims / N
    claim_rate = float(is_claim(pred).mean())
    chi_rate = float(((pred >= 36) & (pred < 99)).mean())
    peng_rate = float(((pred >= 99) & (pred < 133)).mean())
    expert_claim_rate = float(is_claim(exp_act).mean())

    agree = float((pred == exp_act).mean())
    exp_pass = is_pass(exp_act); exp_claim = is_claim(exp_act)
    claim_when_pass = float(is_claim(pred)[exp_pass].mean()) if exp_pass.sum() else 0.0
    pass_when_claim = float(is_pass(pred)[exp_claim].mean()) if exp_claim.sum() else 0.0

    rec = dict(model=os.path.basename(a.model), kind=a.kind,
               channels=a.channels, blocks=a.blocks, n_eval=N,
               claim_rate=round(claim_rate, 4), chi_rate=round(chi_rate, 4),
               peng_rate=round(peng_rate, 4),
               expert_claim_rate=round(expert_claim_rate, 4),
               over_claim_delta=round(claim_rate - expert_claim_rate, 4),
               expert_gap=dict(agree=round(agree, 4),
                               claim_when_pass=round(claim_when_pass, 4),
                               pass_when_claim=round(pass_when_claim, 4),
                               n_expert_pass=int(exp_pass.sum()),
                               n_expert_claim=int(exp_claim.sum())))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=2)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
