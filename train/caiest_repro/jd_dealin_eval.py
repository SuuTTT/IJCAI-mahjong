"""
jd_dealin_eval.py — in-house DEAL-IN RATE for a danger-penalized KD 3-student ENSEMBLE.
Ensemble (exact deploy rule: mean softmax over legal) plays seat 0 vs 3x aug_s0; instrumented
DISim (lever1_eval pattern) counts deal-ins (opponent rong off seat-0's discard) and wins.
Game seeds = 900000 + evalseed*100000 + g: DISJOINT across --evalseed, and SHARED across lambdas
(paired design — every lambda sees the same deals).
  python3 jd_dealin_eval.py --cand a.pkl,b.pkl,c.pkl --evalseed 0 --ngames 250 --out X.json
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sim_cnn import Sim
from models_explore import build

AUG0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/aug/aug_128x40_s0.pkl")


class DISim(Sim):
    def __init__(self, *a, target=0, **k):
        super().__init__(*a, **k); self.target = target; self.dealins = 0; self.wins = 0
    def _score_rong(self, w, src, f):
        super()._score_rong(w, src, f)
        if src == self.target: self.dealins += 1
        if w == self.target: self.wins += 1
    def _score_selfdraw(self, w, f):
        super()._score_selfdraw(w, f)
        if w == self.target: self.wins += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)          # comma-separated fused student pkls
    ap.add_argument("--evalseed", type=int, default=0)
    ap.add_argument("--ngames", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    models = []
    for p in a.cand.split(","):
        mdl = build("resbn_fused", channels=128, blocks=40)
        mdl.load_state_dict(torch.load(p, map_location="cpu")); mdl.eval().to(dev)
        models.append(mdl)
    base = build("resbn_fused", channels=128, blocks=40)
    base.load_state_dict(torch.load(AUG0, map_location="cpu")); base.eval().to(dev)

    @torch.no_grad()
    def ens_fn(obs, mask):
        ob = torch.from_numpy(np.ascontiguousarray(obs)).float().to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(mask)).float().to(dev)
        mkb = mk > 0
        acc = None
        for mdl in models:
            lg = mdl({"is_training": False, "obs": {"observation": ob, "action_mask": mk}}).float()
            lg = torch.where(mkb, lg, torch.full_like(lg, -1e30))
            p = torch.softmax(lg, 1)
            acc = p if acc is None else acc + p
        return [int((acc / len(models))[0].argmax().item())]

    @torch.no_grad()
    def base_fn(obs, mask):
        ob = torch.from_numpy(np.ascontiguousarray(obs)).float().to(dev)
        mk = torch.from_numpy(np.ascontiguousarray(mask)).float().to(dev)
        lg = base({"is_training": False, "obs": {"observation": ob, "action_mask": mk}})
        return [int(lg[0].argmax().item())]

    dealins = wins = 0; scoresum = 0.0
    for g in range(a.ngames):
        pols = [base_fn] * 4; pols[0] = ens_fn
        sim = DISim(pols, seed=900000 + a.evalseed * 100000 + g, quan=0, cnn=True, target=0)
        _, sco = sim.play()
        dealins += sim.dealins; wins += sim.wins; scoresum += sco[0]
    out = dict(cand=[os.path.basename(c) for c in a.cand.split(",")], evalseed=a.evalseed,
               ngames=a.ngames, dealins=int(dealins), wins=int(wins),
               deal_in_rate=round(dealins / a.ngames, 4), win_rate=round(wins / a.ngames, 4),
               mean_score=round(scoresum / a.ngames, 4), seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"DONE {os.path.basename(a.out)}: dealin={out['deal_in_rate']} win={out['win_rate']}", flush=True)


if __name__ == "__main__":
    main()
