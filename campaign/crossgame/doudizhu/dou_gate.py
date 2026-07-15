"""Duplicate-style gate eval. An ensemble candidate (mean softmax over legal of K student
pkls, argmax) plays a FIXED seat vs 2 DouDizhuRuleAgentV1 over a fixed seed set; report mean
payoff. Calibration: candidate==reference (same single pkl) over same seeds -> stable number."""
import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import rlcard
from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
import rlcard.games.doudizhu.utils as U
from dou_model import DouMLP, make_mask, masked_logits, N_ACT, OBS_DIM

I2A = U.ID_2_ACTION

class EnsembleAgent(object):
    """Wraps K student nets as an rlcard raw agent (returns a raw action string)."""
    use_raw = True
    def __init__(self, pkls, hidden, layers, dev):
        self.nets = []
        for p in pkls:
            n = DouMLP(hidden=hidden, layers=layers)
            n.load_state_dict(torch.load(p, map_location="cpu")); n.eval().to(dev)
            self.nets.append(n)
        self.dev = dev
    @torch.no_grad()
    def _act_id(self, state):
        ob = np.asarray(state["obs"], np.int8)
        if ob.shape[0] < OBS_DIM:
            ob = np.concatenate([ob, np.zeros(OBS_DIM - ob.shape[0], np.int8)])
        obs = torch.from_numpy(ob).float().unsqueeze(0).to(self.dev)
        legal_ids = list(state["legal_actions"].keys())
        legal = torch.full((1, 400), -1, dtype=torch.int32, device=self.dev)
        legal[0, :len(legal_ids)] = torch.tensor(legal_ids, dtype=torch.int32, device=self.dev)
        mask = make_mask(legal)
        prob = None
        for n in self.nets:
            p = torch.softmax(masked_logits(n(obs), mask), 1)
            prob = p if prob is None else prob + p
        return int(prob.argmax(1).item())
    def step(self, state):
        return I2A[self._act_id(state)]
    def eval_step(self, state):
        aid = self._act_id(state)
        return I2A[aid], {}

def play(cand, seat, seeds, dev):
    rule = DouDizhuRuleAgentV1()
    tot = 0.0
    for sd in seeds:
        np.random.seed(int(sd)); random.seed(int(sd))  # make rule-agent opponents reproducible per seed
        env = rlcard.make("doudizhu", config={"seed": int(sd)})
        agents = [rule, rule, rule]; agents[seat] = cand
        env.set_agents(agents)
        _, payoffs = env.run(is_training=False)
        tot += payoffs[seat]
    return tot / len(seeds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", required=True, help="comma-separated candidate ensemble pkls")
    ap.add_argument("--ref", default="", help="single reference pkl for calibration")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--nseeds", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(10000, 10000 + a.nseeds))
    cand = EnsembleAgent(a.pkls.split(","), a.hidden, a.layers, dev)
    mp = play(cand, a.seat, seeds, dev)
    print(f"CANDIDATE ensemble(K={len(a.pkls.split(','))}) seat={a.seat} nseeds={a.nseeds} mean_payoff={mp:.4f}", flush=True)
    if a.ref:
        r1 = EnsembleAgent([a.ref], a.hidden, a.layers, dev)
        r2 = EnsembleAgent([a.ref], a.hidden, a.layers, dev)
        m1 = play(r1, a.seat, seeds, dev); m2 = play(r2, a.seat, seeds, dev)
        print(f"CALIBRATION ref==ref over same seeds: {m1:.4f} vs {m2:.4f} (delta {abs(m1-m2):.4f}, should be 0)", flush=True)

if __name__ == "__main__":
    main()
