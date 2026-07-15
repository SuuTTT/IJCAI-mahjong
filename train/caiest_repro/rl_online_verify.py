"""rl_online_verify.py — verify the self-play placement env before the full run.
Loads aug_s0 (BN) into a trainable ResBNCNN, plays a few deals with 1 learner seat vs
3 frozen aug_s0 opponents, and prints the per-deal placement reward (gate formula:
5 - avg_rank on the deal scores). base-vs-base should center ~2.5."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import ResBNCNN
from sim_cnn import Sim

BN = 'ckpt/aug/aug_128x40_s0.bn.pkl'

def placement(scores, seat):
    cs = scores[seat]
    greater = sum(1 for j in range(4) if scores[j] > cs)
    equal = sum(1 for j in range(4) if scores[j] == cs)
    avg_rank = greater + (equal + 1) / 2.0
    return 5.0 - avg_rank

def greedy(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({'is_training': False, 'obs': {
                'observation': torch.from_numpy(np.ascontiguousarray(obs)),
                'action_mask': torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn

def sample(m, store):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({'is_training': False, 'obs': {
                'observation': torch.from_numpy(np.ascontiguousarray(obs)),
                'action_mask': torch.from_numpy(np.ascontiguousarray(mask))}})
            p = torch.softmax(lg, -1); a = int(torch.multinomial(p, 1).item())
            logp = float(torch.log(p[0, a] + 1e-9))
        store.append([obs[0].astype(np.int8), mask[0], a, logp]); return [a]
    return fn

def main():
    torch.set_num_threads(2)
    m = ResBNCNN(channels=128, blocks=40)
    sd = torch.load(BN, map_location='cpu')
    m.load_state_dict(sd); m.eval()
    print("loaded", BN, "params", sum(p.numel() for p in m.parameters()))
    rews = []
    for g in range(12):
        ls = g % 4
        store = []
        pols = [greedy(m)] * 4
        pols[ls] = sample(m, store)
        sim = Sim(pols, seed=900000 + g, quan=0, learner_seats=[ls], cnn=True)
        traj, scores = sim.play()
        r = placement(scores, ls)
        rews.append(r)
        print(f"game {g} learner_seat={ls} scores={scores} reward={r:.2f} learner_decisions={len(store)}")
    print(f"MEAN placement over {len(rews)} deals = {np.mean(rews):.3f} (self-play parity ~2.5)")
    print("ENV_OK")

if __name__ == '__main__':
    main()
