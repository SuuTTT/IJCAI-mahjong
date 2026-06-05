"""
gauntlet_eval.py — fast IN-PROCESS strength measure vs a DIVERSE opponent set (different
architectures), for use as the RL promotion gate (#23). Replaces the blind "net vs own frozen
base" signal that produced the parity-trap illusion. Uses the same fast Sim as the actors
(no subprocess/judge), so it's cheap to call every eval_every iters.

The diverse opponents mirror eval/gauntlet.py: 16-block CNN, CNN+attn, resbn24/56, wide-192.
gauntlet_net(policy_net, n) returns total net points over n rotated games vs EACH opponent
(greedy both sides). Higher = stronger against varied styles.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from sim_cnn import Sim
from models_explore import build, ResBNCNN

_CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arch_ck')
OPP_SPECS = [
    ('cnn16',   'cnn',      {},                              f'{_CK}/base_16x128_final.pkl'),
    ('cnnattn', 'cnn_attn', {},                              f'{_CK}/explore/cnnattn.pkl'),
    ('resbn24', 'resbn',    {'channels': 128, 'blocks': 24}, f'{_CK}/explore/resbn24.pkl'),
    ('resbn56', 'resbn',    {'channels': 128, 'blocks': 56}, f'{_CK}/explore/resbn56.pkl'),
    ('w192',    'resbn',    {'channels': 192, 'blocks': 24}, f'{_CK}/explore/resbnw192.pkl'),
]
_opps = None

def _load_opps():
    global _opps
    if _opps is None:
        _opps = []
        for name, kind, cfg, path in OPP_SPECS:
            if not os.path.exists(path):
                continue
            try:
                m = build(kind, **cfg); m.load_state_dict(torch.load(path, map_location='cpu')); m.eval()
                _opps.append((name, m))
            except Exception as e:
                print(f"[gauntlet] skip {name}: {str(e)[:60]}", flush=True)
    return _opps

def _greedy(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({'is_training': False, 'obs': {'observation': torch.from_numpy(np.ascontiguousarray(obs)),
                                                  'action_mask': torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn

def gauntlet_net(policy_net, n_games=8, seed0=70000, per_opp=False):
    """policy_net: a CPU module whose forward(dict)->logits (e.g. ResBNCNN). Returns total net
    over n_games rotated games vs each diverse opponent (greedy)."""
    policy_net.eval()
    gm = _greedy(policy_net)
    breakdown = {}
    total = 0
    for name, opp in _load_opps():
        go = _greedy(opp); onet = 0
        for g in range(n_games):
            if g % 2 == 0:
                sim = Sim([gm, go, gm, go], seed=seed0 + g, quan=0, learner_seats=[0, 2], cnn=True)
                sim.play(); onet += sim.scores[0] + sim.scores[2]
            else:
                sim = Sim([go, gm, go, gm], seed=seed0 + g, quan=0, learner_seats=[1, 3], cnn=True)
                sim.play(); onet += sim.scores[1] + sim.scores[3]
        breakdown[name] = onet; total += onet
    return (total, breakdown) if per_opp else total

if __name__ == '__main__':
    # quick self-test: score the SL base + a distill model
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default=f'{_CK}/explore/resbn40.pkl')
    ap.add_argument('--blocks', type=int, default=40); ap.add_argument('--games', type=int, default=6)
    a = ap.parse_args()
    net = ResBNCNN(channels=128, blocks=a.blocks); net.load_state_dict(torch.load(a.ckpt, map_location='cpu')); net.eval()
    tot, bd = gauntlet_net(net, n_games=a.games, per_opp=True)
    print(f"{os.path.basename(a.ckpt)}: gauntlet net={tot:+d}  {bd}")
