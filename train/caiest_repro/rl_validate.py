"""
rl_validate.py — periodic VALIDATION of the live RL main policy. Every INTERVAL sec it loads the
current main snapshot (/tmp/lg_main.pkl, updated each league iter) and plays K rotated games via
the fast internal Sim vs each of the 6 strong top-30 imitations (greedy both sides), recording the
total net score. This is a FIXED external bar (unlike main_r, whose pool shifts), so a rising curve
= the policy is genuinely getting stronger. Appends {iter, net, ...} to /tmp/rl_val.json.

Runs on ssh8 (has sim_cnn + g30 anchors). Light: greedy inference, torch threads=1, low frequency.
"""
import os, sys, json, time, re, glob
sys.path.insert(0, '/root/mahjong'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from sim_cnn import Sim
from models_explore import ResBNCNN

MAIN = '/tmp/lg_main.pkl'
LOG = '/root/mahjong/rl_league.log'
OUT = '/tmp/rl_val.json'
OPPS = ['qwqwqawawa', 'dimaria', '渡鸦', 'knight', 'ChloePrice', 'QiuQiuR']
K = 6            # rotated games per opponent (×6 opp = 36 games/eval)
INTERVAL = 240   # sec


def _load(path):
    sd = torch.load(path, map_location='cpu')
    nb = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))
    m = ResBNCNN(channels=128, blocks=nb); m.load_state_dict(sd); m.eval(); return m


def greedy(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({'is_training': False, 'obs': {'observation': torch.from_numpy(np.ascontiguousarray(obs)),
                                                  'action_mask': torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn


def cur_iter():
    try:
        its = [int(m.group(1)) for m in re.finditer(r"it (\d+)/", open(LOG).read())]
        return its[-1] if its else 0
    except Exception:
        return 0


def evaluate():
    main = _load(MAIN); gm = greedy(main)
    opps = {}
    for o in OPPS:
        p = f'/root/mahjong/ckpt/g30_{o}_nf.pkl'
        if os.path.exists(p): opps[o] = greedy(_load(p))
    total = 0; per = {}
    g = 0
    for o, og in opps.items():
        net = 0
        for k in range(K):
            # rotate learner seats to cancel position bias
            if k % 2 == 0:
                sim = Sim([gm, og, gm, og], seed=90000 + g, quan=0, learner_seats=[0, 2], cnn=True); g += 1
                sim.play(); net += sim.scores[0] + sim.scores[2]
            else:
                sim = Sim([og, gm, og, gm], seed=90000 + g, quan=0, learner_seats=[1, 3], cnn=True); g += 1
                sim.play(); net += sim.scores[1] + sim.scores[3]
        per[o] = net; total += net
    return total, per


def main():
    hist = []
    if os.path.exists(OUT):
        try: hist = json.load(open(OUT)).get('points', [])
        except Exception: pass
    while True:
        if os.path.exists(MAIN):
            it = cur_iter()
            try:
                total, per = evaluate()
                hist.append({'it': it, 'net': total, 'per': per})
                json.dump({'updated': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                           'points': hist, 'games_per_eval': K * len(OPPS)}, open(OUT, 'w'))
                print(f"[val] it={it} net={total:+d} per={per}", flush=True)
            except Exception as e:
                print(f"[val] err: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
