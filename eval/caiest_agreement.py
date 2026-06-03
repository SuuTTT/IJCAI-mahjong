"""
caiest_agreement.py — measure how often our model agrees with chunjiandu (#1 bot) at the SAME
decision states, using caiest's (38,4,9) feature. Replays each Botzone log through 4 caiest
FeatureAgents; at every draw-turn discard decision compares our model's argmax to what
chunjiandu actually played. Reports agreement % + sample disagreements. (Eval only — too few
games to train on; this tells us how close we are to #1 and where we differ.)

  python3 eval/caiest_agreement.py <kind> <cfg-json> <model.pkl> <log1> [log2 ...]
  e.g. ... resbn '{"channels":128,"blocks":40}' train/caiest_repro/arch_ck/explore/resbn40.pkl others/.../*.json
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train', 'caiest_repro'))
import numpy as np, torch
from feature import FeatureAgent
from models_explore import build as build_explore

def load_model(kind, cfg, pkl):
    if kind == 'base':
        from model import CNNModel; m = CNNModel()
    else:
        m = build_explore(kind, **cfg)
    m.load_state_dict(torch.load(pkl, map_location='cpu')); m.eval()
    return m

def model_play(m, obs):
    with torch.no_grad():
        lg = m({'is_training': False, 'obs': {'observation': torch.from_numpy(obs['observation'][None]),
                                              'action_mask': torch.from_numpy(obs['action_mask'][None])}})
    return int(lg.numpy().flatten().argmax())

def disps(path):
    for rec in json.load(open(path)):
        d = (rec.get('output') or {}).get('display') or {}
        if d.get('action'): yield d

def agreement(m, path):
    quan = 0; ag = None; total = match = 0; diffs = []
    pending = {}  # seat -> obs at its draw (awaiting its PLAY)
    for d in disps(path):
        a = d['action']
        if a == 'INIT': quan = d.get('quan', 0)
        elif a == 'DEAL':
            ag = [FeatureAgent(s) for s in range(4)]
            for s in range(4):
                ag[s].request2obs('Wind %d' % quan); ag[s].request2obs('Deal ' + ' '.join(d['hand'][s]))
        elif a == 'DRAW':
            p = d['player']; t = d['tile']
            my_obs = None
            for s in range(4):
                r = ag[s].request2obs('Draw %s' % t) if s == p else ag[s].request2obs('Player %d Draw' % p)
                if s == p: my_obs = r
            if my_obs is not None:
                pending[p] = my_obs  # this seat must now act (play/hu/gang)
        elif a == 'PLAY':
            p = d['player']; t = d['tile']
            if p in pending:                       # compare our discard to chunjiandu's
                obs = pending.pop(p)
                if int(obs['action_mask'].sum()) > 1:
                    total += 1
                    ours = model_play(m, obs)
                    actual = ag[p].OFFSET_ACT['Play'] + ag[p].OFFSET_TILE[t]
                    if ours == actual: match += 1
                    elif len(diffs) < 10:
                        diffs.append((ag[p].action2response(ours), 'Play ' + t))
            for s in range(4): ag[s].request2obs('Player %d Play %s' % (p, t))
        elif a == 'CHI':
            p = d['player']; mid = d.get('tileCHI') or d.get('tile')
            for s in range(4): ag[s].request2obs('Player %d Chi %s' % (p, mid))
        elif a == 'PENG':
            p = d['player']
            for s in range(4): ag[s].request2obs('Player %d Peng' % p)
        elif a == 'GANG':
            p = d['player']
            for s in range(4): ag[s].request2obs('Player %d Gang' % p)
    return total, match, diffs

if __name__ == '__main__':
    kind, cfg, pkl = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
    logs = []
    for x in sys.argv[4:]: logs += glob.glob(x)
    logs = [l for l in logs if os.path.getsize(l) > 0]
    m = load_model(kind, cfg, pkl)
    T = M = 0; alldiffs = []
    for lg in logs:
        try:
            t, mm, df = agreement(m, lg); T += t; M += mm; alldiffs += df
        except Exception as e:
            print('skip', os.path.basename(lg), e)
    print(f"\n{kind} vs chunjiandu over {len(logs)} games: discard agreement {M}/{T} = {100*M/max(1,T):.1f}%")
    print("sample disagreements (ours -> chunjiandu's):")
    for o, c in alldiffs[:10]: print(f"   we:{o:12s} | #1:{c}")
