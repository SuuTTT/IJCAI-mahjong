"""
exam_dealin.py — offline DEFENSE exam from real tournament logs (no opponent code needed).
For every game that ended HU-by-discard, reconstruct the DISCARDER's exact decision state
(full logs are omniscient: all hands visible) and ask each candidate model what it would
discard there. A model that picks the lethal tile would have dealt in too.

Paired comparison on identical events is the metric that matters:
  python3 eval/exam_dealin.py --roots dirA dirB ... \
      --model distill100b=/root/mahjong/ckpt/distill100b_fused.pkl \
      --model v1=/root/mahjong/ckpt/sim6_v1_s600.pkl
"""
import os, sys, json, glob, argparse
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_BASE, 'train', 'caiest_repro'), os.path.join(_BASE, 'caiest_repro')):
    if os.path.isdir(_p): sys.path.insert(0, _p)
import numpy as np

def harvest_events(roots):
    """yield (obs38x4x9, mask235, lethal_action_id) for each HU-by-discard finish."""
    from feature import FeatureAgent
    n_games = 0
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, '**', '*_full_log.json'), recursive=True)):
            if os.path.getsize(path) == 0: continue
            try: d = json.load(open(path))
            except Exception: continue
            n_games += 1
            quan = 0; ag = None; pend = {}; last_play = None; last_act = None
            try:
                for rec in d:
                    disp = (rec.get('output') or {}).get('display') or {}
                    a = disp.get('action')
                    if not a: continue
                    if a == 'INIT': quan = disp.get('quan', 0)
                    elif a == 'DEAL':
                        ag = [FeatureAgent(s) for s in range(4)]
                        for s in range(4):
                            ag[s].request2obs('Wind %d' % quan); ag[s].request2obs('Deal ' + ' '.join(disp['hand'][s]))
                    elif a == 'DRAW':
                        p = disp['player']; t = disp['tile']
                        for s in range(4):
                            r = ag[s].request2obs('Draw %s' % t) if s == p else ag[s].request2obs('Player %d Draw' % p)
                            if s == p: pend[p] = r
                        last_act = 'DRAW'
                    elif a == 'PLAY':
                        p = disp['player']; t = disp['tile']
                        if p in pend:
                            o = pend.pop(p)
                            aid = ag[p].OFFSET_ACT['Play'] + ag[p].OFFSET_TILE[t]
                            last_play = (p, o['observation'].astype(np.int8), o['action_mask'].astype(np.bool_), aid)
                        for s in range(4): ag[s].request2obs('Player %d Play %s' % (p, t))
                        last_act = 'PLAY'
                    elif a == 'CHI':
                        p = disp['player']; mid = disp.get('tileCHI') or disp.get('tile')
                        for s in range(4): ag[s].request2obs('Player %d Chi %s' % (p, mid))
                        last_act = 'CHI'
                    elif a == 'PENG':
                        for s in range(4): ag[s].request2obs('Player %d Peng' % disp['player'])
                        last_act = 'PENG'
                    elif a == 'GANG':
                        for s in range(4): ag[s].request2obs('Player %d Gang' % disp['player'])
                        last_act = 'GANG'
                    elif a == 'BUGANG':
                        last_act = 'BUGANG'
                    elif a == 'HU':
                        w = disp.get('player')
                        if last_act == 'PLAY' and last_play and last_play[0] != w:
                            _, o, m, aid = last_play
                            if m.sum() > 1:                  # only real decisions
                                yield o, m, aid
                        break
            except Exception:
                continue
    print(f"(scanned {n_games} games)", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--model', action='append', required=True)  # name=path (resbn_fused 128x40)
    a = ap.parse_args()
    import torch
    from models_explore import build
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for spec in a.model:
        name, path = spec.split('=', 1)
        m = build('resbn_fused', channels=128, blocks=40)
        m.load_state_dict(torch.load(path, map_location='cpu')); m.eval().to(dev)
        models[name] = m
    O, M, L = [], [], []
    for o, m, aid in harvest_events(a.roots):
        O.append(o); M.append(m); L.append(aid)
    n = len(L)
    print(f"{n} deal-in decision events harvested")
    if not n: return
    O = torch.from_numpy(np.stack(O)); M = torch.from_numpy(np.stack(M)); L = np.array(L)
    picks = {}
    with torch.no_grad():
        for name, m in models.items():
            pr = []
            for i in range(0, n, 1024):
                lg = m({'is_training': False, 'obs': {'observation': O[i:i+1024].to(dev),
                                                      'action_mask': M[i:i+1024].float().to(dev)}})
                pr.append(lg.argmax(1).cpu().numpy())
            picks[name] = np.concatenate(pr)
    names = list(models)
    for name in names:
        r = float((picks[name] == L).mean())
        print(f"{name}: would-deal-in rate {r:.3f} ({int((picks[name]==L).sum())}/{n})")
    if len(names) == 2:
        x, y = picks[names[0]] == L, picks[names[1]] == L
        print(f"paired: both {int((x&y).sum())} | only-{names[0]} {int((x&~y).sum())} | only-{names[1]} {int((~x&y).sum())} | neither {int((~x&~y).sum())}")

if __name__ == '__main__':
    main()
