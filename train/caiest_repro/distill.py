"""
distill.py — distill chunjiandu (#1 bot, SL+RL) into our net from its self-play logs.
Two steps:
  extract: replay 4xchunjiandu logs through caiest feature -> (obs,mask,act) npz of the
           champion's decisions (all 4 seats, all decision types with >1 legal action).
  finetune: BLEND the chunjiandu samples (upweighted) into cooked_single.npz and fine-tune
            the resbn base toward the #1 policy WITHOUT overfitting/forgetting.

Needs ~50-100 games to help (16 is too few -> overfit). Ready to fire when data lands.

  python3 distill.py extract <out.npz> <log_glob...>
  python3 distill.py finetune --base arch_ck/explore/resbn40.pkl --champ <out.npz> \
        --kind resbn --cfg '{"channels":128,"blocks":40}' --upweight 30 --epochs 4 --out distilled.pkl
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

def extract(out, globs):
    obs_l, mask_l, act_l = [], [], []
    logs = []
    for g in globs: logs += glob.glob(g)
    logs = [l for l in logs if os.path.getsize(l) > 0]
    for path in logs:
        try:
            d = json.load(open(path)); quan = 0; ag = None; pend = {}
            for rec in d:
                disp = (rec.get('output') or {}).get('display') or {}
                a = disp.get('action')
                if a == 'INIT': quan = disp.get('quan', 0)
                elif a == 'DEAL':
                    ag = [FeatureAgent(s) for s in range(4)]
                    for s in range(4): ag[s].request2obs('Wind %d' % quan); ag[s].request2obs('Deal ' + ' '.join(disp['hand'][s]))
                elif a == 'DRAW':
                    p = disp['player']; t = disp['tile']; my = None
                    for s in range(4):
                        r = ag[s].request2obs('Draw %s' % t) if s == p else ag[s].request2obs('Player %d Draw' % p)
                        if s == p: my = r
                    if my is not None and int(my['action_mask'].sum()) > 1: pend[p] = my
                elif a == 'PLAY':
                    p = disp['player']; t = disp['tile']
                    if p in pend:
                        o = pend.pop(p); act = ag[p].OFFSET_ACT['Play'] + ag[p].OFFSET_TILE[t]
                        obs_l.append(o['observation'].astype(np.int8)); mask_l.append(o['action_mask'].astype(np.bool_)); act_l.append(act)
                    for s in range(4): ag[s].request2obs('Player %d Play %s' % (p, t))
                elif a == 'CHI':
                    p = disp['player']; mid = disp.get('tileCHI') or disp.get('tile')
                    for s in range(4): ag[s].request2obs('Player %d Chi %s' % (p, mid))
                elif a == 'PENG':
                    for s in range(4): ag[s].request2obs('Player %d Peng' % disp['player'])
                elif a == 'GANG':
                    for s in range(4): ag[s].request2obs('Player %d Gang' % disp['player'])
        except Exception as e:
            print('skip', os.path.basename(path), e)
    obs = np.stack(obs_l).reshape((-1, 38, 4, 9)).astype(np.int8)
    np.savez_compressed(out, obs=obs, mask=np.stack(mask_l), act=np.array(act_l, np.int16))
    print(f"extracted {len(act_l)} champion decisions from {len(logs)} games -> {out}")

def finetune(a):
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from models_explore import build
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_np = np.load(os.path.join(os.path.dirname(__file__), 'data', 'cooked_single.npz'))
    champ = np.load(a.champ)
    # blend: champion samples repeated `upweight`x into the official data
    obs = np.concatenate([base_np['obs']] + [champ['obs']] * a.upweight)
    mask = np.concatenate([base_np['mask']] + [champ['mask']] * a.upweight)
    act = np.concatenate([base_np['act']] + [champ['act']] * a.upweight).astype(np.int64)
    print(f"blend: {len(base_np['act'])} official + {a.upweight}x{len(champ['act'])} champ = {len(act)}")
    ds = TensorDataset(torch.from_numpy(obs), torch.from_numpy(mask), torch.from_numpy(act))
    dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    m = build(a.kind, **json.loads(a.cfg)).to(dev); m.load_state_dict(torch.load(a.base, map_location='cpu'))
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == 'cuda'))
    for e in range(a.epochs):
        m.train()
        for o, mk, y in dl:
            o = o.to(dev); mk = mk.float().to(dev); y = y.to(dev)
            with torch.cuda.amp.autocast(enabled=(dev == 'cuda')):
                loss = F.cross_entropy(m({'is_training': True, 'obs': {'observation': o, 'action_mask': mk}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.save(m.state_dict(), a.out, _use_new_zipfile_serialization=False)
        print(f"distill epoch {e+1}/{a.epochs} done -> {a.out}", flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'extract':
        extract(sys.argv[2], sys.argv[3:])
    else:
        ap = argparse.ArgumentParser(); ap.add_argument('cmd')
        ap.add_argument('--base', required=True); ap.add_argument('--champ', required=True)
        ap.add_argument('--kind', default='resbn'); ap.add_argument('--cfg', default='{"channels":128,"blocks":40}')
        ap.add_argument('--upweight', type=int, default=30); ap.add_argument('--epochs', type=int, default=4)
        ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--out', default='distilled.pkl')
        finetune(ap.parse_args())
