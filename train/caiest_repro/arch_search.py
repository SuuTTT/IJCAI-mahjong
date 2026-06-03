"""
arch_search.py — 8h architecture search for the Mahjong CNN (38-plane feature, cooked_single.npz).
Phase A: train each config a few epochs (AMP, big batch), benchmark vs r18 through the OFFICIAL
JUDGE (the real signal), log net/wins/draws + val acc. Phase B: train the strongest config to
convergence as the final deploy candidate. Robust per-variant (one failure won't kill the run).

Status: /tmp/arch_search.json (live).  Run: python3 train/caiest_repro/arch_search.py
"""
import os, sys, json, time, subprocess, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from model_cfg import CfgCNN

DATA = os.path.join(HERE, 'data', 'cooked_single.npz')
CKDIR = os.path.join(HERE, 'arch_ck'); os.makedirs(CKDIR, exist_ok=True)
STATUS = '/tmp/arch_search.json'
R18 = 'MODEL=train/checkpoints/pbt_champion_fp16.npz OPENBLAS_NUM_THREADS=1 python3 bot/ml_bot.py'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BUDGET_S = float(os.environ.get('BUDGET_S', 8*3600))
PHASE_A_EPOCHS = int(os.environ.get('PHASE_A_EPOCHS', 5))
PHASE_B_EPOCHS = int(os.environ.get('PHASE_B_EPOCHS', 16))
BENCH_N = int(os.environ.get('BENCH_N', 60))
BATCH = int(os.environ.get('BATCH', 4096))

CONFIGS = [
    ('base_16x128',  dict(channels=128, blocks=16, head='flatten')),
    ('deep_24x128',  dict(channels=128, blocks=24, head='flatten')),
    ('deep_32x128',  dict(channels=128, blocks=32, head='flatten')),
    ('wide_16x256',  dict(channels=256, blocks=16, head='flatten')),
    ('wide_24x192',  dict(channels=192, blocks=24, head='flatten')),
    ('gap_16x128',   dict(channels=128, blocks=16, head='gap')),
    ('small_8x128',  dict(channels=128, blocks=8,  head='flatten')),
]
SMOKE = bool(os.environ.get('SMOKE'))
if SMOKE:
    CONFIGS = CONFIGS[:1]

# globals set in __main__ (functions read them at call time)
t0 = time.time(); results = {}; tds = vds = vl = None

def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)
def save_status(phase):
    json.dump({'phase': phase, 'elapsed_min': round((time.time()-t0)/60, 1),
               'budget_min': round(BUDGET_S/60), 'results': results}, open(STATUS, 'w'), indent=2)
def save_legacy(model, path):
    torch.save(model.state_dict(), path, _use_new_zipfile_serialization=False)

def train(model, epochs, tag, start_ep=0):
    tl = DataLoader(tds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == 'cuda'))
    best = -1; best_path = os.path.join(CKDIR, f"{tag}.pkl")
    for e in range(start_ep, start_ep + epochs):
        model.train()
        for o, m, y in tl:
            o = o.to(DEV, non_blocking=True); m = m.float().to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEV == 'cuda')):
                loss = F.cross_entropy(model({'is_training': True, 'obs': {'observation': o, 'action_mask': m}}), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); correct = 0
        with torch.no_grad():
            for o, m, y in vl:
                o = o.to(DEV); m = m.float().to(DEV); y = y.to(DEV)
                correct += (model({'is_training': False, 'obs': {'observation': o, 'action_mask': m}}).argmax(1) == y).sum().item()
        acc = correct / len(vds)
        if acc > best: best = acc; save_legacy(model, best_path)
        log(f"  {tag} ep{e+1} val_acc={acc:.4f} best={best:.4f}")
    return best_path, best

def bench(cfg, pkl, n=BENCH_N):
    cai = (f"CAIEST_MODEL={pkl} CAIEST_ARCH='{json.dumps(cfg)}' "
           f"PYTHONPATH={HERE} OPENBLAS_NUM_THREADS=1 python3 {HERE}/bot_cfg.py")
    out = subprocess.run([sys.executable, 'eval/bench_vs_bot.py', R18, cai, str(n), 'r18', 'cand'],
                         cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'}).stdout
    net = wins = draws = ill = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith('cand:'):
            for tok in s.split():
                if tok.startswith('net='): net = int(tok[4:].lstrip('+'))
                if tok.startswith('wins='): wins = int(tok[5:])
                if tok.startswith('illegal='): ill = int(tok[8:])
        if s.startswith('draws='):
            try: draws = int(s.split('=')[1].split()[0])
            except Exception: pass
    return {'net_vs_r18': net, 'cand_wins': wins, 'draws': draws, 'illegal': ill}

def main():
    global t0, results, tds, vds, vl
    t0 = time.time(); results = {}
    log("loading data...")
    d = np.load(DATA)
    obs = torch.from_numpy(d['obs']); mask = torch.from_numpy(d['mask']); act = torch.from_numpy(d['act'].astype(np.int64))
    N = obs.shape[0]; ds = TensorDataset(obs, mask, act)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    nval = N // 20; vds = Subset(ds, perm[:nval].tolist()); tds = Subset(ds, perm[nval:].tolist())
    vl = DataLoader(vds, batch_size=BATCH, shuffle=False)
    log(f"N={N:,} train={len(tds):,} val={len(vds):,}  device={DEV}")

    # Phase A: rank configs
    for tag, cfg in CONFIGS:
        if time.time() - t0 > BUDGET_S * 0.6:
            log(f"budget guard: skip {tag} (Phase A budget reached)"); break
        try:
            log(f"=== Phase A: {tag} {cfg} ===")
            model = CfgCNN(**cfg).to(DEV); np_ = sum(p.numel() for p in model.parameters())
            pkl, acc = train(model, PHASE_A_EPOCHS, tag)
            b = bench(cfg, pkl)
            results[tag] = {'cfg': cfg, 'params': np_, 'val_acc': round(acc, 4), **b, 'phase': 'A'}
            log(f"  {tag}: params={np_:,} val_acc={acc:.4f} net_vs_r18={b['net_vs_r18']} wins={b['cand_wins']}/{BENCH_N} illegal={b['illegal']}")
        except Exception as e:
            results[tag] = {'cfg': cfg, 'error': str(e)}; log(f"  {tag} FAILED: {e}\n{traceback.format_exc()[-400:]}")
        save_status('A')

    if SMOKE:
        log("SMOKE done"); save_status('smoke-done'); return
    # Phase B: train the winner to convergence
    ranked = sorted([k for k in results if results[k].get('net_vs_r18') is not None],
                    key=lambda k: results[k]['net_vs_r18'], reverse=True)
    if ranked:
        win = ranked[0]; cfg = results[win]['cfg']
        log(f"=== Phase B: winner={win} (net {results[win]['net_vs_r18']}) -> train {PHASE_B_EPOCHS} epochs ===")
        try:
            model = CfgCNN(**cfg).to(DEV)
            pkl, acc = train(model, PHASE_B_EPOCHS, win + '_final')
            b = bench(cfg, pkl, n=120)
            results[win + '_final'] = {'cfg': cfg, 'val_acc': round(acc, 4), **b, 'phase': 'B', 'deploy_pkl': pkl}
            log(f"  FINAL {win}: val_acc={acc:.4f} net_vs_r18={b['net_vs_r18']} wins={b['cand_wins']}/120 illegal={b['illegal']}")
        except Exception as e:
            results[win + '_final'] = {'error': str(e)}; log(f"  Phase B FAILED: {e}")
        save_status('B')
    log("DONE"); save_status('done')

if __name__ == '__main__':
    main()
