"""
distill_kl.py — fine-tune distill100b toward a champion WITHOUT the official cooked data: anti-forgetting
comes from a KL leash to the FROZEN base (deepresearch §3), not a data mix. Loss = CE(champ action) +
beta * KL(student || frozen_base) on the champ states. 12x suit+reflection augmentation. Runs on any
GPU box (only needs the champ npz + the fused base) -> keeps idle GPUs busy during the cooked-bound eval.

  python3 distill_kl.py --base <distill100b_fused.pkl> --champ data/lad_chunjiandu_v2.npz \
      --beta 1.0 --steps 800 --out ckpt/klchun_b10.pkl
"""
import os, sys, argparse, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from models_explore import build
from suit_aug import augment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True); ap.add_argument('--champ', required=True)
    ap.add_argument('--kind', default='resbn_fused'); ap.add_argument('--cfg', default='{"channels":128,"blocks":40}')
    ap.add_argument('--beta', type=float, default=1.0); ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--bs', type=int, default=512); ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--seed', type=int, default=1)   # batch-sampler + init-noise seed -> decorrelated ensemble members
    ap.add_argument('--student-blocks', type=int, default=0, help='distill into a SMALLER net (blocks) for fast search/rollout; fresh init via KL+CE')
    ap.add_argument('--aw', action='store_true', help='advantage-weighted BC: weight CE by clip(1+score/48, 0.25, 3) — learn MORE from the teacher seat wins than its losses (offline-RL-lite; npz needs a score array from extract_top30 --scores)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import json
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(a.seed)
    cfg = json.loads(a.cfg)
    z = np.load(a.champ); obs, mask, act = z['obs'], z['mask'], z['act'].astype(np.int64)
    rng = np.random.RandomState(0); idx = rng.permutation(len(act)); nval = max(50, len(idx)//5)
    vi, ti = idx[:nval], idx[nval:]
    cobs, cmask, cact = augment(obs[ti], mask[ti], act[ti])         # 12x
    print(f"champ {len(act)} -> train(aug) {len(cact)} val {len(vi)}", flush=True)
    O = torch.from_numpy(cobs); M = torch.from_numpy(cmask); A = torch.from_numpy(cact)
    WT = None
    if a.aw:
        w = np.clip(1.0 + z['score'].astype(np.float32) / 48.0, 0.25, 3.0)
        WT = torch.from_numpy(np.tile(w[ti], 12))    # augment = 12 same-order blocks of ti
        print(f"AW weights: mean={w.mean():.2f} range[{w.min():.2f},{w.max():.2f}]", flush=True)
    vO = torch.from_numpy(obs[vi]); vM = torch.from_numpy(mask[vi]).float(); vA = act[vi]
    frozen = build(a.kind, **cfg).to(dev); frozen.load_state_dict(torch.load(a.base, map_location='cpu')); frozen.eval()
    for p in frozen.parameters(): p.requires_grad_(False)
    if a.student_blocks and a.student_blocks != cfg.get('blocks'):
        scfg = dict(cfg); scfg['blocks'] = a.student_blocks       # SMALL fast net for search/rollout
        student = build(a.kind, **scfg).to(dev)                    # fresh init; learns via KL(student||frozen)+CE
        print(f"cross-size distill: student blocks={a.student_blocks} <- teacher blocks={cfg.get('blocks')}", flush=True)
    else:
        student = build(a.kind, **cfg).to(dev); student.load_state_dict(torch.load(a.base, map_location='cpu'))
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    def agree():
        student.eval()
        with torch.no_grad():
            pr = student({'is_training':False,'obs':{'observation':vO.to(dev),'action_mask':vM.to(dev)}}).argmax(1).cpu().numpy()
        student.train(); return float((pr==vA).mean())
    print(f"beta={a.beta} steps={a.steps} | base agreement {agree():.3f}", flush=True)
    student.train(); n=len(cact); r=np.random.RandomState(a.seed)
    for step in range(1, a.steps+1):
        b = r.randint(0, n, size=a.bs)
        o=O[b].float().to(dev); mk=M[b].float().to(dev); y=A[b].to(dev)
        with torch.no_grad():
            tl = frozen({'is_training':False,'obs':{'observation':o,'action_mask':mk}})
        with torch.cuda.amp.autocast(enabled=(dev=='cuda')):
            sl = student({'is_training':True,'obs':{'observation':o,'action_mask':mk}})
            if WT is not None:
                wb = WT[b].to(dev)
                ce = (F.cross_entropy(sl, y, reduction='none') * wb).sum() / wb.sum()
            else:
                ce = F.cross_entropy(sl, y)
            loss = ce + a.beta*F.kl_div(F.log_softmax(sl,1), F.softmax(tl,1), reduction='batchmean')
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 400 == 0: print(f"  step {step}/{a.steps} agree={agree():.3f}", flush=True)
    torch.save(student.state_dict(), a.out, _use_new_zipfile_serialization=False)
    print(f"DONE agree={agree():.3f} -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
