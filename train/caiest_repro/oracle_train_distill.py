"""
oracle_train_distill.py (Suphx oracle-guiding). Stage 1: train an ORACLE teacher on (50,4,9) =
38 public + 12 opponent-hand planes (perfect info). Stage 2: distill to a 38-plane PUBLIC student
via CE(expert) + alpha*KL(student||teacher) on the SAME states (student sees only the 38 public
planes). The hidden info is a TRAIN-time scaffold; the deployable student is public-info only.
Fuses the student to a torch-1.4-safe pkl + gauntlet-ready.

  python3 oracle_train_distill.py --data data/oracle_cooked.npz --epochs 8 --out /root/mahjong/ckpt/oracle_student.pkl
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import models_explore as ME

def build(in_planes, ch=128, blocks=40):
    ME.IN_PLANES = in_planes
    m = ME.ResBNCNN(channels=ch, blocks=blocks)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--blocks', type=int, default=40)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(a.data); obs = d['obs']; mask = d['mask']; act = d['act'].astype(np.int64)
    n = len(act); rng = np.random.RandomState(0); perm = rng.permutation(n)
    nval = min(40000, n // 10); vi, ti = perm[:nval], perm[nval:]
    print(f"oracle data {n:,} planes={obs.shape[1]} | train {len(ti)} val {len(vi)}", flush=True)
    O = torch.from_numpy(obs); M = torch.from_numpy(mask); A = torch.from_numpy(act)

    def batches(idx, bs, shuf=True):
        order = np.random.permutation(idx) if shuf else idx
        for i in range(0, len(order) - (bs if shuf else 0), bs):
            yield order[i:i+bs]

    def acc(model, planes):
        model.eval(); c = 0
        with torch.no_grad():
            for i in range(0, len(vi), 8192):
                b = vi[i:i+8192]
                o = O[b][:, :planes].float().to(dev); mk = M[b].float().to(dev)
                pr = model({'is_training': False, 'obs': {'observation': o, 'action_mask': mk}}).argmax(1)
                c += (pr.cpu() == A[b]).sum().item()
        model.train(); return c / len(vi)

    # ---- Stage 1: oracle teacher (50 planes) ----
    teacher = build(obs.shape[1], blocks=a.blocks).to(dev)
    opt = torch.optim.AdamW(teacher.parameters(), lr=a.lr, weight_decay=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    for e in range(a.epochs):
        t0=time.time(); teacher.train()
        for b in batches(ti, a.bs):
            o=O[b].float().to(dev); mk=M[b].float().to(dev); y=A[b].to(dev)
            with torch.cuda.amp.autocast(enabled=(dev=='cuda')):
                loss=F.cross_entropy(teacher({'is_training':True,'obs':{'observation':o,'action_mask':mk}}), y)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        print(f"[teacher] ep{e+1}/{a.epochs} val_acc(oracle)={acc(teacher,obs.shape[1]):.4f} ({time.time()-t0:.0f}s)", flush=True)
    teacher.eval()

    # ---- Stage 2: distill to 38-plane public student ----
    student = build(38, blocks=a.blocks).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=(dev=='cuda'))
    best=0.0
    for e in range(a.epochs):
        t0=time.time(); student.train()
        for b in batches(ti, a.bs):
            o_full=O[b].float().to(dev); mk=M[b].float().to(dev); y=A[b].to(dev)
            o_pub=o_full[:, :38]
            with torch.no_grad():
                tl=teacher({'is_training':False,'obs':{'observation':o_full,'action_mask':mk}})
            with torch.cuda.amp.autocast(enabled=(dev=='cuda')):
                sl=student({'is_training':True,'obs':{'observation':o_pub,'action_mask':mk}})
                ce=F.cross_entropy(sl, y)
                kl=F.kl_div(F.log_softmax(sl,1), F.softmax(tl.detach(),1), reduction='batchmean')
                loss=ce + a.alpha*kl
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        va=acc(student,38)
        if va>best:
            best=va; torch.save(student.state_dict(), a.out, _use_new_zipfile_serialization=False)
        print(f"[student] ep{e+1}/{a.epochs} val_acc(pub)={va:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # fuse best student -> deployable
    ME.IN_PLANES=38
    student.load_state_dict(torch.load(a.out, map_location='cpu'))
    fused=ME.fuse_resbn(student.cpu().eval())
    fout=a.out.replace('.pkl','_fused.pkl')
    torch.save(fused.state_dict(), fout, _use_new_zipfile_serialization=False)
    print(f"DONE student best_val={best:.4f} -> {a.out} | fused -> {fout}", flush=True)

if __name__ == '__main__':
    main()
