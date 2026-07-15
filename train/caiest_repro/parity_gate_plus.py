"""
parity_gate_plus.py — strength gate vs moyu for a candidate trained on AUGMENTED features.

Candidate seats are fed (38+P) planes (base caiest + featplus[sets]); moyu seats are fed the
base 38 planes. Per-seat flavor is wired through SimPlus._obs_mask -> the candidate's extra
feature extractor is ACTUALLY USED in the gate (no-op trap closed). Bias-corrected duplicate:
each seed plays cand@{0,2} and cand@{1,3} on the same wall.

NO-OP GUARD (run before gating): cand md5 != moyu, AND on a handful of states the cand's
masked-argmax with extra planes differs from feeding it zero extra planes (proves the planes
are read). Reported in the JSON ('noop_guard').

  python3 parity_gate_plus.py --cand CAND.pkl --sets A --ref moyu.pkl \
      --games 400 --workers 64 --seed0 50000 --out gate.json
"""
import os, sys, json, argparse, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn_plus import SimPlus
from models_explore import build as build_base
from models_plus import build_plus
import featplus

torch.set_num_threads(1)
_CACHE = {}

def _md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()

def _load_base(path, blocks=40, channels=128):
    key = ('base', path)
    if key not in _CACHE:
        m = build_base('resbn', channels=channels, blocks=blocks)
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith(('stem', 'body', 'foot')) for k in sd):
            sd = sd['state_dict']
        m.load_state_dict(sd); m.eval(); _CACHE[key] = m
    return _CACHE[key]

def _load_plus(path, in_planes, blocks=40, channels=128):
    key = ('plus', path, in_planes)
    if key not in _CACHE:
        m = build_plus(in_planes=in_planes, channels=channels, blocks=blocks)
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith(('stem', 'body', 'foot')) for k in sd):
            sd = sd['state_dict']
        m.load_state_dict(sd); m.eval(); _CACHE[key] = m
    return _CACHE[key]

def _greedy(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({'is_training': False, 'obs': {
                'observation': torch.from_numpy(np.ascontiguousarray(obs)).float(),
                'action_mask': torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn

_G = {}
def _work(arg):
    seed, cand, ref, sets, blocks = arg
    P = featplus.total_extra(sets)
    fc = _greedy(_load_plus(cand, 38 + P, blocks)); fr = _greedy(_load_base(ref, blocks))
    # seating A: cand @ {0,2}
    ss_a = ['', '', '', '']; ss_a[0] = sets; ss_a[2] = sets
    sim = SimPlus([fc, fr, fc, fr], seed=seed, quan=0, learner_seats=[0, 2], cnn=True, seat_sets=ss_a); sim.play()
    a_cand = sim.scores[0] + sim.scores[2]; a_ref = sim.scores[1] + sim.scores[3]
    # seating B: cand @ {1,3}
    ss_b = ['', '', '', '']; ss_b[1] = sets; ss_b[3] = sets
    sim = SimPlus([fr, fc, fr, fc], seed=seed, quan=0, learner_seats=[1, 3], cnn=True, seat_sets=ss_b); sim.play()
    b_cand = sim.scores[1] + sim.scores[3]; b_ref = sim.scores[0] + sim.scores[2]
    return a_cand, a_ref, b_cand, b_ref

def _noop_guard(cand, ref, sets, blocks):
    """Verify cand md5 != ref AND cand reads the extra planes (argmax changes when planes zeroed)."""
    P = featplus.total_extra(sets)
    m = _load_plus(cand, 38 + P, blocks)
    # run a few real sim states through cand with real extra planes vs zeroed extra planes
    ss = [sets, sets, sets, sets]  # all 4 seats use the plus model -> all need extra planes
    diff = 0; tot = 0
    # collect real decision-point obs by capturing them inside a played game (seat 0 = plus)
    captured = []
    class _CapSim(SimPlus):
        def _ask(self, seat):
            if seat == 0 and len(captured) < 40:
                o, mk = self._obs_mask(0)
                if mk.sum() > 0: captured.append((o.copy(), mk.copy()))
            return super()._ask(seat)
    for s in range(6):
        if len(captured) >= 25: break
        sm = _CapSim([_greedy(m)] * 4, seed=7000 + s, quan=0, learner_seats=[0], cnn=True, seat_sets=ss)
        sm.play()
    for o, mk in captured[:25]:
        with torch.no_grad():
            a1 = m({'is_training': False, 'obs': {'observation': torch.from_numpy(o[None]).float(),
                    'action_mask': torch.from_numpy(mk[None])}}).numpy().flatten().argmax()
            o0 = o.copy(); o0[38:] = 0.0   # zero the extra planes
            a0 = m({'is_training': False, 'obs': {'observation': torch.from_numpy(o0[None]).float(),
                    'action_mask': torch.from_numpy(mk[None])}}).numpy().flatten().argmax()
        tot += 1
        if int(a1) != int(a0): diff += 1
    md5_diff = (_md5(cand) != _md5(ref))
    return dict(md5_differs=md5_diff, states=tot, argmax_changed_when_planes_zeroed=diff,
                reads_planes=(diff > 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True); ap.add_argument('--ref', required=True)
    ap.add_argument('--sets', required=True)  # e.g. 'A','B','C','AC'
    ap.add_argument('--blocks', type=int, default=40)
    ap.add_argument('--games', type=int, default=400); ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--seed0', type=int, default=50000); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time()
    guard = _noop_guard(a.cand, a.ref, a.sets, a.blocks)
    args = [(a.seed0 + i, a.cand, a.ref, a.sets, a.blocks) for i in range(a.games)]
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=4)
    cand_net = sum(r[0] + r[2] for r in res); ref_net = sum(r[1] + r[3] for r in res)
    ngames = len(res) * 2
    seatA = sum(r[0] - r[1] for r in res) / len(res) if res else 0.0
    seatB = sum(r[2] - r[3] for r in res) / len(res) if res else 0.0
    edge = (cand_net - ref_net) / ngames if ngames else 0.0
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref), sets=a.sets,
               games=ngames, seeds=len(res), cand_net=int(cand_net), ref_net=int(ref_net),
               edge_per_game=round(edge, 4), seatA_edge_per_seed=round(seatA, 4),
               seatB_edge_per_seed=round(seatB, 4), seconds=round(time.time() - t0, 1),
               noop_guard=guard, seed0=a.seed0)
    with open(a.out, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    if ngames == 0: print('FAIL: n=0 games', flush=True); sys.exit(2)

if __name__ == '__main__':
    main()
