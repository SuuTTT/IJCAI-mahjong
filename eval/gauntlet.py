"""
gauntlet.py — measure candidates against a DIVERSE opponent set (different architectures/styles),
not their own near-twin base. Net-vs-gauntlet separates strong policies where vs-base ties
(the monoculture problem). This is the local stand-in for the Botzone ladder.

Each (candidate, opponent) pair plays G games (2v2, seats rotated to cancel position bias) through
the official judge. We report each candidate's TOTAL net + wins across all opponents, and the
per-opponent breakdown. Usage: edit CANDIDATES / OPPONENTS below, then:
    OPENBLAS_NUM_THREADS=1 python3 eval/gauntlet.py [games_per_pair]
"""
import sys, os
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from run_match_kr import run_match_kr
from data.log_collector import make_wall

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
R = os.path.join(REPO, "train", "caiest_repro")
EB = os.path.join(R, "explore_bot.py")
CK = os.path.join(R, "arch_ck")

def srv(kind, ckpt, cfg='{"channels":128,"blocks":40}'):
    return (f"EXP_KIND={kind} EXP_CFG='{cfg}' "
            f"CAIEST_MODEL={ckpt} BOTZONE_JSON=0 python3 {EB}")

CFG40 = '{"channels":128,"blocks":40}'
# DIVERSE opponents — different architectures => different play styles => real separation
OPPONENTS = [
    ("cnn16",   srv("cnn",      f"{CK}/base_16x128_final.pkl",         "{}")),
    ("cnnattn", srv("cnn_attn", f"{CK}/explore/cnnattn.pkl",           "{}")),
    ("resbn24", srv("resbn",    f"{CK}/explore/resbn24.pkl",   '{"channels":128,"blocks":24}')),
    ("resbn56", srv("resbn",    f"{CK}/explore/resbn56.pkl",   '{"channels":128,"blocks":56}')),
    ("w192",    srv("resbn",    f"{CK}/explore/resbnw192.pkl", '{"channels":192,"blocks":24}')),
]
# candidates to rank (all resbn40-family — the ones that "tied" the base in monoculture eval)
EB2 = os.path.join(R, "ensemble_bot.py")
def ens(paths):
    return (f"MODELS={','.join(paths)} BOTZONE_JSON=0 python3 {EB2}")

CANDIDATES = [
    ("distill",    srv("resbn", f"{CK}/explore/resbn40_distill100b.pkl", CFG40)),   # current deploy floor
    ("curriculum", srv("resbn", f"{CK}/explore/resbn40_cl.pkl",          CFG40)),   # curriculum-RL result
    ("base",       srv("resbn", f"{CK}/explore/resbn40.pkl",             CFG40)),   # control
]


def pair_net(A, B, G, seed0):
    """A vs B over G games, 2v2 rotated. Return (netA, winsA, winsB, draws, illA)."""
    layouts = [([0, 2], [A, B, A, B]), ([1, 3], [B, A, B, A])]
    netA = wA = wB = draws = illA = 0
    for g in range(G):
        aseats, bots = layouts[g % 2]
        r = run_match_kr([{"cmd": c, "kr": True} for c in bots],
                         wall_json=make_wall(seed0 + g), quan=0, timeout=12)
        sc = r["scores"]
        for s in range(4):
            if s in aseats: netA += sc[s]
        for s in range(4):
            if sc[s] == -30 and sum(1 for x in sc if x == 10) == 3 and s in aseats:
                illA += 1
        w = max(range(4), key=lambda i: sc[i]) if max(sc) > 0 else -1
        if w == -1: draws += 1
        elif w in aseats: wA += 1
        else: wB += 1
    return netA, wA, wB, draws, illA


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    print(f"GAUNTLET: {len(CANDIDATES)} candidates x {len(OPPONENTS)} diverse opponents x {G} games\n")
    results = {}
    for cname, ccmd in CANDIDATES:
        tot_net = tot_w = tot_l = tot_d = tot_ill = 0
        row = []
        for oname, ocmd in OPPONENTS:
            net, w, l, d, ill = pair_net(ccmd, ocmd, G, seed0=SEED0)
            tot_net += net; tot_w += w; tot_l += l; tot_d += d; tot_ill += ill
            row.append(f"{oname}:{net:+d}({w}-{l})")
            print(f"  {cname:9s} vs {oname:9s}  net={net:+5d}  W{w}-L{l}  draws={d}  ill={ill}", flush=True)
        results[cname] = (tot_net, tot_w, tot_l, tot_d, tot_ill)
        print(f"  --> {cname:9s} TOTAL net={tot_net:+6d}  W{tot_w}-L{tot_l}  draws={tot_d}  ill={tot_ill}\n", flush=True)
    print("\n==== RANKING (by total net vs diverse gauntlet) ====")
    for cname, (net, w, l, d, ill) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {cname:9s}  net={net:+6d}  wins={w:3d}  losses={l:3d}  draws={d}  illegal={ill}")


if __name__ == '__main__':
    main()
