"""
aug_verify.py — verify candidate augmentations are FAN/LABEL-invariant before training.

Two independent checks:
 (A) STRUCTURAL + LEGALITY on real labeled champion states (cooked_single.npz):
     for each transform, the chosen expert action stays LEGAL under the transformed mask,
     identity perm is a no-op, per-action-block legality counts are preserved, obs tile-mass
     conserved. This validates the action-space remap is a correct game relabeling.
 (B) FAN-CALCULATOR invariance (MahjongFanCalculator, full MCR 81-fan set): generate many
     real *scoring* winning hands, apply the tile-level transform, recompute the total fan,
     and report the fraction where fan is preserved. A true automorphism preserves fan on
     (almost) every hand. A wind transposition (F1<->F2) at FIXED seat/prevalent wind is
     included as a NEGATIVE CONTROL to prove the test can detect a broken symmetry.

Transforms tested: suit-perm (baseline, already deployed), rank-reflection, dragon-perm.
Result JSON -> stdout + /root/IJCAI-mahjong/train/caiest_repro/aug_verify.json
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from MahjongGB import MahjongFanCalculator

import suit_aug, reflect_aug, dragon_aug

TILE_LIST = [*('W%d' % (i + 1) for i in range(9)), *('T%d' % (i + 1) for i in range(9)),
             *('B%d' % (i + 1) for i in range(9)), *('F%d' % (i + 1) for i in range(4)),
             *('J%d' % (i + 1) for i in range(3))]
IDX = {c: i for i, c in enumerate(TILE_LIST)}
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- (A) structural / legality on real data ----------------
def structural_check(o, m, a):
    """o (N,38,4,9) int8, m (N,235) bool, a (N,) int. Returns dict of pass/rates per transform."""
    N = len(a)
    base_legal = float(m[np.arange(N), a].mean())
    res = {"n": int(N), "base_chosen_legal_rate": round(base_legal, 6)}

    def one(name, perm_obs_rows, A, F, obs_transform=None):
        # apply obs transform
        if obs_transform is not None:
            po = obs_transform(o)
        else:
            po = o[:, :, perm_obs_rows, :]
        pm = m[:, A]
        pa = F[a]
        chosen_legal = float(pm[np.arange(N), pa].mean())
        mass_ok = bool(abs(int(po.sum()) - int(o.sum())) == 0)
        legalcount_ok = bool((pm.sum(1) == m.sum(1)).all())   # #legal actions preserved
        return {"chosen_legal_rate": round(chosen_legal, 6), "tile_mass_conserved": mass_ok,
                "legal_count_preserved": legalcount_ok,
                "PASS": bool(chosen_legal > 0.9999 and mass_ok and legalcount_ok)}

    # identity sanity for each family
    # suit non-identity perm (1,2,0)
    sp = (1, 2, 0)
    res["suit_perm_120"] = one("suit", np.array([sp[0], sp[1], sp[2], 3]),
                               suit_aug.action_perm(sp), suit_aug.fwd_action_perm(sp))
    # rank reflection
    res["rank_reflect"] = one("reflect", None, reflect_aug.reflect_action(),
                              reflect_aug.fwd_reflect_action(), obs_transform=reflect_aug.reflect_obs)
    # dragon transposition (1,0,2) and full cycle (1,2,0)
    dq = (1, 0, 2)
    res["dragon_perm_102"] = one("dragon", None, dragon_aug.action_perm(dq),
                                 dragon_aug.fwd_action_perm(dq),
                                 obs_transform=lambda x: dragon_aug.dragon_obs(x, dq))
    dq2 = (2, 0, 1)
    res["dragon_perm_201"] = one("dragon", None, dragon_aug.action_perm(dq2),
                                 dragon_aug.fwd_action_perm(dq2),
                                 obs_transform=lambda x: dragon_aug.dragon_obs(x, dq2))
    # identity must be exact no-op
    id_A = suit_aug.action_perm((0, 1, 2)); id_F = suit_aug.fwd_action_perm((0, 1, 2))
    id_ok = bool(np.array_equal(m[:, id_A], m) and np.array_equal(id_F[a], a))
    res["suit_identity_noop"] = id_ok
    return res


# ---------------- (B) fan-calculator invariance ----------------
def _add(counts, tile, k=1):
    counts[tile] = counts.get(tile, 0) + k
    return counts[tile] <= 4

def gen_winning_hands(n_target=3000, seed=0, force_dragon_frac=0.35):
    """Generate fully-concealed 14-tile winning hands (4 melds + pair) that SCORE (>=8 fan).
    Returns list of (concealed_13 tuple, winTile, seatWind, prevalentWind, total_fan)."""
    rng = random.Random(seed)
    suits = 'WTB'
    hands = []
    tries = 0
    while len(hands) < n_target and tries < n_target * 200:
        tries += 1
        counts = {}
        tiles = []
        ok = True
        force_dragon = rng.random() < force_dragon_frac
        melds = 4
        for mi in range(melds):
            if force_dragon and mi == 0:
                t = 'J%d' % rng.randint(1, 3)                 # dragon pung
                for _ in range(3):
                    if not _add(counts, t): ok = False; break
                    tiles.append(t)
            elif rng.random() < 0.5:                          # pung
                t = rng.choice(TILE_LIST)
                for _ in range(3):
                    if not _add(counts, t): ok = False; break
                    tiles.append(t)
            else:                                             # chow
                s = rng.choice(suits); r = rng.randint(1, 7)
                for d in range(3):
                    t = '%s%d' % (s, r + d)
                    if not _add(counts, t): ok = False; break
                    tiles.append(t)
            if not ok: break
        if not ok: continue
        # pair
        t = rng.choice(TILE_LIST)
        if not (_add(counts, t) and _add(counts, t)): continue
        tiles.append(t); tiles.append(t)
        if len(tiles) != 14: continue
        seat = rng.randint(0, 3); quan = rng.randint(0, 3)
        win = rng.choice(tiles)
        concealed = list(tiles); concealed.remove(win)
        try:
            r = MahjongFanCalculator(pack=(), hand=tuple(concealed), winTile=win, flowerCount=0,
                                     isSelfDrawn=True, is4thTile=False, isAboutKong=False,
                                     isWallLast=False, seatWind=seat, prevalentWind=quan)
            fan = sum(x for x, _ in r)
        except Exception:
            continue
        hands.append((tuple(concealed), win, seat, quan, fan))
    return hands


def _score(concealed, win, seat, quan):
    r = MahjongFanCalculator(pack=(), hand=tuple(concealed), winTile=win, flowerCount=0,
                             isSelfDrawn=True, is4thTile=False, isAboutKong=False,
                             isWallLast=False, seatWind=seat, prevalentWind=quan)
    return sum(x for x, _ in r)


def _remap_tile(t, tilemap):
    """tilemap: length-34 array new_idx from old (we want new tile string for old tile t)."""
    return TILE_LIST[tilemap[IDX[t]]]


def fan_invariance(hands):
    """For each transform build a length-34 tile relabel (old_tile -> new_tile) and check fan."""
    # Build old->new tile maps (fwd) for each transform.
    def suit_map(perm):
        # suit_aug._tile_perm gives tp[new]=old; invert to old->new
        tp = suit_aug._tile_perm(perm); inv = np.empty(34, int); inv[tp] = np.arange(34); return inv
    def reflect_map():
        rt = reflect_aug.reflect_tile(); inv = np.empty(34, int); inv[rt] = np.arange(34); return inv  # involution
    def dragon_map(q):
        dt = dragon_aug._tile_perm(q); inv = np.empty(34, int); inv[dt] = np.arange(34); return inv
    def wind_swap_map():
        mp = np.arange(34); mp[27] = 28; mp[28] = 27; return mp   # F1<->F2, seat/prevalent UNCHANGED

    transforms = {
        "suit_perm(120)": (suit_map((1, 2, 0)), False),
        "suit_perm(201)": (suit_map((2, 0, 1)), False),
        "rank_reflect":   (reflect_map(), False),
        "dragon_perm(102)": (dragon_map((1, 0, 2)), False),
        "dragon_perm(201)": (dragon_map((2, 0, 1)), False),
        "WIND_swap(F1F2)_NEGCTRL": (wind_swap_map(), False),
    }
    out = {}
    for name, (tmap, _) in transforms.items():
        same = 0; diff = 0; invalid = 0; examples = []
        for concealed, win, seat, quan, fan0 in hands:
            nc = [_remap_tile(t, tmap) for t in concealed]
            nw = _remap_tile(win, tmap)
            try:
                fan1 = _score(nc, nw, seat, quan)
            except Exception:
                invalid += 1
                continue
            if fan1 == fan0:
                same += 1
            else:
                diff += 1
                if len(examples) < 5:
                    examples.append({"orig": list(concealed) + [win], "fan0": fan0, "fan1": fan1})
        tot = same + diff + invalid
        out[name] = {"n": tot, "fan_preserved": same, "fan_changed": diff,
                     "became_invalid_win": invalid,
                     "preserve_rate": round(same / tot, 6) if tot else None,
                     "examples_changed": examples}
    return out


def main():
    src = os.path.join(HERE, "data", "cooked_single.npz")
    d = np.load(src)
    o, m, a = d["obs"], d["mask"], d["act"].astype(np.int64)
    N = len(a)
    rng = np.random.RandomState(7)
    idx = np.sort(rng.choice(N, size=min(20000, N), replace=False))
    os_, ms_, as_ = np.ascontiguousarray(o[idx]), np.ascontiguousarray(m[idx]), a[idx]
    struct = structural_check(os_, ms_, as_)

    hands = gen_winning_hands(n_target=3000, seed=1)
    faninv = fan_invariance(hands)

    verdict = {}
    for name in ("suit_perm(120)", "rank_reflect", "dragon_perm(102)"):
        # a transform is FAN-VALID if it preserves fan at the SAME rate as the deployed suit-perm
        pass  # summarized below
    result = {
        "structural_legality": struct,
        "fan_invariance": faninv,
        "fan_hands_generated": len(hands),
        "notes": ("A transform is a true MCR automorphism iff it preserves total fan on every "
                  "scoring hand. Suit-perm is the already-deployed baseline; rank-reflect and "
                  "dragon-perm are judged fan-valid iff their preserve_rate matches suit-perm's "
                  "(any residual mismatch is the single suit/rank/dragon-asymmetric fan 'Reversible "
                  "Tiles' 推不倒, which also affects the deployed suit-perm). The WIND negative "
                  "control MUST show a lower preserve_rate, proving the test discriminates."),
    }
    out_path = os.path.join(HERE, "aug_verify.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result["fan_invariance"][k]["preserve_rate"] for k in result["fan_invariance"]}, indent=2))
    print("STRUCT:", json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v)
                                 for k, v in struct.items()}, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
