"""
suit_aug.py — suit-permutation data augmentation for Chinese Standard Mahjong distillation.
The 3 number suits W/T/B (obs rows 0,1,2; tile idx 0-8/9-17/18-26; Chi color 'WTB') are FULLY
symmetric in MCR scoring, so permuting them is a label-preserving 6x data multiplier — the fix for
the distillation overfitting (train_agree 1.0 vs val 0.73 = data-starved). Honors (row 3) are fixed.

Action layout (feature.py): Pass0 Hu1 Play[2+t] Chi[36+color*21+w] Peng[99+t] Gang[133+t]
AnGang[167+t] BuGang[201+t], tile t in 0..33 (W0-8,T9-17,B18-26,F27-30,J31-33).
"""
import numpy as np
from itertools import permutations

OFF = {'Play': 2, 'Chi': 36, 'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201}

def _tile_perm(perm):
    """perm: tuple mapping suit-slot -> source suit (e.g. (1,2,0)). Return length-34 tile remap."""
    tp = np.arange(34)
    for new_s in range(3):
        src_s = perm[new_s]
        for r in range(9):
            tp[new_s * 9 + r] = src_s * 9 + r     # new suit row gets ranks from src suit
    return tp                                      # honors 27-33 unchanged

def action_perm(perm):
    """Return length-235 array A: A[new_action] = old_action (gather map for masks)."""
    tp = _tile_perm(perm)                          # tp[new_tile]=old_tile
    A = np.arange(235)
    A[0] = 0; A[1] = 1                             # Pass, Hu
    for base in (OFF['Play'], OFF['Peng'], OFF['Gang'], OFF['AnGang'], OFF['BuGang']):
        for nt in range(34):
            A[base + nt] = base + tp[nt]
    # Chi: 36 + color*21 + within ; new color slot gets src color's block
    for new_c in range(3):
        src_c = perm[new_c]
        for w in range(21):
            A[OFF['Chi'] + new_c * 21 + w] = OFF['Chi'] + src_c * 21 + w
    return A

def fwd_action_perm(perm):
    """Return F: F[old_action]=new_action (for remapping the chosen label)."""
    A = action_perm(perm); F = np.empty(235, np.int64)
    F[A] = np.arange(235)
    return F

PERMS = [p for p in permutations(range(3))]        # 6 suit permutations (incl identity)

def augment(obs, mask, act):
    """obs (N,38,4,9), mask (N,235) bool, act (N,) int -> 6x augmented arrays."""
    out_o, out_m, out_a = [], [], []
    for perm in PERMS:
        A = action_perm(perm)                      # gather: new_mask = old_mask[A]
        F = fwd_action_perm(perm)                  # new_act = F[old_act]
        rows = np.array([perm[0], perm[1], perm[2], 3])   # obs row order (suits permuted, honors fixed)
        out_o.append(obs[:, :, rows, :])
        out_m.append(mask[:, A])
        out_a.append(F[act])
    return (np.concatenate(out_o), np.concatenate(out_m), np.concatenate(out_a))

def _selftest():
    # synthetic consistency: a chosen action must remain legal under the permuted mask
    import os, sys
    d = np.load('/tmp/chunjiandu100.npz')
    o, m, a = d['obs'][:200], d['mask'][:200], d['act'][:200].astype(np.int64)
    # base legality: chosen act is legal
    assert m[np.arange(len(a)), a].all(), "base: some chosen acts illegal in their mask!"
    ao, am, aa = augment(o, m, a)
    assert ao.shape == (1200, 38, 4, 9) and am.shape == (1200, 235) and aa.shape == (1200,)
    # every augmented chosen action legal under augmented mask
    legal = am[np.arange(len(aa)), aa]
    assert legal.all(), f"AUG BUG: {(~legal).sum()} augmented acts illegal!"
    # identity perm (first block) must be unchanged
    assert (ao[:200] == o).all() and (am[:200] == m).all() and (aa[:200] == a).all(), "identity perm changed data!"
    # a non-identity perm: hand tile counts conserved (sum over suits preserved)
    assert abs(ao[200:400, 2].sum() - o[:, 2].sum()) < 1e-6, "hand channel mass not conserved"
    # all chosen acts are Play (distillation); verify they map to Play range
    assert ((aa >= OFF['Play']) & (aa < OFF['Chi'])).all(), "non-Play action in champ data?"
    print(f"selftest PASS: {len(a)}->{len(aa)} samples, all augmented acts legal, identity preserved.")

if __name__ == '__main__':
    _selftest()
