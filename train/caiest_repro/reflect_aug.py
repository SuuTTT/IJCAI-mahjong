"""
reflect_aug.py — tile-reflection symmetry (1<->9,2<->8,3<->7,4<->6,5 fixed) for CSM, an EXACT
label-preserving 2x augmentation on top of suit_aug's 6x (-> 12x total). Number suits W/T/B
(tile idx 0-8/9-17/18-26, obs rows 0,1,2) reverse rank; honors (27-33, row 3) unchanged.

Action layout: Pass0 Hu1 Play[2+t] Chi[36+color*21+w] Peng[99+t] Gang[133+t] AnGang[167+t] BuGang[201+t].
Chi w = (mid-1)*3 + offer, mid = middle rank 1..7 (0-indexed within suit), offer in {0,1,2}.
Reflection: rank r->8-r => mid m -> 8-m (so 0-idx mid 1..7 -> 7..1), offer o -> 2-o.

Run `python3 reflect_aug.py` to self-test (involution + block-permutation + obs round-trip).
"""
import numpy as np

OFF = {'Play': 2, 'Chi': 36, 'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201}

def reflect_tile():
    """34-remap rt[new]=old for tile reflection (rank r -> 8-r per number suit; honors fixed)."""
    rt = np.arange(34)
    for s in range(3):
        for r in range(9):
            rt[s * 9 + r] = s * 9 + (8 - r)
    return rt                                        # honors 27-33 unchanged (involutive)

def reflect_action():
    """235-remap A[new_action]=old_action under tile reflection (gather map for masks)."""
    rt = reflect_tile()
    A = np.arange(235)
    A[0] = 0; A[1] = 1
    for base in (OFF['Play'], OFF['Peng'], OFF['Gang'], OFF['AnGang'], OFF['BuGang']):
        for t in range(34):
            A[base + t] = base + rt[t]
    for c in range(3):                               # Chi: color fixed; reflect (mid,offer)
        for mid in range(1, 8):                      # 0-indexed middle rank 1..7
            for off in range(3):
                w = (mid - 1) * 3 + off
                nm = 8 - mid; noff = 2 - off          # reflected middle & offer
                nw = (nm - 1) * 3 + noff
                A[OFF['Chi'] + c * 21 + nw] = OFF['Chi'] + c * 21 + w
    return A

def fwd_reflect_action():
    A = reflect_action(); F = np.empty(235, np.int64); F[A] = np.arange(235); return F

def reflect_obs(o):
    """o (...,38,4,9) -> reverse rank cols for number rows 0,1,2; honor row 3 unchanged."""
    out = o.copy()
    out[..., :3, :] = o[..., :3, ::-1]
    return out

if __name__ == '__main__':
    A = reflect_action(); F = fwd_reflect_action()
    # 1) involution: reflecting twice = identity over all 235 actions
    assert np.array_equal(A[A], np.arange(235)), "action reflection NOT involutive"
    assert np.array_equal(F, A), "fwd should equal A for an involution"
    # 2) each action block maps onto itself (no leakage across action types)
    blocks = [(0,1),(1,2),(OFF['Play'],OFF['Play']+34),(OFF['Chi'],OFF['Chi']+63),
              (OFF['Peng'],OFF['Peng']+34),(OFF['Gang'],OFF['Gang']+34),
              (OFF['AnGang'],OFF['AnGang']+34),(OFF['BuGang'],OFF['BuGang']+34)]
    for lo,hi in blocks:
        assert set(A[lo:hi].tolist()) == set(range(lo,hi)), f"block {lo}:{hi} leaks"
    # 3) tile reflection involutive + honors fixed
    rt = reflect_tile(); assert np.array_equal(rt[rt], np.arange(34)); assert np.array_equal(rt[27:], np.arange(27,34))
    # 4) obs round-trip
    o = np.random.RandomState(0).randint(0,2,(38,4,9)).astype(np.int8)
    assert np.array_equal(reflect_obs(reflect_obs(o)), o), "obs reflection not involutive"
    assert np.array_equal(reflect_obs(o)[:, 3, :], o[:, 3, :]), "honor row (dim -2 idx 3) changed"
    assert not np.array_equal(reflect_obs(o)[:, 0, :], o[:, 0, :]), "number row 0 should reverse"
    # 5) spot-check semantics: Play W1 (t=0) <-> Play W9 (t=8); Play T5(t=13) fixed
    assert A[OFF['Play']+8] == OFF['Play']+0 and A[OFF['Play']+0] == OFF['Play']+8
    assert A[OFF['Play']+13] == OFF['Play']+13     # T5 (idx 9+4) reflects to itself
    assert A[OFF['Play']+27] == OFF['Play']+27     # honor F1 fixed
    print("ALL SELF-TESTS PASS: reflection is exact + involutive (12x aug ready)")
