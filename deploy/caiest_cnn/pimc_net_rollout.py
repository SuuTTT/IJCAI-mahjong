# pimc_net_rollout.py — net-driven rollout for PIMC (the variant that addresses the greedy-rollout
# null: a converting rollout policy). The fast policy net drives every seat's discard via the
# byte-exact per-seat obs encoder (pimc_obs); MahjongGB checks Hu (self-draw + robbing); at a depth
# limit the value net evaluates OUR resulting state. Returns our MCR duplicate score (terminal) or
# the value-net leaf estimate (truncated) — so rollouts reflect real conversion, not shanten-racing.
import os
import numpy as np

import pimc_obs as PO
import csm_rollout as CR          # reuse _fan_count, _scores_* (verified scoring)

_TORCH = None
_FAST = None
_VAL = None
DEPTH = int(os.environ.get('CAIEST_PIMC_DEPTH', '16'))     # plies before value-net leaf
VSCALE = float(os.environ.get('CAIEST_PIMC_VSCALE', '1.0'))


def _load():
    global _TORCH, _FAST, _VAL
    if _TORCH is not None:
        return
    import torch
    _TORCH = torch
    from model_resfused import ResFused
    fp = os.environ.get('CAIEST_PIMC_FAST')
    fb = int(os.environ.get('CAIEST_PIMC_FAST_BLOCKS', '8'))
    fc = int(os.environ.get('CAIEST_PIMC_FAST_CH', '64'))
    net = ResFused(channels=fc, blocks=fb)
    net.load_state_dict(torch.load(fp, map_location='cpu')); net.eval()
    _FAST = net
    vp = os.environ.get('CAIEST_PIMC_VAL')
    if vp:
        import value_search as VS
        _VAL = VS.load(vp, blocks=40)


def _net_discard(hand, obs):
    """argmax legal discard from the fast policy net. obs (38,4,9); returns a tile string."""
    mask = PO.legal_play_mask(hand)                       # only Play actions for held tiles
    with _TORCH.no_grad():
        x = {'obs': {'observation': _TORCH.from_numpy(obs[None].astype(np.float32)),
                     'action_mask': _TORCH.from_numpy(mask[None].astype(np.float32))}}
        lg = _FAST(x)[0].numpy()
    a = int(lg[2:36].argmax())                            # Play block = indices 2..35
    return PO.TILE_LIST[a]


def rollout_once_net(my_hand_after, world, packs, seatwinds, prevalent):
    """One net-driven, opponent-aware rollout. Our seat = index 0, already discarded (my_hand_after).
    Returns our MCR score (if someone Hus) or the value-net leaf estimate (truncated draw)."""
    hands = [list(my_hand_after), list(world['hands'][1]), list(world['hands'][2]), list(world['hands'][3])]
    discards = [[], [], [], []]
    wall = list(world['wall'])
    shown = {}
    cur = 1                                               # after our discard, player 1 acts
    for ply in range(DEPTH):
        if not wall:
            break
        wallLast = len(wall) <= 1
        t = wall.pop(); hands[cur].append(t)
        is4 = shown.get(t, 0) == 4
        he = list(hands[cur]); he.remove(t)
        if CR._fan_count(packs[cur], he, t, True, seatwinds[cur], prevalent, is4, wallLast):
            return CR._scores_self_draw(cur, CR._fan_count(packs[cur], he, t, True, seatwinds[cur], prevalent, is4, wallLast))[0]
        obs = PO.obs_for_seat(cur, hands, discards, packs, seatwinds, prevalent)
        d = _net_discard(hands[cur], obs)
        hands[cur].remove(d); discards[cur].append(d); shown[d] = shown.get(d, 0) + 1
        is4d = shown[d] == 4
        for r in range(4):
            if r == cur:
                continue
            fc = CR._fan_count(packs[r], hands[r], d, False, seatwinds[r], prevalent, is4d, wallLast)
            if fc:
                return CR._scores_discard_win(r, cur, fc)[0]
        cur = (cur + 1) % 4
    # truncated -> value-net leaf on OUR current state (our hand may have grown via draws if cur cycled)
    if _VAL is None:
        return 0.0
    obs0 = PO.obs_for_seat(0, hands, discards, packs, seatwinds, prevalent)
    with _TORCH.no_grad():
        v = float(_VAL.v(_TORCH.from_numpy(obs0[None].astype(np.float32)))[0])
    return v * VSCALE
