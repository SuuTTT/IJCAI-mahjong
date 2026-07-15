"""T2-JAX v3 LEAGUE -- opponent-pool self-play on top of the v2 fused PPO trainer.

WHY (breaks the F3 wall, see docs/mahjong_rl_findings.md F3 + "What to try next" #1/#2):
  v2 trains seat-0 against seats 1-3 = FROZEN copies of the kdens3 SL anchor -- opponents
  exactly as strong as the learner, so there is no exploitable weakness to climb and the
  reward gradient toward "beat them" is weak/noisy. v3 replaces the frozen-self field with
  a growing OPPONENT POOL: periodically snapshot the learner into a pool (last ~N snapshots
  + the kdens3 anchor always in), and each rollout sample the opponents' policies from that
  pool (a mix of recent-strong / older / anchor). This gives the learner varied,
  sometimes-weaker opponents to beat -- the "league" effect.

V3 CHANGES vs baselines/mahjong_t2_jax_v2.py (everything else IDENTICAL: PopArt critic,
KL-leash + trust-region rollback, kdens3 warm-start, illegal-Hu-forfeit + meld-cap gates,
frozen-anchor strength eval):
  L1. OPPONENT POOL. A host-side PoolManager holds up to --pool-cap learner snapshots plus
      the kdens3 anchor (always slot 0, never evicted). Every --pool-refresh updates the
      current learner params are snapshotted in (ring-buffer eviction of the oldest
      non-anchor snapshot). --pool-cap 0 => pool = {anchor} only, which EXACTLY reproduces
      v2 (opponents = frozen anchor); the league machinery is a clean superset.
  L2. PER-ROLLOUT, PER-SEAT OPPONENT SAMPLING. At each training rollout we sample THREE
      pool members (one per opponent seat 1/2/3) and pass them into the jitted rollout as a
      stacked pytree (leaves gain a leading axis of 3). The opponent forward is a
      vmap-over-seat of pforward, then reshaped game-major to match v2's opponent RNG stream
      EXACTLY -- so with pool={anchor} the opponent actions are bit-identical to v2. See the
      JIT/VMAP NOTE below for why per-seat (not per-game) is the tractable granularity.
      Sampling mix (per seat, independent): with prob --p-anchor use the kdens3 anchor; else
      sample a snapshot with recency-linear weights (recent snapshots favoured). Documented,
      tunable.
  L3. KL-ANNEAL. --kl-target is the START target; --kl-target-final / --kl-anneal-updates
      give a linear tight->loose schedule (F3 "What to try next" #1: KL headroom is unused,
      widen it once the critic is warm). The trust-region reject threshold, beta ladder, and
      PPO gate all use the annealed per-update target. Defaults reproduce a constant target.
  The frozen-anchor strength EVAL is UNCHANGED: eval rollouts always play seat-0 vs seats
  1-3 = the frozen kdens3 anchor (FROZEN), so win_rate/mean_score/dealin remain measured on
  the SAME fixed yardstick and are directly comparable to v2 / the wide-explore sweep.

JIT/VMAP NOTE (per-seat vs per-game pooling). Full per-(game,seat) pooling would need a
different pool member for each of the B*3 opponent slots -> a gather of up to B*3 full
resnet param-sets onto the device (each set ~48 MB fp32) -- impossible. Running ALL P pool
members over ALL obs and selecting costs P x the opponent forward. Neither is tractable.
Per-ROLLOUT, per-SEAT (3 sampled param-sets, held fixed for the whole B x T scan) needs
only 3 param copies on device (~150 MB) and is a plain vmap-over-3 -- fully jittable, no
dynamic gather inside the scan, opponents still vary across the 3 seats AND across updates
(as the pool churns). This is strictly richer than the task's fallback ("one opponent for
all 3 seats per rollout") and is what we do.


V2 CHANGES vs baselines/mahjong_t2_jax.py (isolated critic fixes; policy path is
IDENTICAL -- same net, KL<=0.05 leash, adaptive beta, trust-region rollback, PPO
clip, entropy coef, on-device Hu check, reward=score0/8 with fan>=8 forfeit):
  A. RETURN NORMALIZATION (PopArt-lite, --popart 1 default). Running EMA estimates
     ret_mean/ret_std (decay 0.99, updated once per training update from that
     update's raw GAE return targets) are carried in the train state.  The critic
     predicts a NORMALIZED value v_norm; everywhere it feeds GAE/advantage it is
     DENORMALIZED (v_raw = v_norm*ret_std + ret_mean) so ALL GAE/advantage math stays
     in RAW reward units.  The value-loss target is the NORMALIZED return
     (rt-ret_mean)/(ret_std+1e-8) vs v_norm -- O(1) targets the fresh net can regress.
     --popart 0 pins ret_mean=0,ret_std=1 (exactly reproduces the baseline critic).
  B. STRONGER VALUE LEARNING (--value-epochs 6). After the shared PPO epochs, extra
     VALUE-ONLY gradient steps (policy frozen) minimise the normalized value loss.
  The logged ``vloss`` is always RAW-units (norm_loss * ret_std^2) for direct
  comparability with the baseline control.


This is the JAX-track payoff: it REPLACES the CPU-bound torch T2 trainer
(``baselines/mahjong_t2.py``, ~313 env-steps/s -- the env step was Python/CPU and
dominated wall time) with a fused rollout where BOTH the vmapped MCR env
(``mahjong/jax_env.py``) AND the kdens3 policy live on the GPU inside a single
``lax.scan``.  No host round-trip except the terminal fan callback (once per
rollout, not per step).

It is a NEW file; it imports ``mahjong/jax_env.py`` + ``baselines/
mahjong_jax_policy.py`` and edits neither.

ALGORITHM (mirrors baselines/mahjong_t2.py, the torch reference):
  * Learner  = seat 0, policy = flax kdens3 net (init from kdens_s0_fp16.npz).
  * Frozen SL reference = a second copy of the same init (the KL-leash target).
  * Opponents seats 1-3 = the frozen SL net (self-play vs SL == the eval field).
  * Reward = terminal MCR score(seat0) / 8   (fan>=8 enforced, see below).
  * PPO: masked-categorical clip loss, GAE (gamma=1, lam=0.95), value + entropy.
  * KL LEASH: penalty beta * KL(current || frozen_SL) with an adaptive beta and a
    TRUST-REGION rollback -- after each update we measure the true on-distribution
    KL over the rollout; if it breached --kl-target we REJECT the policy move and
    ramp beta (keeps KL ~0.02-0.04, < 0.05 ceiling), exactly like the torch run.

DESIGN NOTES / deliberate choices (documented, honest):
  1. HU GATING.  ``jax_env.legal_mask`` intentionally omits Hu (a win needs the
     fan-calc, which the random rollout policy never declares).  A real trainer
     needs Hu as the reward source, so we compute an ON-DEVICE, vmappable
     winning-hand check (standard 4-melds+pair via a reachability DP over the 34
     tiles, plus seven-pairs) and OR a Hu bit into the mask for every seat that
     can win.  This keeps the whole rollout on the GPU.  Rarer special MCR shapes
     (thirteen orphans / knitted straight / honors-knitted, <1% of wins) are NOT
     gated on-device; the SL net almost never reaches them.
  2. FAN>=8.  The on-device check is a structural superset (fan-agnostic).  The
     terminal reward is read straight from the GROUND-TRUTH ``jax_env.host_score``
     (called ONCE per finished game).  As of the issue-#4 fix, host_score already
     FORFEITS any Hu with fan < 8 (MCR's legal minimum): the declarer gets -24 and
     the other three +8 (zero-sum).  We use ``reward_seat0 = scores[0] / 8`` verbatim,
     so a structural-but-illegal Hu is PENALISED (-3.0 for the declarer), never paid
     -- the intended "illegal action penalised, not silently rewarded" RL setup, and
     it closes the fan=-1 -> 3*(8+fan) reward-hack.
  3. CRITIC.  We use a FRESH lightweight flax conv critic, trained from scratch by
     GAE.  In v2 it predicts a POPART-NORMALIZED return (see V2 CHANGES A above): the
     regression target is O(1) instead of the raw score0/8 (which spans -3..+6 and has
     huge variance from single high-fan Hu terminals), so the fresh net can actually
     converge.  The KL trust-region rollback still protects stability regardless of
     early critic noise.
  4. Peng/Chi FOLLOW-DISCARD.  When seat 0 is granted a Peng/Chi, the follow-up
     discard is the same held-non-live heuristic the fused rollout uses (not a 2nd
     policy head).  The dominant seat-0 decision (which tile to discard on its own
     draw) IS a full policy decision; claims are a small minority of decisions.

Usage:
  python -m baselines.mahjong_t2_jax_v2 --selftest        # validate the Hu checker
  python -m baselines.mahjong_t2_jax_v2 --parity 300      # flax policy vs numpy champion
  python -m baselines.mahjong_t2_jax_v2 --seed 1 --updates 100000 --resume \
      --popart 1 --value-epochs 6 --out /root/ludus_train/mahjong_t2_jax_v2/
"""
import argparse, json, os, sys, time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, "/root/ludus/baselines")
sys.path.insert(0, "/root/mcr_champion")
os.environ.setdefault("MCR_CHAMPION_DIR", "/root/mcr_champion")

from mahjong import jax_env as J
from mahjong.jax_env import (A_HU, A_PASS, A_PLAY, A_CHI, A_PENG, A_GANG,
                             A_ANGANG, A_BUGANG, P_DRAW, P_DISCARD, P_BUGANG, N_ACT)
from mahjong_jax_policy import load_params, _conv

NPZ = "/root/mcr_champion/kdens_s0_fp16.npz"

# ---- globals set in main() (closed over by the jitted fns; constant => no recompile)
FROZEN = None            # frozen SL params dict (KL ref + opponents)
BLOCKS = 40
DT = jnp.float32         # policy dtype (fp32 => exact champion parity)
OPT_PI = None
OPT_V = None


# ============================================================ policy forward
def pforward(params, obs, mask, dtype=None):
    """Masked kdens3 logits for a trainable params dict (grad flows to params).
    Identical maths to mahjong_jax_policy.forward."""
    dtype = DT if dtype is None else dtype
    cast = lambda t: jax.tree_util.tree_map(lambda a: a.astype(dtype), t)
    p = cast(params)
    x = obs.astype(dtype)
    x = jax.nn.relu(_conv(x, p["stem_w"], p["stem_b"]))
    for i in range(BLOCKS):
        y = jax.nn.relu(_conv(x, p[f"body_{i}_c1_w"], p[f"body_{i}_c1_b"]))
        y = _conv(y, p[f"body_{i}_c2_w"], p[f"body_{i}_c2_b"])
        x = jax.nn.relu(x + y)
    f = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(f @ p["foot1_w"].T + p["foot1_b"])
    out = (h @ p["foot3_w"].T + p["foot3_b"]).astype(jnp.float32)
    return jnp.where(mask > 0, out, -1e9)


# ============================================================ fresh flax critic
def v_init(key):
    ks = jax.random.split(key, 5)
    he = lambda k, shp, fin, sc=1.0: jax.random.normal(k, shp, jnp.float32) * (sc * np.sqrt(2.0 / fin))
    return {
        "c1": he(ks[0], (64, 38, 3, 3), 38 * 9), "c1b": jnp.zeros(64),
        "c2": he(ks[1], (64, 64, 3, 3), 64 * 9), "c2b": jnp.zeros(64),
        "c3": he(ks[2], (64, 64, 3, 3), 64 * 9), "c3b": jnp.zeros(64),
        "fw": he(ks[3], (256, 64 * 4 * 9), 64 * 4 * 9), "fb": jnp.zeros(256),
        "ow": he(ks[4], (1, 256), 256, sc=0.01), "ob": jnp.zeros(1),
    }


def v_forward(vp, obs):
    x = jax.nn.relu(_conv(obs.astype(jnp.float32), vp["c1"], vp["c1b"]))
    x = jax.nn.relu(_conv(x, vp["c2"], vp["c2b"]))
    x = jax.nn.relu(_conv(x, vp["c3"], vp["c3b"]))
    f = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(f @ vp["fw"].T + vp["fb"])
    return (h @ vp["ow"].T + vp["ob"])[:, 0]


# ============================================ on-device winning-hand check
_R = 5   # max carried partial-sequences per state slot (counts <= 4)


def _transition(reach, c, allow_seq):
    """Advance the standard-form reachability set by one tile (VECTORISED).
    reach: (2,_R,_R) bool over (pair_used, r1, r2):
       r1 = #seqs that used (i-2,i-1) and still need tile i,
       r2 = #seqs that used (i-1)     and still need tiles i,i+1.
    c: int32 count of tile i.  allow_seq: may new sequences start here (suit)?
    A transition maps source (p,r1,r2) -> target (p, new_r1=r2, new_r2=k), OR-ing
    over r1, for each number k of new sequences started here."""
    r1 = jnp.arange(_R)[:, None]                  # (R,1)
    r2 = jnp.arange(_R)[None, :]                   # (1,R)
    need = r1 + r2                                 # (R,R)
    rem = c - need                                 # (R,R)
    base = reach & (c >= need)                     # (2,R,R)
    allow = jnp.asarray(allow_seq)
    nxt = jnp.zeros((2, _R, _R), bool)
    for k in range(_R):                            # k new seqs starting here (<=4)
        left = rem - k
        okk = base & (k <= rem) & (left >= 0) & (allow | (k == 0))       # (2,R,R)
        lmod = jnp.mod(jnp.maximum(left, 0), 3)                          # (R,R)
        c0 = (okk & (lmod == 0)).any(axis=1)       # (2,R) over source r1 -> (p, new_r1=r2)
        nxt = nxt.at[:, :, k].set(nxt[:, :, k] | c0)
        c2 = (okk[0] & (lmod == 2)).any(axis=0)    # pair from p=0 only, over r1 -> (new_r1=r2,)
        nxt = nxt.at[1, :, k].set(nxt[1, :, k] | c2)
    return nxt


def _standard_win(counts, n_sets):
    """counts int[34] decomposes into n_sets melds + exactly one pair?"""
    reach = jnp.zeros((2, _R, _R), bool).at[0, 0, 0].set(True)
    for i in range(34):
        if i in (9, 18, 27):                      # suit / honour boundary: carry must be 0
            keep = jnp.zeros((2, _R, _R), bool).at[:, 0, 0].set(reach[:, 0, 0])
            reach = keep
        reach = _transition(reach, counts[i], allow_seq=(i < 27))
    total_ok = counts.sum() == (3 * n_sets + 2)
    return reach[1, 0, 0] & total_ok


def _seven_pairs(counts, meld_cnt):
    even = jnp.all((counts % 2) == 0)
    npair = (counts // 2).sum()
    return (meld_cnt == 0) & even & (npair == 7) & (counts.sum() == 14)


def can_win(counts, meld_cnt):
    """Vmappable structural MCR win check (standard 4-melds+pair OR seven pairs)."""
    n_sets = 4 - meld_cnt
    return _standard_win(counts, n_sets) | _seven_pairs(counts, meld_cnt)


# ---------------------------------------------------------- Hu-augmented mask
def hu_legal(state):
    """(4,) bool: is Hu structurally legal for each seat at the current phase?"""
    seats = jnp.arange(4)
    s = state.seat_to_act
    # zimo winning tile is last_draw_tile; host_score removes it, so it MUST be in
    # hand. Gate on that -- rules out the rare case where the concealed hand is
    # already a complete shape but last_draw_tile isn't part of it (an invalid zimo
    # the oracle scorer can't parse). Never false for a legitimate self-draw win.
    ldt = jnp.clip(state.last_draw_tile, 0, 33)
    zimo = can_win(state.hands[s], state.meld_cnt[s]) & (state.hands[s][ldt] > 0)
    zv = jnp.zeros((4,), bool).at[s].set(zimo)

    live = jnp.clip(state.last_discard_tile, 0, 33)
    disc = state.last_discard_seat
    def ron_seat(seat):
        h = state.hands[seat].at[live].add(1)
        return (seat != disc) & can_win(h, state.meld_cnt[seat])
    ron = jax.vmap(ron_seat)(seats)

    bt = jnp.clip(state.last_ev_t, 0, 33)
    turn = state.seat_to_act
    def rob_seat(seat):
        h = state.hands[seat].at[bt].add(1)
        return (seat != turn) & can_win(h, state.meld_cnt[seat])
    rob = jax.vmap(rob_seat)(seats)

    return jnp.where(state.phase == P_DRAW, zv,
           jnp.where(state.phase == P_DISCARD, ron,
           jnp.where(state.phase == P_BUGANG, rob, jnp.zeros((4,), bool))))


_MELD_BLOCK = (jnp.zeros(N_ACT, bool)
               .at[A_CHI:A_PENG].set(True).at[A_PENG:A_GANG].set(True)
               .at[A_GANG:A_ANGANG].set(True).at[A_ANGANG:A_BUGANG].set(True))


def full_mask(state):
    """base legal mask (4,235): add the Hu bit where structurally legal, and FORBID
    an illegal 5th meld. jax_env.legal_mask does not cap meld_cnt at 4 (it can offer
    a Chi/Peng/Gang/AnGang to a seat that already has 4 melds -- taking it drives
    meld_cnt to 5 and silently corrupts state via an out-of-bounds _add_meld). We
    mask the NEW-meld actions when meld_cnt>=4 (Play/Pass/Hu/BuGang stay legal;
    BuGang only upgrades a Peng in place, no new slot)."""
    m = J.legal_mask(state)
    hu = hu_legal(state)
    m = m.at[:, A_HU].set(m[:, A_HU] | hu)
    full4 = state.meld_cnt >= 4                                  # (4,)
    return jnp.where(full4[:, None] & _MELD_BLOCK[None, :], False, m)


def seat0_is_decider(state):
    ph = state.phase
    return (((ph == P_DRAW) & (state.seat_to_act == 0)) |
            ((ph == P_DISCARD) & (state.last_discard_seat != 0)) |
            ((ph == P_BUGANG) & (state.seat_to_act != 0)))


def _seat_obs(state):
    return jax.vmap(lambda s: J.obs(state, s))(jnp.arange(4))


def _aux_follow(state):
    live = state.last_discard_tile
    hh = jnp.where(jnp.arange(34)[None, :] == live, 0, state.hands)
    return jnp.argmax(hh, axis=1).astype(jnp.int32)


# ==================================================================== rollout
def _act_batch(pparams, opp3, states, keyL, keyO, greedy):
    """One fused decision tick over B games. Returns actions(B,4,2) plus the
    seat-0 (obs,mask,act,logp,val_slot,rec).

    opp3 is a STACKED pytree of the 3 opponent policies (seats 1/2/3); every leaf has
    a leading axis of 3. The opponent forward is a vmap-over-seat of pforward, then
    reshaped GAME-MAJOR (b0s1,b0s2,b0s3,b1s1,...) so the gumbel draw over (B*3,N_ACT)
    matches v2's opponent RNG stream exactly (pool={anchor} => bit-identical to v2)."""
    obs = jax.vmap(_seat_obs)(states)                     # (B,4,38,4,9)
    mask = jax.vmap(full_mask)(states)                    # (B,4,235)
    aux = jax.vmap(_aux_follow)(states)                   # (B,4)
    B = mask.shape[0]
    obs0 = obs[:, 0]
    mask0 = mask[:, 0]
    logits0 = pforward(pparams, obs0, mask0)
    if greedy:
        act0 = jnp.argmax(logits0, axis=1).astype(jnp.int32)
    else:
        g0 = jax.random.gumbel(keyL, logits0.shape)
        act0 = jnp.argmax(logits0 + g0, axis=1).astype(jnp.int32)
    lsm0 = jax.nn.log_softmax(logits0, axis=1)
    logp0 = jnp.take_along_axis(lsm0, act0[:, None], axis=1)[:, 0]
    # opponents (seats 1-3): each seat uses its own sampled pool policy (opp3, leading
    # axis 3). vmap pforward over the seat axis -> (3,B,N_ACT); transpose+reshape to
    # game-major (B*3,N_ACT) so gumbel/argmax is identical to v2's single (B*3) forward.
    obsO_s = jnp.transpose(obs[:, 1:], (1, 0, 2, 3, 4))    # (3,B,38,4,9)
    maskO_s = jnp.transpose(mask[:, 1:], (1, 0, 2))        # (3,B,N_ACT)
    logitsO_s = jax.vmap(pforward, in_axes=(0, 0, 0))(opp3, obsO_s, maskO_s)  # (3,B,N_ACT)
    logitsO = jnp.transpose(logitsO_s, (1, 0, 2)).reshape(B * 3, N_ACT)       # game-major
    if greedy:
        actO = jnp.argmax(logitsO, axis=1)
    else:
        gO = jax.random.gumbel(keyO, logitsO.shape)
        actO = jnp.argmax(logitsO + gO, axis=1)
    actO = actO.astype(jnp.int32).reshape(B, 3)
    primary = jnp.concatenate([act0[:, None], actO], axis=1)      # (B,4)
    actions = jnp.stack([primary, aux], axis=2)                   # (B,4,2)
    rec = jax.vmap(seat0_is_decider)(states) & (mask0.sum(1) > 1) & (states.done == 0)
    return actions, obs0, mask0, act0, logp0, rec


def make_train_rollout(T):
    def step_fn(carry, _):
        states, key = carry
        key, kL, kO = jax.random.split(key, 3)
        actions, obs0, mask0, act0, logp0, rec = _act_batch(
            carry_p[0], carry_p[1], states, kL, kO, greedy=False)
        nstates = jax.vmap(J.step)(states, actions)
        return (nstates, key), (obs0, mask0, act0, logp0, rec)

    # carry_p[0]=learner params, carry_p[1]=opp3 (stacked 3 opponent policies)
    carry_p = [None, None]

    @jax.jit
    def rollout(pparams, opp3, walls, quans, key):
        carry_p[0] = pparams
        carry_p[1] = opp3
        states = jax.vmap(J.reset)(walls, quans)
        (final, _), ys = lax.scan(step_fn, (states, key), None, length=T)
        return final, ys
    return rollout


def make_eval_rollout(T):
    # EVAL always plays seat-0 vs seats 1-3 = the FROZEN kdens3 anchor (the fixed
    # strength yardstick). Caller passes opp3 = FROZEN stacked x3; opponents never come
    # from the pool here, so win_rate/mean_score/dealin stay comparable to v2.
    carry_p = [None, None]

    def step_fn(carry, _):
        states, key, ks, kc = carry
        actions, obs0, mask0, act0, logp0, rec = _act_batch(
            carry_p[0], carry_p[1], states, key, key, greedy=True)
        # eval KL(current||frozen) on seat-0 decider steps
        logits = pforward(carry_p[0], obs0, mask0)
        flog = pforward(FROZEN, obs0, mask0)
        p = jax.nn.softmax(logits, 1)
        kl = (p * (jax.nn.log_softmax(logits, 1) - jax.nn.log_softmax(flog, 1))).sum(1)
        w = rec.astype(jnp.float32)
        nstates = jax.vmap(J.step)(states, actions)
        return (nstates, key, ks + (kl * w).sum(), kc + w.sum()), None

    @jax.jit
    def rollout(pparams, opp3, walls, quans, key):
        carry_p[0] = pparams
        carry_p[1] = opp3
        states = jax.vmap(J.reset)(walls, quans)
        (final, _, ks, kc), _ = lax.scan(
            step_fn, (states, key, 0.0, 0.0), None, length=T)
        return final, ks, kc
    return rollout


# ==================================================================== opponent pool
class PoolManager:
    """Host-side opponent pool for league self-play. Slot 0 is ALWAYS the kdens3 anchor
    (never evicted). Learner snapshots fill slots 1..pool_cap as a ring buffer (oldest
    non-anchor snapshot evicted first). Params are held on host as numpy pytrees; the 3
    sampled opponents are stacked and moved to device per rollout (~150 MB).

    Sampling (per opponent seat, independent): with prob p_anchor return the anchor; else
    return a snapshot drawn with recency-linear weights (weight = rank i+1 for the i-th
    oldest snapshot, so the newest is most likely). Falls back to the anchor when no
    snapshot exists yet (pool_cap==0 => pool is anchor-only => v2 behavior)."""

    def __init__(self, anchor_params, pool_cap, p_anchor, rng):
        # store as numpy for cheap host residency; anchor kept separately + as slot 0
        self.anchor = jax.tree_util.tree_map(lambda x: np.asarray(x), anchor_params)
        self.pool_cap = int(pool_cap)
        self.p_anchor = float(p_anchor)
        self.rng = rng
        self.snaps = []          # list of numpy param pytrees, oldest first
        self.n_added = 0         # total snapshots ever added (for logging)

    def add(self, params):
        if self.pool_cap <= 0:
            return
        snap = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), params)
        self.snaps.append(snap)
        if len(self.snaps) > self.pool_cap:
            self.snaps.pop(0)    # evict oldest non-anchor snapshot
        self.n_added += 1

    def size(self):
        return 1 + len(self.snaps)          # anchor + snapshots

    def _sample_one(self):
        """Return (params, tag) where tag is 'anchor' or a snapshot index (0=oldest)."""
        if not self.snaps or self.rng.random() < self.p_anchor:
            return self.anchor, "A"
        n = len(self.snaps)
        w = np.arange(1, n + 1, dtype=np.float64)   # recency-linear (newest heaviest)
        w /= w.sum()
        i = int(self.rng.choice(n, p=w))
        return self.snaps[i], "S%d" % i

    def sample_opp3(self):
        """Sample 3 opponent policies -> (stacked device pytree, [tag,tag,tag])."""
        chosen = [self._sample_one() for _ in range(3)]
        tags = [c[1] for c in chosen]
        p1, p2, p3 = (c[0] for c in chosen)
        opp3 = jax.tree_util.tree_map(
            lambda a, b, c: jnp.stack([jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)]),
            p1, p2, p3)
        return opp3, tags


# ==================================================================== GAE
@partial(jax.jit, static_argnames=("lam",))
def compute_gae(val, rec, R, lam):
    """Backward GAE over seat-0 DECISIONS (gamma=1, terminal-only reward R).
    val,rec: (T,B); R: (B,). Returns adv,ret (T,B) (meaningful where rec)."""
    T, B = val.shape
    def body(carry, t):
        lastgae, nextval, seen = carry
        v = val[t]
        r = rec[t]
        delta = jnp.where(seen, nextval - v, R - v)               # gamma=1
        newg = jnp.where(seen, delta + lam * lastgae, delta)
        lastgae2 = jnp.where(r, newg, lastgae)
        nextval2 = jnp.where(r, v, nextval)
        seen2 = jnp.where(r, jnp.ones_like(seen), seen)
        adv_t = jnp.where(r, newg, 0.0)
        return (lastgae2, nextval2, seen2), adv_t
    init = (jnp.zeros(B), jnp.zeros(B), jnp.zeros(B, bool))
    _, adv = lax.scan(body, init, jnp.arange(T)[::-1])
    adv = adv[::-1]
    ret = adv + val
    return adv, ret


@partial(jax.jit, static_argnames=("K",))
def compact(obs, mask, act, logp, adv, ret, rec, K):
    """Gather up to K recorded seat-0 decisions into a fixed buffer + valid mask."""
    fr = rec.reshape(-1)
    order = jnp.argsort(~fr)                                       # valid (True) first
    sel = order[:K]
    valid = fr[sel]
    g = lambda a: a.reshape((-1,) + a.shape[2:])[sel]
    return (g(obs), g(mask), g(act), g(logp),
            adv.reshape(-1)[sel], ret.reshape(-1)[sel], valid)


# ==================================================================== PPO
def _buffer_kl(pparams, obs, mask, valid, mb):
    """Mean KL(current||frozen) over the recorded buffer. lax.scan over chunks so
    the resnet forward is compiled ONCE (not unrolled per chunk)."""
    N = obs.shape[0]
    nch = N // mb
    O = obs[:nch * mb].reshape(nch, mb, 38, 4, 9)
    Mk = mask[:nch * mb].reshape(nch, mb, N_ACT)
    W = valid[:nch * mb].reshape(nch, mb).astype(jnp.float32)

    def body(carry, x):
        o, mk, w = x
        m = pforward(pparams, o, mk)
        fl = pforward(FROZEN, o, mk)
        p = jax.nn.softmax(m, 1)
        kl = (p * (jax.nn.log_softmax(m, 1) - jax.nn.log_softmax(fl, 1))).sum(1)
        return carry, ((kl * w).sum(), w.sum())
    _, (ks, kc) = lax.scan(body, None, (O, Mk, W))
    return ks.sum() / jnp.maximum(kc.sum(), 1.0)


@partial(jax.jit, static_argnames=("epochs", "nmb", "mb", "clip", "vcoef",
                                    "entcoef", "gate", "maxgn"))
def ppo_update(pparams, vparams, popt, vopt, obs, mask, act, oldlp, adv, ret,
               ret_mean, ret_std, valid, beta, key, epochs, nmb, mb, clip, vcoef,
               entcoef, gate, maxgn):
    w = valid.astype(jnp.float32)
    sw = jnp.maximum(w.sum(), 1.0)
    mean = (adv * w).sum() / sw
    var = ((adv - mean) ** 2 * w).sum() / sw
    adv = (adv - mean) / jnp.sqrt(var + 1e-8)
    N = obs.shape[0]

    def loss_fn(pp, vp, idx):
        O = obs[idx]; Mk = mask[idx]; A = adv[idx]; wv = w[idx]
        ac = act[idx]; olp = oldlp[idx]; rt = ret[idx]
        logits = pforward(pp, O, Mk)
        lsm = jax.nn.log_softmax(logits, 1)
        newlp = jnp.take_along_axis(lsm, ac[:, None], axis=1)[:, 0]
        ratio = jnp.exp(newlp - olp)
        pg = -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip, 1 + clip) * A)
        p = jax.nn.softmax(logits, 1)
        ent = -(p * lsm).sum(1)
        vpred = v_forward(vp, O)                       # v_norm (normalized units)
        tgt = (rt - ret_mean) / (ret_std + 1e-8)       # PopArt-normalized return
        vloss = 0.5 * (vpred - tgt) ** 2               # normalized value loss
        flog = jax.lax.stop_gradient(pforward(FROZEN, O, Mk))
        kl = (p * (lsm - jax.nn.log_softmax(flog, 1))).sum(1)
        swv = jnp.maximum(wv.sum(), 1.0)
        mb_kl = (kl * wv).sum() / swv
        use_pg = mb_kl < gate
        full = vcoef * vloss + beta * kl + jnp.where(use_pg, pg - entcoef * ent, 0.0)
        return (full * wv).sum() / swv, (pg, vloss, ent, kl, wv)

    gfn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    def one(carry, idx):
        pp, vp, po, vo = carry
        (loss, (pg, vl, ent, kl, wv)), (gp, gv) = gfn(pp, vp, idx)
        upd_p, po = OPT_PI.update(gp, po, pp); pp = optax.apply_updates(pp, upd_p)
        upd_v, vo = OPT_V.update(gv, vo, vp); vp = optax.apply_updates(vp, upd_v)
        swv = jnp.maximum(wv.sum(), 1.0)
        stat = jnp.array([(pg * wv).sum() / swv, (vl * wv).sum() / swv,
                          (ent * wv).sum() / swv])
        return (pp, vp, po, vo), stat

    # Build all epoch permutations up front (cheap, no resnet), then lax.scan the
    # grad steps so the 40-block resnet fwd+bwd is compiled ONCE (not unrolled
    # epochs*nmb times -- that unroll was the compile-time blow-up).
    perms = []
    for e in range(epochs):
        key, ke = jax.random.split(key)
        perms.append(jax.random.permutation(ke, N)[:nmb * mb].reshape(nmb, mb))
    idx_all = jnp.concatenate(perms, axis=0)                # (epochs*nmb, mb)
    carry = (pparams, vparams, popt, vopt)
    carry, stats = lax.scan(one, carry, idx_all)
    pparams, vparams, popt, vopt = carry
    st = jnp.mean(stats, axis=0)
    post_kl = _buffer_kl(pparams, obs, mask, valid, mb)
    return pparams, vparams, popt, vopt, dict(
        kl=post_kl, pg=st[0], vloss=st[1] * (ret_std ** 2), ent=st[2])


# ==================================================== value-only extra epochs (v2)
@partial(jax.jit, static_argnames=("epochs", "nmb", "mb"))
def value_update(vparams, vopt, obs, ret, ret_mean, ret_std, valid, key,
                 epochs, nmb, mb):
    """Extra VALUE-ONLY gradient steps (policy frozen). Minimises the same
    PopArt-normalized value loss the shared PPO epochs use, giving the fresh critic
    more fitting capacity per unit of rollout. No-op when epochs==0."""
    w = valid.astype(jnp.float32)
    tgt = (ret - ret_mean) / (ret_std + 1e-8)
    N = obs.shape[0]

    def loss_fn(vp, idx):
        O = obs[idx]; tg = tgt[idx]; wv = w[idx]
        vpred = v_forward(vp, O)
        vl = 0.5 * (vpred - tg) ** 2
        swv = jnp.maximum(wv.sum(), 1.0)
        return (vl * wv).sum() / swv

    gfn = jax.value_and_grad(loss_fn)

    def one(carry, idx):
        vp, vo = carry
        _, gv = gfn(vp, idx)
        upd_v, vo = OPT_V.update(gv, vo, vp)
        vp = optax.apply_updates(vp, upd_v)
        return (vp, vo), 0.0

    perms = []
    for e in range(epochs):
        key, ke = jax.random.split(key)
        perms.append(jax.random.permutation(ke, N)[:nmb * mb].reshape(nmb, mb))
    idx_all = jnp.concatenate(perms, axis=0) if epochs > 0 else jnp.zeros((0, mb), jnp.int32)
    (vparams, vopt), _ = lax.scan(one, (vparams, vopt), idx_all)
    return vparams, vopt


@partial(jax.jit, static_argnames=("nmb", "mb"))
def eval_vloss(vparams, obs, ret, ret_mean, ret_std, valid, nmb, mb):
    """RAW-units value loss of the FINAL critic over the buffer (for logging /
    comparability with the baseline). raw = 0.5*(v_norm*std+mean - rt)^2."""
    w = valid.astype(jnp.float32)
    O = obs[:nmb * mb].reshape(nmb, mb, 38, 4, 9)
    Rt = ret[:nmb * mb].reshape(nmb, mb)
    W = w[:nmb * mb].reshape(nmb, mb)

    def body(carry, x):
        o, rt, wv = x
        vraw = v_forward(vparams, o) * ret_std + ret_mean
        vl = 0.5 * (vraw - rt) ** 2
        return carry, ((vl * wv).sum(), wv.sum())
    _, (ls, cs) = lax.scan(body, None, (O, Rt, W))
    return ls.sum() / jnp.maximum(cs.sum(), 1.0)


# ============================================================ terminal reward
def terminal_stats(final):
    """Host: per-game seat-0 reward (score0/8, fan>=8 enforced) + eval flags."""
    fs = jax.device_get(final)
    B = int(fs.done.shape[0])
    R = np.zeros(B, np.float32)
    zimo = np.zeros(B, bool); dealin = np.zeros(B, bool); win = np.zeros(B, bool)
    for i in range(B):
        st_i = jax.tree_util.tree_map(lambda a: a[i], fs)
        ending = int(st_i.ending)
        if ending in (1, 2):
            w = int(st_i.winner)
            try:
                fan, scores = J.host_score(st_i)   # forfeits fan<8 (declarer -24)
            except Exception as e:
                # A terminal the oracle scorer can't parse (rare malformed Hu the
                # structural gate let through): FORFEIT the declarer -- never a payout.
                fan, scores = -1, [(-24 if s == w else 8) for s in range(4)]
                d = getattr(terminal_stats, "_dbg", 0)
                if d < 3:
                    terminal_stats._dbg = d + 1
                    hw = np.asarray(st_i.hands)[w]
                    print("[score-fail]", type(e).__name__, str(e)[:50], "ending", ending,
                          "is_self", int(st_i.is_self), "winner", w, "win_tile",
                          int(st_i.win_tile), "mc", int(np.asarray(st_i.meld_cnt)[w]),
                          "handsum", int(hw.sum()), flush=True)
            R[i] = scores[0] / 8.0                  # penalise illegal Hu, never pay it
            legal = fan >= 8                        # only legal wins count in eval stats
            win[i] = legal and (w == 0)
            zimo[i] = legal and (ending == 1 and w == 0)
            dealin[i] = legal and (ending == 2 and w != 0 and int(st_i.discarder) == 0)
    return R, zimo, dealin, win


# ==================================================================== trainer
class Trainer:
    def __init__(self, a):
        global FROZEN, BLOCKS, DT, OPT_PI, OPT_V
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        DT = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}[a.dtype]
        bundle = load_params(NPZ)
        BLOCKS = bundle["blocks"]
        self.pparams = jax.tree_util.tree_map(lambda x: jnp.asarray(x), bundle["params"])
        FROZEN = jax.tree_util.tree_map(lambda x: jnp.asarray(x), bundle["params"])
        self.vparams = v_init(jax.random.PRNGKey(a.seed))
        OPT_PI = optax.chain(optax.clip_by_global_norm(a.max_grad_norm), optax.adam(a.lr_pi))
        OPT_V = optax.chain(optax.clip_by_global_norm(a.max_grad_norm), optax.adam(a.lr_v))
        self.popt = OPT_PI.init(self.pparams)
        self.vopt = OPT_V.init(self.vparams)
        self.beta = a.beta0
        # PopArt-lite EMA return stats (carried in train state; RAW reward units).
        # popart off => pinned to (0,1) so the critic sees raw targets == baseline.
        self.ret_mean = 0.0
        self.ret_std = 1.0
        self.update = 0
        self.env_steps = 0
        self.key = jax.random.PRNGKey(1000 + a.seed)
        self.gctr = 0
        self.t0 = time.time()
        self.last_ckpt = time.time()
        self.train_roll = make_train_rollout(a.T)
        self.eval_roll = make_eval_rollout(a.eval_T)
        self.nmb = a.K // a.minibatch
        # --- LEAGUE: opponent pool + the fixed FROZEN-anchor eval field (stacked x3)
        self.pool = PoolManager(FROZEN, a.pool_cap, a.p_anchor,
                                np.random.default_rng(4242 + a.seed))
        self.FROZEN3 = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (3,) + x.shape), FROZEN)
        self.last_tags = ["A", "A", "A"]

    # --- LEAGUE: linear tight->loose KL-target schedule (constant if not configured)
    def current_kl_target(self):
        a = self.a
        if a.kl_anneal_updates <= 0 or a.kl_target_final == a.kl_target:
            return a.kl_target
        frac = min(1.0, self.update / float(a.kl_anneal_updates))
        return a.kl_target + frac * (a.kl_target_final - a.kl_target)

    # -- deterministic random walls for a batch (seeded by seed + counter)
    def _walls(self, B):
        rng = np.random.default_rng(1_000_000 * self.a.seed + self.gctr)
        self.gctr += 1
        base = np.repeat(np.arange(34, dtype=np.int8), 4)
        W = np.stack([rng.permutation(base) for _ in range(B)])
        Q = np.full(B, self.a.quan, np.int32)
        return jnp.asarray(W), jnp.asarray(Q)

    def collect_and_update(self):
        a = self.a
        self.key, kr, ku = jax.random.split(self.key, 3)
        W, Q = self._walls(a.B)
        # LEAGUE: sample 3 opponent policies from the pool for this rollout
        opp3, self.last_tags = self.pool.sample_opp3()
        klt = self.current_kl_target()
        final, (obs0, mask0, act0, logp0, rec) = self.train_roll(self.pparams, opp3, W, Q, kr)
        R, _, _, _ = terminal_stats(final)
        Rj = jnp.asarray(R)
        # critic predicts a NORMALIZED value; DENORMALIZE with current EMA stats so
        # ALL GAE/advantage math below stays in RAW reward units.
        rm0, rs0 = jnp.float32(self.ret_mean), jnp.float32(self.ret_std)
        val0n = jax.vmap(lambda o: v_forward(self.vparams, o))(obs0)    # (T,B) v_norm
        val0 = val0n * rs0 + rm0                                        # (T,B) v_raw
        adv, ret = compute_gae(val0, rec, Rj, a.lam)                    # RAW units
        obsB, maskB, actB, logpB, advB, retB, valid = compact(
            obs0, mask0, act0, logp0, adv, ret, rec, a.K)
        nvalid = int(jax.device_get(valid.sum()))
        # --- PopArt-lite: EMA-update ret_mean/ret_std from THIS update's raw return
        # targets (valid decisions only) BEFORE fitting the critic to them.
        if a.popart:
            retnp = np.asarray(jax.device_get(retB))
            valnp = np.asarray(jax.device_get(valid)).astype(bool)
            rv = retnp[valnp]
            if rv.size > 1:
                bm = float(rv.mean()); bs = float(rv.std())
                self.ret_mean = 0.99 * self.ret_mean + 0.01 * bm
                self.ret_std = 0.99 * self.ret_std + 0.01 * bs
        rm, rs = jnp.float32(self.ret_mean), jnp.float32(self.ret_std)
        old_p = self.pparams
        np_, nv_, po_, vo_, stats = ppo_update(
            self.pparams, self.vparams, self.popt, self.vopt,
            obsB, maskB, actB, logpB, advB, retB, rm, rs, valid,
            jnp.float32(self.beta), ku,
            a.epochs, self.nmb, a.minibatch, a.clip, a.vcoef, a.entcoef,
            a.kl_gate_frac * klt, a.max_grad_norm)
        # extra VALUE-ONLY epochs (policy frozen) on the accepted critic
        if a.value_epochs > 0:
            self.key, kv = jax.random.split(self.key)
            nv_, vo_ = value_update(nv_, vo_, obsB, retB, rm, rs, valid, kv,
                                    a.value_epochs, self.nmb, a.minibatch)
        post_kl = float(jax.device_get(stats["kl"]))
        rejected = post_kl > klt
        # RAW-units value loss of the FINAL critic (for comparability with baseline)
        vloss_raw = float(jax.device_get(
            eval_vloss(nv_, obsB, retB, rm, rs, valid, self.nmb, a.minibatch)))
        # trust region: on breach reject the policy move, keep critic, ramp beta
        self.vparams, self.vopt = nv_, vo_
        if rejected:
            self.pparams = old_p
            self.popt = po_            # keep Adam moments (matches torch)
            self.beta = min(self.beta * 2.0, 50.0)
        else:
            self.pparams, self.popt = np_, po_
            if post_kl > 0.5 * klt:
                self.beta = min(self.beta * 1.3, 20.0)
            elif post_kl < 0.3 * klt:
                self.beta = max(self.beta / 1.2, 1e-3)
        self.env_steps += a.B * a.T
        # LEAGUE: periodically snapshot the (post-update) learner into the pool
        if a.pool_cap > 0 and a.pool_refresh > 0 and \
                (self.update + 1) % a.pool_refresh == 0:
            self.pool.add(self.pparams)
        return dict(kl=post_kl, rejected=rejected, nvalid=nvalid,
                    pg=float(stats["pg"]), vloss=vloss_raw,
                    ret_std=float(self.ret_std), ret_mean=float(self.ret_mean),
                    ent=float(stats["ent"]), R_mean=float(R.mean()),
                    kl_target=klt, pool_size=self.pool.size(),
                    opp_tags=self.last_tags)

    def bench_rollout(self, reps=5):
        """Time the PURE fused rollout (env+policy on GPU, no PPO) -> game-steps/s.
        This is the env-throughput headline vs the CPU torch env's ~313/s."""
        a = self.a
        W, Q = self._walls(a.B)
        self.key, k = jax.random.split(self.key)
        opp3, _ = self.pool.sample_opp3()
        final, _ = self.train_roll(self.pparams, opp3, W, Q, k)   # compile + warm
        final.done.block_until_ready()
        t = time.time()
        for _ in range(reps):
            self.key, k = jax.random.split(self.key)
            final, _ = self.train_roll(self.pparams, opp3, W, Q, k)
            final.done.block_until_ready()
        dt = (time.time() - t) / reps
        sps = (a.B * a.T) / dt
        print(f"[bench] pure fused rollout: B={a.B} T={a.T} dt={dt:.3f}s "
              f"game-steps/s={sps:,.0f}  policy-fwd/s={sps*4:,.0f}  "
              f"done={int(final.done.sum())}/{a.B}", flush=True)
        return sps

    def evaluate(self):
        a = self.a
        n = a.eval_games
        scores = []; zi = []; di = []; wi = []
        ks = 0.0; kc = 0.0
        while len(scores) < n:
            self.key, ke = jax.random.split(self.key)
            W, Q = self._walls(a.B)
            # EVAL opponents = FROZEN kdens3 anchor (fixed yardstick), NOT the pool
            final, kss, kcc = self.eval_roll(self.pparams, self.FROZEN3, W, Q, ke)
            R, zimo, dealin, win = terminal_stats(final)
            scores.extend((R * 8.0).tolist())
            zi.extend(zimo.tolist()); di.extend(dealin.tolist()); wi.extend(win.tolist())
            ks += float(kss); kc += float(kcc)
        scores = scores[:n]; zi = zi[:n]; di = di[:n]; wi = wi[:n]
        rec = {"step": self.update, "step_kind": "ppo_updates", "env_steps": self.env_steps,
               "KL": round(ks / max(1.0, kc), 5), "mean_score": round(float(np.mean(scores)), 4),
               "zimo_rate": round(float(np.mean(zi)), 4), "dealin_rate": round(float(np.mean(di)), 4),
               "win_rate": round(float(np.mean(wi)), 4), "n_games": len(scores),
               "beta": round(self.beta, 5), "ret_std": round(self.ret_std, 4),
               "ret_mean": round(self.ret_mean, 4), "steps_s": round(self.steps_s, 1),
               "pool_size": self.pool.size(), "pool_added": self.pool.n_added,
               "kl_target": round(self.current_kl_target(), 5),
               "wall_s": round(time.time() - self.t0, 1)}
        with open(os.path.join(a.out, "seed%d_jax_results.jsonl" % a.seed), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def save_ckpt(self):
        import flax.serialization as fser
        a = self.a
        blob = {"pparams": self.pparams, "vparams": self.vparams,
                "popt": self.popt, "vopt": self.vopt,
                "meta": {"beta": self.beta, "update": self.update,
                         "env_steps": self.env_steps, "gctr": self.gctr,
                         "ret_mean": self.ret_mean, "ret_std": self.ret_std}}
        with open(os.path.join(a.out, "latest.msgpack"), "wb") as f:
            f.write(fser.to_bytes(blob))
        # LEAGUE: persist the opponent pool (variable-length -> separate pickle)
        import pickle
        with open(os.path.join(a.out, "pool.pkl"), "wb") as f:
            pickle.dump({"snaps": self.pool.snaps, "n_added": self.pool.n_added}, f)

    def try_resume(self):
        import flax.serialization as fser
        p = os.path.join(self.a.out, "latest.msgpack")
        if not os.path.exists(p):
            return False
        tmpl = {"pparams": self.pparams, "vparams": self.vparams,
                "popt": self.popt, "vopt": self.vopt,
                "meta": {"beta": 0.0, "update": 0, "env_steps": 0, "gctr": 0,
                         "ret_mean": 0.0, "ret_std": 1.0}}
        with open(p, "rb") as f:
            blob = fser.from_bytes(tmpl, f.read())
        self.pparams = blob["pparams"]; self.vparams = blob["vparams"]
        self.popt = blob["popt"]; self.vopt = blob["vopt"]
        m = blob["meta"]
        self.beta = float(m["beta"]); self.update = int(m["update"])
        self.env_steps = int(m["env_steps"]); self.gctr = int(m["gctr"])
        self.ret_mean = float(m.get("ret_mean", 0.0)); self.ret_std = float(m.get("ret_std", 1.0))
        # LEAGUE: restore the opponent pool if present (keep cap in sync)
        import pickle
        pp = os.path.join(self.a.out, "pool.pkl")
        if os.path.exists(pp):
            with open(pp, "rb") as f:
                pd = pickle.load(f)
            self.pool.snaps = pd["snaps"][-self.pool.pool_cap:] if self.pool.pool_cap > 0 else []
            self.pool.n_added = int(pd.get("n_added", len(self.pool.snaps)))
        print(f"[resume] update={self.update} env_steps={self.env_steps} beta={self.beta:.4f} "
              f"ret_mean={self.ret_mean:.3f} ret_std={self.ret_std:.3f} "
              f"pool_size={self.pool.size()}", flush=True)
        return True

    def run(self):
        a = self.a
        self.steps_s = 0.0
        if a.eval_at_start and self.update == 0:
            print("[eval@start]", json.dumps(self.evaluate()), flush=True)
        while self.update < a.updates:
            t = time.time()
            st = self.collect_and_update()
            self.update += 1
            dt = time.time() - t
            self.steps_s = (a.B * a.T) / dt
            if self.update % a.log_every == 0:
                print(f"[u{self.update}] kl={st['kl']:.4f}{'(REJ)' if st['rejected'] else ''} "
                      f"klt={st['kl_target']:.3f} beta={self.beta:.4f} pg={st['pg']:.4f} "
                      f"vloss={st['vloss']:.3f} ret_std={st['ret_std']:.3f} ent={st['ent']:.3f} "
                      f"R={st['R_mean']:.3f} nrec={st['nvalid']} "
                      f"pool={st['pool_size']} opp={','.join(st['opp_tags'])} "
                      f"steps/s={self.steps_s:,.0f} env_steps={self.env_steps}", flush=True)
            do_eval = a.eval_every_updates > 0 and self.update % a.eval_every_updates == 0
            do_ckpt = time.time() - self.last_ckpt >= a.ckpt_every_sec
            if do_ckpt:
                do_eval = True
            if do_eval:
                print("[eval]", json.dumps(self.evaluate()), flush=True)
            if do_ckpt:
                self.save_ckpt()
                self.last_ckpt = time.time()
                print(f"[ckpt] saved at update {self.update}", flush=True)
        self.save_ckpt()
        print("[eval-final]", json.dumps(self.evaluate()), flush=True)


# ==================================================================== selftest
def selftest():
    """Validate can_win against a plain-python recursive reference on random hands."""
    import random as _r
    def ref_std(counts, n_sets):
        counts = list(counts)
        def rec(c, need_sets, need_pair):
            if need_sets == 0 and need_pair == 0:
                return sum(c) == 0
            i = next((k for k in range(34) if c[k] > 0), None)
            if i is None:
                return False
            ok = False
            if need_pair and c[i] >= 2:
                c[i] -= 2
                ok = rec(c, need_sets, 0); c[i] += 2
                if ok: return True
            if need_sets > 0 and c[i] >= 3:
                c[i] -= 3
                ok = rec(c, need_sets - 1, need_pair); c[i] += 3
                if ok: return True
            if need_sets > 0 and i < 27 and (i % 9) <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
                c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
                ok = rec(c, need_sets - 1, need_pair); c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
                if ok: return True
            return False
        if sum(counts) != 3 * n_sets + 2:
            return False
        return rec(counts, n_sets, 1)

    cw = jax.jit(lambda c, mc: can_win(c, mc))
    rng = np.random.default_rng(0)
    mism = 0; nwins = 0; N = 4000
    for t in range(N):
        mc = int(rng.integers(0, 3))
        n_needed = 14 - 3 * mc
        # build a random 34-count hand of size n_needed, max 4 each
        counts = np.zeros(34, np.int32)
        left = n_needed
        while left > 0:
            k = int(rng.integers(0, 34))
            if counts[k] < 4:
                counts[k] += 1; left -= 1
        got = bool(jax.device_get(cw(jnp.asarray(counts), jnp.int32(mc))))
        exp = ref_std(counts, 4 - mc) or (mc == 0 and _seven_ref(counts))
        nwins += int(exp)
        if got != exp:
            mism += 1
            if mism <= 5:
                print("MISMATCH mc=%d got=%d exp=%d counts=%s" % (mc, got, exp, counts.tolist()))
    print(f"SELFTEST can_win vs reference: {mism}/{N} mismatches ({nwins} true-wins in sample)")
    # also force many CONSTRUCTED wins to ensure no false negatives
    fn = 0; NC = 2000
    for t in range(NC):
        mc = int(rng.integers(0, 3)); n_sets = 4 - mc
        counts = np.zeros(34, np.int32)
        ok = True
        # random pair
        pv = int(rng.integers(0, 34)); counts[pv] += 2
        for _s in range(n_sets):
            if rng.random() < 0.5:                         # triplet
                tv = int(rng.integers(0, 34))
                if counts[tv] + 3 > 4: ok = False; break
                counts[tv] += 3
            else:                                          # sequence
                suit = int(rng.integers(0, 3)); lo = int(rng.integers(0, 7))
                base = suit * 9 + lo
                if any(counts[base + d] + 1 > 4 for d in range(3)): ok = False; break
                for d in range(3): counts[base + d] += 1
        if not ok: continue
        got = bool(jax.device_get(cw(jnp.asarray(counts), jnp.int32(mc))))
        if not got:
            fn += 1
            if fn <= 5: print("FALSE-NEG mc=%d counts=%s" % (mc, counts.tolist()))
    print(f"SELFTEST constructed-wins false-negatives: {fn}/{NC}")


def _seven_ref(counts):
    return all(c % 2 == 0 for c in counts) and sum(c // 2 for c in counts) == 7 and sum(counts) == 14


# ==================================================================== parity
def run_parity(n):
    from numpy_resfused import NumpyResFused
    obs = np.load("/root/parity_obs.npy")[:n]
    mask = np.load("/root/parity_mask.npy")[:n]
    bundle = load_params(NPZ)
    global BLOCKS
    BLOCKS = bundle["blocks"]
    p = jax.tree_util.tree_map(lambda x: jnp.asarray(x), bundle["params"])
    logits = np.asarray(jax.jit(lambda o, m: pforward(p, o, m, DT))(
        jnp.asarray(obs), jnp.asarray(mask.astype(bool))))
    npm = NumpyResFused(NPZ)
    dis = 0
    for i in range(obs.shape[0]):
        fa = int(logits[i].argmax())
        na = int(npm.logits(obs[i], mask[i]).argmax())
        dis += (fa != na)
    print(f"PARITY flax-policy vs numpy-champion ({DT.__name__}): {dis}/{obs.shape[0]} argmax disagreements", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--updates", type=int, default=100000)
    ap.add_argument("--out", type=str, default="/root/ludus_train/mahjong_t2_jax_v2/")
    ap.add_argument("--B", type=int, default=512)              # parallel games
    ap.add_argument("--T", type=int, default=256)             # rollout scan length
    ap.add_argument("--eval-T", type=int, default=256)
    ap.add_argument("--K", type=int, default=20480)           # buffer capacity
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr-pi", type=float, default=3e-5)
    ap.add_argument("--lr-v", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--popart", type=int, default=1)          # v2: return normalization (PopArt-lite)
    ap.add_argument("--value-epochs", type=int, default=6)    # v2: extra value-only grad steps/update
    ap.add_argument("--entcoef", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--kl-target", type=float, default=0.05)
    ap.add_argument("--beta0", type=float, default=0.08)
    ap.add_argument("--kl-gate-frac", type=float, default=0.7)
    # --- LEAGUE: opponent pool
    ap.add_argument("--pool-cap", type=int, default=16,
                    help="max learner snapshots in the pool (0 => anchor-only == v2)")
    ap.add_argument("--pool-refresh", type=int, default=50,
                    help="snapshot the learner into the pool every N updates")
    ap.add_argument("--p-anchor", type=float, default=0.3,
                    help="per-seat prob of drawing the kdens3 anchor vs a snapshot")
    # --- LEAGUE: KL tight->loose anneal (defaults => constant --kl-target)
    ap.add_argument("--kl-target-final", type=float, default=0.05,
                    help="final KL target; set > --kl-target with --kl-anneal-updates to widen")
    ap.add_argument("--kl-anneal-updates", type=int, default=0,
                    help="linearly anneal kl-target -> kl-target-final over N updates (0=off)")
    ap.add_argument("--quan", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--eval-games", type=int, default=2000)
    ap.add_argument("--eval-every-updates", type=int, default=25)
    ap.add_argument("--eval-at-start", action="store_true")
    ap.add_argument("--ckpt-every-sec", type=float, default=1200)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    print("device:", jax.devices(), flush=True)
    global DT
    DT = {"fp32": jnp.float32, "bf16": jnp.bfloat16, "fp16": jnp.float16}[args.dtype]
    if args.selftest:
        selftest(); return
    if args.parity:
        run_parity(args.parity); return

    tr = Trainer(args)
    if args.bench:
        tr.bench_rollout(); return
    if args.resume:
        tr.try_resume()
    tr.run()


if __name__ == "__main__":
    main()
