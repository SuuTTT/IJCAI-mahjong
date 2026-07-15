# Mahjong RL environment (`MahjongEnv`)

A reusable, single-agent reinforcement-learning environment for **Chinese Standard
Mahjong (MCR / 国标麻将)**, built on the Ludus MCR engine. It exposes the *same*
38×4×9 observation and 235-action interface that the IJCAI kdens3 agent uses, so
a policy trained here speaks the standard MCR RL interface.

The game is driven by `mahjong.validate_adapter.MyEngine`, which reproduces the
official judge **byte-exact on all 12,288 games** of the reference test set
(`Dannibal/mcr-final2026-testset`). So every move the env accepts is legal and
every terminal fan/score is the ground-truth PyMahjongGB result — you are
training against a correct referee, not an approximation.

---

## Interface

```python
class MahjongEnv:
    obs_shape = (38, 4, 9)      # float32
    n_actions = 235

    def reset(self, opponents, seed=0) -> (obs, legal_mask):
        # opponents: a list of 3 Botzone-protocol objects (seats 1,2,3), each
        #   exposing  .respond(request:str) -> response:str
        # returns obs float32(38,4,9), legal_mask bool(235)

    def step(self, action:int) -> (obs, legal_mask, reward, done, info):
        # action must be legal (legal_mask[action] == True)
        # reward is 0.0 until the terminal step; info carries {ending,winner,fan,...}
```

The learner always plays **seat 0 (East)**. The three opponents can be anything
that responds to Botzone request strings — the shipped `mahjong.bots.EfficiencyBot`
/ `RandomLegalBot`, the `mahjong.champion.ChampionBot` (kdens3), or **frozen
copies of your own policy** for self-play (see `mahjong.rl_env.PolicyAgent`).

Non-decision requests (deal, other players' draws, your own echoes) are handled
internally — `step` only ever asks you for **real choices**: which tile to
discard, whether to Chi/Peng/Gang/Hu, and the follow-up discard after a Peng/Chi.

---

## Observation — `(38, 4, 9)` float32

Reshaped from `(38, 36)`; the 36 columns are the 34 tile codes
`W1–9, T1–9, B1–9, F1–4, J1–3` padded to 4×9. Planes:

| planes | content |
|---|---|
| 0 | seat wind (門風), one-hot at the seat-wind honour |
| 1 | prevalent wind (圈風), one-hot at the round-wind honour |
| 2–5 | own concealed hand, count-encoded (plane *k* set where you hold ≥*k* copies) |
| 6–21 | discards — 4 planes × 4 players (self first, then downstream) |
| 22–37 | melds (副露) — 4 planes × 4 players, tiles laid out flat |

The encoding is reused verbatim from the kdens3 champion's `feature.py:FeatureAgent`,
so observations are identical to the ones the champion was trained on.

## Action space — 235 discrete, with a boolean legality `mask`

| index | action |
|---|---|
| 0 | Pass |
| 1 | Hu (declare a win) |
| 2–35 | Play (discard) tile *t* = `2 + tile_index` |
| 36–98 | Chi = `36 + suit*21 + (num-2)*3 + offset` |
| 99–132 | Peng tile *t* |
| 133–166 | Gang (melded kong of a discard) |
| 167–200 | AnGang (concealed kong) |
| 201–234 | BuGang (added kong) |

**Always mask your policy logits with `legal_mask` before sampling** — the env
raises on an illegal action.

## Reward — terminal, sparse, zero-sum

The engine's final MCR score for seat 0, divided by 8. For an 8-fan hand:

| outcome | reward |
|---|---|
| self-draw win (zimo) | `3*(8+fan)/8` = **+6.0** |
| win on a discard (ron) | `(24+fan)/8` = **+4.0** |
| deal-in (your discard was won) | `-(8+fan)/8` = **−2.0** |
| passive loss (another player's ron) | **−1.0** |
| passive loss (another player's zimo) | `-(8+fan)/8` = **−2.0** |
| exhaustive draw (荒牌) | **0.0** |

Higher-fan hands scale linearly. An optional `reward_shaping` (default `0.0`)
subtracts a small per-step constant to encourage faster wins; leave it at `0.0`
for pure terminal reward. Note the ≥8-fan minimum makes wins **genuinely sparse**
from scratch — plan for long training or curriculum/opponent shaping.

---

## Minimal usage

```python
import numpy as np
from mahjong.rl_env import MahjongEnv
from mahjong.bots import EfficiencyBot

env = MahjongEnv()
obs, mask = env.reset(opponents=[EfficiencyBot(), EfficiencyBot(), EfficiencyBot()], seed=0)
done = False
while not done:
    legal = np.flatnonzero(mask)
    action = int(np.random.choice(legal))        # <- replace with your policy(obs, mask)
    obs, mask, reward, done, info = env.step(action)
print(info)      # {'ending': 'hu'|'draw', 'winner': seat, 'fan': int, ...}, reward on the last step
```

### Self-play

```python
from mahjong.rl_env import MahjongEnv, PolicyAgent
# frozen snapshots of your net become the 3 opponents:
opps = [PolicyAgent(params_snapshot), PolicyAgent(params_snapshot), PolicyAgent(params_snapshot)]
obs, mask = env.reset(opponents=opps, seed=k)
```

---

## Reference trainer

`baselines/mahjong_ppo.py` is a working **JAX/flax masked actor-critic PPO** with
multiprocess CPU rollouts and self-play (mixing frozen policy snapshots with the
scripted `EfficiencyBot`/random for early signal). Run it:

```bash
JAX_PLATFORMS=cpu python -m baselines.mahjong_ppo \
    --updates 4000 --workers 24 --games-per-update 128 \
    --lr 3e-4 --eval-every 4 --out /root/ludus_train/mahjong/agent
```

It checkpoints `best.msgpack` / `last.msgpack` (flax) and logs reward, entropy,
self-play win count, and eval win-rates vs random/efficiency. On the Ludus box it
sustains **~50–90 games/s across 24 CPU cores** (pin `OMP_NUM_THREADS=1` /
`OPENBLAS_NUM_THREADS=1` to avoid BLAS oversubscription across workers).

**Reality check:** the ≥8-fan win condition makes this a hard sparse-reward
problem. From scratch, expect the reward to trend up and self-play wins to climb
while absolute win-rate stays low for a long time — the strong published agent
(kdens3) was *distilled from a large game corpus*, not learned tabula-rasa. Good
next levers: reward shaping toward tenpai/shanten, curriculum against weaker
opponents, behavioural-cloning warm-start from the champion, or league-style
opponent pools.

### Fused on-GPU trainer (`baselines/mahjong_t2_jax.py`)

For fine-tuning the published SL policy, `baselines/mahjong_t2_jax.py` runs the
**entire** rollout — the `vmap`ped `jax_env`, the kdens3 flax policy, and PPO —
on the GPU inside `lax.scan`; the only host touch is the terminal fan callback
(once per rollout, not per step). It is a **KL-leashed** fine-tune (trust-region
rollback + adaptive β, KL ≤ 0.05) so the policy improves without drifting off the
SL anchor. Measured on one 3090 (B=512, T=256):

| path | throughput | vs CPU trainer |
|---|---|---|
| pure fused rollout (env+policy), bf16 | **13,050 game-steps/s** | ~42× |
| full training loop (rollout+GAE+PPO+scoring), bf16 | **~7.9–8.8k env-steps/s** | **~26×** |

Policy parity vs the numpy champion is `0/300` argmax disagreements in both fp32
and bf16. Terminal reward is read from `host_score`, which **forfeits** any
sub-8-fan Hu (see correctness note below), so the agent cannot reward-hack the Hu
action — it may safely declare Hu and simply eats the forfeit if the hand is not a
legal ≥8-fan win.

### Live monitor

The platform serves a training dashboard at **`/train`** (API `/api/train`),
auto-refreshing per run: status (live/paused/idle), headline stats, and canvas
sparklines for mean-score, win-rate, deal-in, KL, and steps/s. It reads each
run's JSONL append-log; add a run with one entry in `TRAIN_RUNS`
(`arena/play_server.py`).

---

## Dependencies & notes

- **Engine (serving path):** pure-Python, CPU (~450 games/s single-thread with
  light bots); this `MahjongEnv` wraps it, so *this* trainer is CPU-bound and
  parallelises rollouts across processes.
- **JAX-native env (`mahjong/jax_env.py`):** a fully `vmap`pable MCR engine —
  fixed-size state tensors, a `lax.switch` phase machine, branchless
  priority-matrix claim resolution, and a `jax.pure_callback` fan scorer (v1
  hybrid). It reproduces the oracle **12,288/12,288** games exactly (221/221
  golden verified independently) and runs **~3.07M game-steps/s** on one 3090
  (batch 4096, env-only). This is the GPU-native path for fused env+policy
  self-play — an *earlier note here claimed the sim was inherently CPU because
  MCR is "sequential and branchy"; that was wrong* (pgx vmaps chess/Go/Shogi;
  one resolution round per discard is a fixed priority argmax, not recursion).
  v2 (a tensorized on-device fan scorer to drop the callback) is future work.
  **Correctness gates (for RL safety):** a Hu below the 8-fan MCR minimum is
  *forfeited* at scoring (`host_score` → declarer penalised, zero-sum) rather than
  paid, so an agent cannot farm illegal wins; and `legal_mask` caps melds at 4
  (no impossible 5th Chi/Peng/Gang/AnGang). Both are covered by regression tests
  in `tests/test_jax_env.py` and leave the oracle at 221/221.
- **Fan calculator:** `pip install PyMahjongGB` (the ground-truth referee).
- **Feature encoder:** the env imports `feature.py` from the kdens3 bundle
  (`deploy/bot_KDENS_*.zip` in the HF weights repo `Dannibal/ijcai-mahjong-ckpts-2026`);
  place it on `sys.path` (the Ludus box keeps it at `/root/mcr_champion/`). This is
  what guarantees the obs/action interface matches the champion's.
- **Correctness:** validate the underlying engine anytime against
  `Dannibal/mcr-final2026-testset` with
  `validate_engine.py --engine mahjong.validate_adapter:MyEngine <testset>`.
  As of commit `1a9b231` the engine emits the per-seat win-eligibility array
  `canHu[4]` on every display event, so **strict** mode (which also compares
  `canHu` + `tileCnt`, not just the loose `action/player/tile/hand/fan/score`
  keys) passes **12,288/12,288** — same as loose. Golden subset: 221/221.
