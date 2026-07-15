"""Clone-independence check: deep-copy a mid-game sim, mutate the clone, confirm the
original is untouched. Also a 1-game timing + correctness smoke of the oracle rollout."""
import sys, copy, time
sys.path.insert(0, "/root/caiest_repro")
import s2_oracle_gate as S
S._init_worker()

# build a sim, advance a few turns manually to reach mid-game
from sim_cnn import ACT, TILE_LIST
sim = S.OracleSim([None] * 4, seed=8500000, quan=0, learner_seats=[], cnn=True)
sim.reset()
# run a few loop turns by hand: draw+discard for a couple seats using kdens3
for _ in range(3):
    cur = sim.cur
    if not sim.walls[cur]:
        break
    t = sim.walls[cur].pop(); sim.hand[cur].append(t)
    sim.cai[cur].request2obs(f"Draw {t}")
    for s in range(4):
        if s != cur:
            sim.cai[s].request2obs(f"Player {cur} Draw")
    act, _, _ = sim._kd_ask(cur)
    tile = TILE_LIST[act - ACT["Play"]] if ACT["Play"] <= act < ACT["Chi"] else sim.hand[cur][0]
    if tile not in sim.hand[cur]:
        tile = sim.hand[cur][0]
    sim.hand[cur].remove(tile)
    sim._broadcast(f"Player {cur} Play {tile}")
    nxt = sim._resolve_claims(tile, cur)
    if nxt == "HU":
        break
    sim.cur = nxt

# snapshot original state
orig_hands = [list(h) for h in sim.hand]
orig_walls = [len(w) for w in sim.walls]
orig_scores = list(sim.scores)
orig_cur = sim.cur

# clone and mutate hard: run a full rollout on the clone
clone = copy.deepcopy(sim)
clone.search_seat = -1
seat = clone.cur
if clone.hand[seat]:
    place, score = clone._rollout_discard(seat, clone.hand[seat][0])

# verify original unchanged
ok = (orig_hands == [list(h) for h in sim.hand]
      and orig_walls == [len(w) for w in sim.walls]
      and orig_scores == list(sim.scores)
      and orig_cur == sim.cur)
print("CLONE-INDEPENDENCE:", "OK" if ok else "BROKEN")
print("  rollout result placement/score:", round(place, 3), score)
print("  clone rollout depth (asks):", clone._asks)

# timing: one full oracle game (4 rotations), seed 8500001
t0 = time.time()
r = S._work((0, 8500001))
print("ONE-SEED oracle (4 rotations) sec:", round(time.time() - t0, 1),
      "psum:", round(r[2], 2), "overrides:", r[3], "decisions:", r[4],
      "mean_depth:", round(r[5] / max(1, r[6]), 1))
