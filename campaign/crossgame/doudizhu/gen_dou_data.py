"""Generate Doudizhu BC data by DouDizhuRuleAgentV1 self-play.
Extracts per-decision (obs int8[790], action_id int32, padded legal ids int32[MAXLEGAL]).
Parallel across CPU cores. Saves compressed npz."""
import argparse, os, time
import numpy as np
import multiprocessing as mp
import rlcard
from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
import rlcard.games.doudizhu.utils as U

A2I = U.ACTION_2_ID
MAXLEGAL = 400   # observed max legal ~307; pad to 400 with -1
OBS_DIM = 901    # state_shape [[790],[901],[901]]; landlord's 790 is zero-padded to 901

class NoisyRuleAgent(object):
    """DouDizhuRuleAgentV1 that with prob epsilon plays a uniformly-random LEGAL action
    instead of the rule action. Injects label noise / target diversity. Raw agent."""
    use_raw = True
    def __init__(self, epsilon, rng):
        self.base = DouDizhuRuleAgentV1()
        self.eps = epsilon
        self.rng = rng
    def step(self, state):
        if self.eps > 0 and self.rng.random() < self.eps:
            raw_legal = list(state["raw_legal_actions"])
            return raw_legal[self.rng.randint(len(raw_legal))]
        return self.base.step(state)
    def eval_step(self, state):
        return self.step(state), {}

def worker(args):
    wid, ngames, seed_base, epsilon = args
    if epsilon > 0:
        rng = np.random.RandomState(seed_base + wid)
        ag = NoisyRuleAgent(epsilon, rng)
    else:
        ag = DouDizhuRuleAgentV1()
    env = rlcard.make("doudizhu", config={"seed": seed_base + wid})
    env.set_agents([ag, ag, ag])
    obs_l, act_l, leg_l = [], [], []
    local_max = 0
    for g in range(ngames):
        trajs, _ = env.run(is_training=True)
        for seat in trajs:
            # seat = [state, raw_action, state, raw_action, ..., final_state]
            for i in range(0, len(seat) - 1, 2):
                st = seat[i]
                raw_act = seat[i + 1]
                if not isinstance(st, dict):
                    continue
                aid = A2I.get(raw_act, None)
                if aid is None:
                    continue
                legal_ids = list(st["legal_actions"].keys())
                local_max = max(local_max, len(legal_ids))
                if len(legal_ids) > MAXLEGAL:
                    legal_ids = legal_ids[:MAXLEGAL]
                padded = np.full(MAXLEGAL, -1, dtype=np.int32)
                padded[:len(legal_ids)] = np.asarray(legal_ids, dtype=np.int32)
                ob = np.asarray(st["obs"], dtype=np.int8)
                if ob.shape[0] < OBS_DIM:
                    ob = np.concatenate([ob, np.zeros(OBS_DIM - ob.shape[0], np.int8)])
                obs_l.append(ob)
                act_l.append(np.int32(aid))
                leg_l.append(padded)
    if not obs_l:
        return (np.zeros((0, OBS_DIM), np.int8), np.zeros((0,), np.int32),
                np.zeros((0, MAXLEGAL), np.int32), local_max)
    return (np.stack(obs_l), np.asarray(act_l, np.int32), np.stack(leg_l), local_max)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="prob of a uniformly-random legal action (label noise); 0=pure rule agent")
    ap.add_argument("--out", type=str, default="dou_data.npz")
    args = ap.parse_args()
    t0 = time.time()
    per = args.games // args.workers
    tasks = [(w, per, args.seed, args.epsilon) for w in range(args.workers)]
    print(f"epsilon(label-noise)={args.epsilon}", flush=True)
    print(f"launching {args.workers} workers x {per} games = {per*args.workers} games", flush=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(worker, tasks)
    obs = np.concatenate([r[0] for r in results], axis=0)
    act = np.concatenate([r[1] for r in results], axis=0)
    leg = np.concatenate([r[2] for r in results], axis=0)
    gmax = max(r[3] for r in results)
    print(f"decisions={len(obs)} obs{obs.shape}{obs.dtype} act{act.shape} leg{leg.shape} MAXLEGAL_observed={gmax}", flush=True)
    np.savez_compressed(args.out, obs=obs, act=act, legal=leg)
    print(f"saved {args.out} in {time.time()-t0:.1f}s size={os.path.getsize(args.out)/1e6:.1f}MB", flush=True)

if __name__ == "__main__":
    main()
