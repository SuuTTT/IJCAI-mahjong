"""
oracle_extract.py (Suphx oracle-guiding, foundation) — = preprocess_single's PROVEN replay of
data.txt, augmented with ORACLE planes: at each captured decision for seat p, append the 3 opponents'
TRUE concealed hands (read from the other FeatureAgents' .hand at that instant) as 12 count planes.
Oracle obs = (50,4,9) = 38 normal + 12 opponent-hand. Train an oracle teacher on these, then distill
to the 38-plane public student (hidden info is a TRAIN-time scaffold only).

  python3 oracle_extract.py --data data/data.txt --out data/oracle_cooked.npz --limit-matches 20000
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

TIDX = {t: i for i, t in enumerate(FeatureAgent.TILE_LIST)}

def opp_planes(agents, seat):
    """3 opponents' concealed hands -> (12,4,9): each opp 4 count-levels, padded 34->36."""
    out = np.zeros((12, 36), np.int8); k = 0
    for d in (1, 2, 3):
        o = (seat + d) % 4
        cnt = np.zeros(34, np.int8)
        for t in getattr(agents[o], 'hand', []):
            if t in TIDX: cnt[TIDX[t]] += 1
        for lvl in range(4):
            out[k, :34] = (cnt > lvl).astype(np.int8); k += 1
    return out.reshape(12, 4, 9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit-matches', type=int, default=0)
    a = ap.parse_args()
    all_obs, all_mask, all_act = [], [], []
    obs = [[] for _ in range(4)]; orc = [[] for _ in range(4)]; actions = [[] for _ in range(4)]
    agents = None; curTile = None; matchid = -1

    def cap(p):
        o = agents[p].request2obs_last if False else None
        return None

    def flush():
        for j in range(4):
            for i, act in enumerate(actions[j]):
                o = obs[j][i]
                if np.sum(o['action_mask']) != 1:
                    base = o['observation'].astype(np.int8)            # (38,4,9)
                    full = np.concatenate([base, orc[j][i]], axis=0)   # (50,4,9)
                    all_obs.append(full); all_mask.append(o['action_mask'].astype(np.bool_)); all_act.append(act)

    with open(a.data, encoding='UTF-8') as f:
        for line in f:
            t = line.split()
            if not t: continue
            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]; matchid += 1
                if a.limit_matches and matchid >= a.limit_matches: break
                if matchid % 2000 == 0: print(f'match {matchid} samples {len(all_obs)}', flush=True)
            elif t[0] == 'Wind':
                for ag in agents: ag.request2obs(line.strip())
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(' '.join(t[2:]))); orc[p].append(opp_planes(agents, p)); actions[p].append(0)
                        else:
                            agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    actions[p].pop(); actions[p].append(agents[p].response2action(' '.join(t[2:])))
                    for i in range(4):
                        if i == p: agents[p].request2obs(line.strip())
                        else:
                            obs[i].append(agents[i].request2obs(line.strip())); orc[i].append(opp_planes(agents, i)); actions[i].append(0)
                    curTile = t[3]
                elif t[2] == 'Chi':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3]))); orc[p].append(opp_planes(agents, p)); actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'Peng':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3]))); orc[p].append(opp_planes(agents, p)); actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                elif t[2] == 'Gang':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                    for i in range(4): agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                elif t[2] == 'AnGang':
                    actions[p].pop(); actions[p].append(agents[p].response2action('AnGang %s' % t[3]))
                    for i in range(4):
                        if i == p: agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                        else: agents[i].request2obs('Player %d AnGang' % p)
                elif t[2] == 'BuGang':
                    actions[p].pop(); actions[p].append(agents[p].response2action('BuGang %s' % t[3]))
                    for i in range(4):
                        if i == p: agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                        else:
                            obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3]))); orc[i].append(opp_planes(agents, i)); actions[i].append(0)
                elif t[2] == 'Hu':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Hu'))
            elif t[0] == 'Score':
                flush()
                for x in obs: x.clear()
                for x in orc: x.clear()
                for x in actions: x.clear()
    obs_a = np.stack(all_obs).astype(np.int8); mask_a = np.stack(all_mask).astype(np.bool_); act_a = np.array(all_act, np.int16)
    print(f'oracle: {len(act_a)} samples obs {obs_a.shape} -> {a.out}', flush=True)
    np.savez_compressed(a.out, obs=obs_a, mask=mask_a, act=act_a)

if __name__ == '__main__':
    main()
