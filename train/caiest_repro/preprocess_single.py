"""
preprocess_single.py — same per-decision extraction as caiest's preprocess.py (all players,
filtering out forced single-option states), but accumulates ALL samples and writes ONE
compressed npz (obs int8 (N,38,4,9), mask bool (N,235), act int16 (N,)) instead of 98k
uncompressed per-match files (which blew the disk). Compressed sparse binary planes are tiny.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cooked_single.npz')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data.txt')

all_obs, all_mask, all_act = [], [], []

def flush(obs, actions):
    # keep only multi-option decisions (mask sum != 1), per caiest cleaning
    for j in range(4):
        for i, a in enumerate(actions[j]):
            o = obs[j][i]
            if np.sum(o['action_mask']) != 1:
                all_obs.append(o['observation'].astype(np.int8))
                all_mask.append(o['action_mask'].astype(np.bool_))
                all_act.append(a)

def main():
    obs = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    agents = None
    curTile = None
    matchid = -1
    with open(DATA, encoding='UTF-8') as f:
        for line in f:
            t = line.split()
            if not t: continue
            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]
                matchid += 1
                if matchid % 4000 == 0:
                    print(f'match {matchid}  samples so far {len(all_obs)}', flush=True)
            elif t[0] == 'Wind':
                for ag in agents: ag.request2obs(line.strip())
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(' '.join(t[2:]))); actions[p].append(0)
                        else:
                            agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    actions[p].pop(); actions[p].append(agents[p].response2action(' '.join(t[2:])))
                    for i in range(4):
                        if i == p: agents[p].request2obs(line.strip())
                        else:
                            obs[i].append(agents[i].request2obs(line.strip())); actions[i].append(0)
                    curTile = t[3]
                elif t[2] == 'Chi':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[3])))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Chi %s' % (p, t[3]))); actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'Peng':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Peng %s' % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs('Player %d Peng %s' % (p, t[3]))); actions[p].append(0)
                        else:
                            agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                elif t[2] == 'Gang':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Gang %s' % t[3]))
                    for i in range(4):
                        agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
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
                            obs[i].append(agents[i].request2obs('Player %d BuGang %s' % (p, t[3]))); actions[i].append(0)
                elif t[2] == 'Hu':
                    actions[p].pop(); actions[p].append(agents[p].response2action('Hu'))
                if t[2] in ['Peng', 'Gang', 'Hu']:
                    for k in range(5, 15, 5):
                        if len(t) > k:
                            p = int(t[k + 1])
                            if t[k + 2] == 'Chi':
                                actions[p].pop(); actions[p].append(agents[p].response2action('Chi %s %s' % (curTile, t[k + 3])))
                            elif t[k + 2] == 'Peng':
                                actions[p].pop(); actions[p].append(agents[p].response2action('Peng %s' % t[k + 3]))
                            elif t[k + 2] == 'Gang':
                                actions[p].pop(); actions[p].append(agents[p].response2action('Gang %s' % t[k + 3]))
                            elif t[k + 2] == 'Hu':
                                actions[p].pop(); actions[p].append(agents[p].response2action('Hu'))
                        else: break
            elif t[0] == 'Score':
                flush(obs, actions)
                for x in obs: x.clear()
                for x in actions: x.clear()
    print(f'total matches {matchid+1}  total samples {len(all_obs)}', flush=True)
    obs_a = np.stack(all_obs).reshape((-1, 38, 4, 9)).astype(np.int8)
    mask_a = np.stack(all_mask).astype(np.bool_)
    act_a = np.array(all_act, dtype=np.int16)
    print(f'shapes obs {obs_a.shape} mask {mask_a.shape} act {act_a.shape}', flush=True)
    np.savez_compressed(OUT, obs=obs_a, mask=mask_a, act=act_a)
    print(f'wrote {OUT}', flush=True)

if __name__ == '__main__':
    main()
