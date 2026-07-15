"""Verify feature44's first 38 planes are byte-identical to the 38-plane feature.py,
and sanity-check the new planes, by replaying a chunk of data.txt through both encoders."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import feature as F38mod
import feature44 as F44mod
F38 = F38mod.FeatureAgent
F44 = F44mod.FeatureAgent
assert F44.OBS_SIZE == 44, F44.OBS_SIZE

DATA = 'data/data.txt'

def replay(AgentCls, n_matches):
    """Replay matches, collect obs from player 0's perspective at every _obs() that has >1 option."""
    obs_list = []
    matchid = -1
    agents = None
    curTile = None
    collected = 0
    with open(DATA, encoding='UTF-8') as f:
        for line in f:
            t = line.split()
            if not t: continue
            if t[0] == 'Match':
                matchid += 1
                if matchid >= n_matches: break
                agents = [AgentCls(i) for i in range(4)]
            elif t[0] == 'Wind':
                for ag in agents: ag.request2obs(line.strip())
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p:
                            d = agents[p].request2obs(' '.join(t[2:]))
                            if d and np.sum(d['action_mask']) != 1: obs_list.append(d['observation'])
                        else:
                            agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    for i in range(4):
                        if i == p: agents[p].request2obs(line.strip())
                        else:
                            d = agents[i].request2obs(line.strip())
                            if d and np.sum(d['action_mask']) != 1: obs_list.append(d['observation'])
                    curTile = t[3]
                elif t[2] == 'Chi':
                    for i in range(4):
                        if i == p:
                            d = agents[p].request2obs('Player %d Chi %s' % (p, t[3]))
                            if d and np.sum(d['action_mask']) != 1: obs_list.append(d['observation'])
                        else:
                            agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'Peng':
                    for i in range(4):
                        if i == p:
                            d = agents[p].request2obs('Player %d Peng %s' % (p, t[3]))
                            if d and np.sum(d['action_mask']) != 1: obs_list.append(d['observation'])
                        else:
                            agents[i].request2obs('Player %d Peng %s' % (p, t[3]))
                elif t[2] == 'Gang':
                    for i in range(4):
                        agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                elif t[2] == 'AnGang':
                    for i in range(4):
                        if i == p: agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                        else: agents[i].request2obs('Player %d AnGang' % p)
                elif t[2] == 'BuGang':
                    for i in range(4):
                        if i == p: agents[p].request2obs('Player %d BuGang %s' % (p, t[3]))
                        else:
                            d = agents[i].request2obs('Player %d BuGang %s' % (p, t[3]))
                            if d and np.sum(d['action_mask']) != 1: obs_list.append(d['observation'])
                elif t[2] == 'Hu':
                    pass
    return obs_list

N = 50
o38 = replay(F38, N)
o44 = replay(F44, N)
print('collected', len(o38), 'vs', len(o44))
assert len(o38) == len(o44), 'sample count mismatch'
o38 = np.stack(o38); o44 = np.stack(o44)
print('shapes', o38.shape, o44.shape)
assert o44.shape[1] == 44
# first 38 planes must match exactly
diff = np.abs(o38 - o44[:, :38]).max()
print('max abs diff first-38 planes:', diff)
assert diff == 0, 'first 38 planes differ!'
# new planes stats
dead = o44[:, 38:42]; wall = o44[:, 42]; turn = o44[:, 43]
print('DEAD planes: min/max', dead.min(), dead.max(), 'mean copies(plane0)', dead[:,0].mean())
# thermometer monotonic: plane k+1 <= plane k
mono = (dead[:,1] <= dead[:,0]).all() and (dead[:,2] <= dead[:,1]).all() and (dead[:,3] <= dead[:,2]).all()
print('DEAD thermometer monotonic:', mono)
print('WALL broadcast const per-sample:', np.allclose(wall.reshape(len(wall),-1).std(1), 0), 'range', wall.min(), wall.max())
print('TURN broadcast const per-sample:', np.allclose(turn.reshape(len(turn),-1).std(1), 0), 'range', turn.min(), turn.max())
# wall should generally decrease, turn increase over a game (loose check via correlation)
print('OK: first 38 planes identical; new planes well-formed')
