"""
preprocess_value.py — same per-decision extraction & cleaning as preprocess_single.py, but
ALSO emits, per kept decision, the acting seat's deal OUTCOME (final MCR duplicate score for
that hand from the 'Score a b c d' line, and the placement 1/2/3/4 obtained by ranking the
four seats' deal scores).

Produces data/cooked_value.npz = (obs int8 (N,38,4,9), mask bool (N,235), act int16 (N,),
  deal_score float32 (N,), deal_place int8 (N,)) with EXACTLY the same N and ordering as
cooked_single.npz (the per-decision loop is identical), so labels align 1:1 with cooked_single.

placement: rank seats by deal score; 1 = best (most points), 4 = worst. Ties share the better
rank by stable ordering (argsort descending then position+1); we report the resulting dist.
savez_compressed (disk-tight).
"""
import os, sys
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np
from feature import FeatureAgent

OUT = '/root/IJCAI-mahjong/train/caiest_repro/data/cooked_value.npz'
DATA = '/root/IJCAI-mahjong/train/caiest_repro/data/data.txt'

all_obs, all_mask, all_act, all_score, all_place = [], [], [], [], []


def placement_from_scores(scores):
    # scores: list of 4 ints. place[seat] = 1..4, 1 = highest score.
    order = sorted(range(4), key=lambda s: -scores[s])  # seats best->worst
    place = [0, 0, 0, 0]
    for rank, seat in enumerate(order):
        place[seat] = rank + 1
    return place


def flush(obs, actions, scores):
    place = placement_from_scores(scores)
    for j in range(4):
        for i, a in enumerate(actions[j]):
            o = obs[j][i]
            if np.sum(o['action_mask']) != 1:
                all_obs.append(o['observation'].astype(np.int8))
                all_mask.append(o['action_mask'].astype(np.bool_))
                all_act.append(a)
                all_score.append(np.float32(scores[j]))
                all_place.append(np.int8(place[j]))


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
                scores = [int(t[1]), int(t[2]), int(t[3]), int(t[4])]
                flush(obs, actions, scores)
                for x in obs: x.clear()
                for x in actions: x.clear()
    print(f'total matches {matchid+1}  total samples {len(all_obs)}', flush=True)
    obs_a = np.stack(all_obs).reshape((-1, 38, 4, 9)).astype(np.int8)
    mask_a = np.stack(all_mask).astype(np.bool_)
    act_a = np.array(all_act, dtype=np.int16)
    score_a = np.array(all_score, dtype=np.float32)
    place_a = np.array(all_place, dtype=np.int8)
    print(f'shapes obs {obs_a.shape} mask {mask_a.shape} act {act_a.shape} score {score_a.shape} place {place_a.shape}', flush=True)
    # placement distribution
    for pl in (1, 2, 3, 4):
        frac = float((place_a == pl).mean())
        print(f'  place=={pl}: {frac:.4f}', flush=True)
    print(f'  score mean {score_a.mean():+.3f} std {score_a.std():.3f} min {score_a.min()} max {score_a.max()}', flush=True)
    np.savez_compressed(OUT, obs=obs_a, mask=mask_a, act=act_a, deal_score=score_a, deal_place=place_a)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
