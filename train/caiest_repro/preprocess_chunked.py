"""
preprocess_chunked.py — memory-bounded variant of preprocess_single.py for small-RAM boxes.
Same per-decision extraction (all players, multi-option states only), but flushes compressed
shards every SHARD samples, then concatenates them into three uncompressed memmap arrays:
  data/cooked_obs.npy  int8  (N,38,4,9)
  data/cooked_mask.npy bool  (N,235)
  data/cooked_act.npy  int64 (N,)
distill.py finetune_frac auto-detects the triplet and np.load(mmap_mode='r')s it (bounded RAM).
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'data.txt')
SHARD_DIR = os.path.join(HERE, 'data', 'shards')
SHARD = 250_000

all_obs, all_mask, all_act = [], [], []
shard_id = [0]

def flush(obs, actions):
    for j in range(4):
        for i, a in enumerate(actions[j]):
            o = obs[j][i]
            if np.sum(o['action_mask']) != 1:
                all_obs.append(o['observation'].astype(np.int8))
                all_mask.append(o['action_mask'].astype(np.bool_))
                all_act.append(a)
    if len(all_obs) >= SHARD:
        write_shard()

def write_shard():
    if not all_obs: return
    p = os.path.join(SHARD_DIR, 'shard_%04d.npz' % shard_id[0])
    np.savez_compressed(p,
                        obs=np.stack(all_obs).reshape((-1, 38, 4, 9)).astype(np.int8),
                        mask=np.stack(all_mask).astype(np.bool_),
                        act=np.array(all_act, np.int16))
    print('wrote %s (%d samples)' % (p, len(all_obs)), flush=True)
    shard_id[0] += 1
    all_obs.clear(); all_mask.clear(); all_act.clear()

def concat_shards():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, 'shard_*.npz')))
    counts = []
    for s in shards:
        with np.load(s) as z: counts.append(len(z['act']))
    n = sum(counts)
    print('concat %d shards, N=%d' % (len(shards), n), flush=True)
    oo = np.lib.format.open_memmap(os.path.join(HERE, 'data', 'cooked_obs.npy'), mode='w+', dtype=np.int8, shape=(n, 38, 4, 9))
    mm = np.lib.format.open_memmap(os.path.join(HERE, 'data', 'cooked_mask.npy'), mode='w+', dtype=np.bool_, shape=(n, 235))
    aa = np.lib.format.open_memmap(os.path.join(HERE, 'data', 'cooked_act.npy'), mode='w+', dtype=np.int64, shape=(n,))
    k = 0
    for s, c in zip(shards, counts):
        with np.load(s) as z:
            oo[k:k+c] = z['obs']; mm[k:k+c] = z['mask']; aa[k:k+c] = z['act'].astype(np.int64)
        k += c
        os.remove(s)  # reclaim disk as we go
        print('  merged %s -> %d/%d' % (os.path.basename(s), k, n), flush=True)
    oo.flush(); mm.flush(); aa.flush()
    print('done: cooked_{obs,mask,act}.npy  N=%d' % n, flush=True)

def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
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
                    print('match %d  shards %d  buf %d' % (matchid, shard_id[0], len(all_obs)), flush=True)
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
    write_shard()
    print('total matches %d' % (matchid + 1), flush=True)
    concat_shards()

if __name__ == '__main__':
    main()
