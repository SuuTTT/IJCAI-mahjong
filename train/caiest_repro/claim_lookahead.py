"""
claim_lookahead.py — value-guided claim sanity check.

Re-parses data.txt. At every CLAIM-LEGAL response state (an opponent Play where the responding
seat has Pass available AND at least one Chi/Peng available), we form a 1-ply lookahead with the
trained value model:
  - after-PASS obs  = the response-state observation (seat unchanged; it will pass and the game
    continues). V_pass = model value of this obs.
  - after-CLAIM obs = clone the agent, apply the Chi/Peng meld (the strongest available claim:
    prefer Peng, else the first legal Chi), read the resulting obs (now the seat must Play),
    then UnChi/UnPeng to restore. V_claim = model value of that obs.
We say "value recommends CLAIM" iff V_claim is better than V_pass.

VALUE direction: lower expected-placement number is better (1 best ... 4 worst), and higher
score is better, and lower P(4th) is better. We use expected placement from the place4 head as
the primary criterion (claim helps iff exp_place(after-claim) < exp_place(after-pass) - margin).
We also report with the 4th-head and score-head.

Reports the value-guided claim rate over claim-legal states, vs moyu 0.29 / leaders 0.25.
Subsamples matches for speed (--matches N).

  python3 claim_lookahead.py --model ckpt/value_mt.pkl --gpu 1 --matches 8000
"""
import os, sys, argparse, copy
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np, torch, torch.nn as nn
from feature import FeatureAgent

IN_PLANES = 38
OFF = FeatureAgent.OFFSET_ACT
CHI0, CHI1 = OFF['Chi'], OFF['Peng']          # Chi range [36,99)
PENG0, PENG1 = OFF['Peng'], OFF['Gang']       # Peng range [99,133)

class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class ValueMT(nn.Module):
    def __init__(self, channels=128, blocks=20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.place_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 4))
        self.fourth_head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
        self.score_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, obs):
        x = self.body(self.stem(obs)); x = x.mean(dim=(2,3))
        return self.place_head(x), self.fourth_head(x).squeeze(1), self.score_head(x).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/IJCAI-mahjong/train/caiest_repro/ckpt/value_mt.pkl')
    ap.add_argument('--data', default='/root/IJCAI-mahjong/train/caiest_repro/data/data.txt')
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--matches', type=int, default=8000)
    ap.add_argument('--margin', type=float, default=0.0)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}'; torch.cuda.set_device(a.gpu)
    ck = torch.load(a.model, map_location='cpu')
    net = ValueMT(ck['channels'], ck['blocks']); net.load_state_dict(ck['state']); net.to(dev).eval()
    print(f'loaded {a.model} ch={ck["channels"]} blocks={ck["blocks"]}', flush=True)

    # collect (after_pass_obs, after_claim_obs) pairs over claim-legal states
    pass_obs, claim_obs = [], []
    human_claimed = []   # did the human actually claim at this state?

    def value(obs_arr):
        ob = torch.from_numpy(np.stack(obs_arr).astype(np.float32)).to(dev)
        with torch.no_grad():
            pl, fo, sc = net(ob)
        sm = torch.softmax(pl, 1).cpu().numpy()
        exp_place = (sm * np.array([1,2,3,4])).sum(1)
        p4 = torch.sigmoid(fo).cpu().numpy()
        score = sc.cpu().numpy()
        return exp_place, p4, score

    agents = None; curTile = None; matchid = -1
    n_claimlegal = 0
    with open(a.data, encoding='UTF-8') as f:
        for line in f:
            t = line.split()
            if not t: continue
            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]; matchid += 1
                if matchid >= a.matches: break
                if matchid % 1000 == 0: print(f'match {matchid} claim-legal so far {n_claimlegal}', flush=True)
            elif t[0] == 'Wind':
                for ag in agents: ag.request2obs(line.strip())
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    for i in range(4):
                        if i == p: agents[p].request2obs(' '.join(t[2:]))
                        else: agents[i].request2obs(' '.join(t[:3]))
                elif t[2] == 'Play':
                    curTile = t[3]
                    # BEFORE applying the play to responders, the play creates response-states.
                    # We must apply the play to each responder i!=p to get their response obs.
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs(line.strip())
                        else:
                            o = agents[i].request2obs(line.strip())
                            m = o['observation']  # not used; mask below
                            am = o['action_mask']
                            chi_ok = bool(am[CHI0:CHI1].any()); peng_ok = bool(am[PENG0:PENG1].any())
                            if (am[OFF['Pass']] == 1) and (chi_ok or peng_ok):
                                n_claimlegal += 1
                                # after-pass obs = this response obs
                                ap_obs = o['observation'].astype(np.int8)
                                # choose claim action: prefer Peng, else first legal Chi
                                if peng_ok:
                                    claim_str = 'Peng'
                                else:
                                    ci = CHI0 + int(np.argmax(am[CHI0:CHI1]))
                                    tt = (ci - CHI0) // 3
                                    chitile = 'WTB'[tt//7] + str(tt%7 + 2)
                                    claim_str = 'Chi ' + chitile
                                # clone agent, apply claim, read obs, undo
                                ag2 = copy.deepcopy(agents[i])
                                try:
                                    if claim_str == 'Peng':
                                        oc = ag2.request2obs('Player %d Peng %s' % (i, '')) if False else None
                                except Exception:
                                    oc = None
                                # Use the documented request strings for self-claim
                                ag2 = copy.deepcopy(agents[i])
                                if claim_str == 'Peng':
                                    oc = ag2.request2obs('Player %d Peng' % i)
                                else:
                                    chitile = claim_str.split()[1]
                                    oc = ag2.request2obs('Player %d Chi %s' % (i, chitile))
                                if oc is None or 'observation' not in oc:
                                    n_claimlegal -= 1
                                    continue
                                ac_obs = oc['observation'].astype(np.int8)
                                pass_obs.append(ap_obs); claim_obs.append(ac_obs)
                                human_claimed.append(0)  # filled below if we detect a claim event
                elif t[2] in ('Chi','Peng','Gang','AnGang','BuGang','Hu'):
                    # apply to advance agent states (mirror preprocess) so later states stay valid
                    if t[2] == 'Chi':
                        for i in range(4):
                            if i == p: agents[p].request2obs('Player %d Chi %s' % (p, t[3]))
                            else: agents[i].request2obs('Player %d Chi %s' % (p, t[3]))
                    elif t[2] == 'Peng':
                        for i in range(4):
                            if i == p: agents[p].request2obs('Player %d Peng' % p)
                            else: agents[i].request2obs('Player %d Peng' % p)
                    elif t[2] == 'Gang':
                        for i in range(4): agents[i].request2obs('Player %d Gang %s' % (p, t[3]))
                    elif t[2] == 'AnGang':
                        for i in range(4):
                            if i == p: agents[p].request2obs('Player %d AnGang %s' % (p, t[3]))
                            else: agents[i].request2obs('Player %d AnGang' % p)
                    elif t[2] == 'BuGang':
                        for i in range(4): agents[i].request2obs('Player %d BuGang %s' % (p, t[3]))
            elif t[0] == 'Score':
                pass

    print(f'claim-legal states collected: {len(pass_obs)}', flush=True)
    if not pass_obs:
        print('NO claim-legal states found — check parser'); return
    # batch value eval
    def eval_all(arr):
        eps=[]; p4s=[]; scs=[]
        for i in range(0, len(arr), 8192):
            ep, p4, sc = value(arr[i:i+8192]); eps.append(ep); p4s.append(p4); scs.append(sc)
        return np.concatenate(eps), np.concatenate(p4s), np.concatenate(scs)
    ep_pass, p4_pass, sc_pass = eval_all(pass_obs)
    ep_claim, p4_claim, sc_claim = eval_all(claim_obs)

    m = a.margin
    rec_place = float((ep_claim < ep_pass - m).mean())   # claim if it lowers expected placement
    rec_4th   = float((p4_claim < p4_pass - m).mean())    # claim if it lowers P(4th)
    rec_score = float((sc_claim > sc_pass + m).mean())     # claim if it raises score
    print('=== VALUE-GUIDED CLAIM RECOMMENDATION (over claim-legal states) ===', flush=True)
    print(f'N claim-legal = {len(pass_obs)}', flush=True)
    print(f'  by expected-placement (lower better): claim-recommended rate = {rec_place:.4f}', flush=True)
    print(f'  by P(4th)            (lower better):  claim-recommended rate = {rec_4th:.4f}', flush=True)
    print(f'  by score             (higher better): claim-recommended rate = {rec_score:.4f}', flush=True)
    print(f'  mean exp_place: pass {ep_pass.mean():.3f} claim {ep_claim.mean():.3f} (delta {ep_claim.mean()-ep_pass.mean():+.3f})', flush=True)
    print(f'  mean P(4th):    pass {p4_pass.mean():.3f} claim {p4_claim.mean():.3f} (delta {p4_claim.mean()-p4_pass.mean():+.3f})', flush=True)
    print(f'  mean score:     pass {sc_pass.mean():+.3f} claim {sc_claim.mean():+.3f} (delta {sc_claim.mean()-sc_pass.mean():+.3f})', flush=True)
    print('compare: moyu claim rate 0.29 / leaders 0.25', flush=True)


if __name__ == '__main__':
    main()
