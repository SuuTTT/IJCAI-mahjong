# Serve any exploration arch (models_explore) through the proven deploy I/O, for benchmarking.
# Env: EXP_KIND, EXP_CFG (json), CAIEST_MODEL (state_dict .pkl), BOTZONE_JSON=0 for local kr.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from feature import FeatureAgent
from models_explore import build
import importlib.util
# reuse the deploy bot's process()/run() I/O by importing its module-level handler logic
_spec = importlib.util.spec_from_file_location(
    "deploybot", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "deploy", "caiest_cnn", "__main__.py"))
# Simplest: replicate the minimal serving loop here using models_explore model + feature.py.
SENT = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
JSON_OUT = os.environ.get("BOTZONE_JSON", "1") != "0"
model = build(os.environ["EXP_KIND"], **json.loads(os.environ.get("EXP_CFG", "{}")))
model.load_state_dict(torch.load(os.environ["CAIEST_MODEL"], map_location="cpu")); model.eval()
agent=None; seatWind=0; zimo=False; angang=None
FAN_MASK = os.environ.get("FAN_MASK", "0") != "0"
if FAN_MASK:
    import fan_mask

def _logits(obs):
    with torch.no_grad():
        lg = model({'is_training': False, 'obs': {'observation': torch.from_numpy(obs['observation'][None]),
                                                  'action_mask': torch.from_numpy(obs['action_mask'][None])}})
    return lg.numpy().flatten()

def a2r(obs):
    return agent.action2response(int(_logits(obs).argmax()))

def ranked_plays(obs):
    """Legal discard tiles ordered by model preference (best first)."""
    lg = _logits(obs); base = agent.OFFSET_ACT['Play']; m = obs['action_mask']
    cand = [(lg[base + i], agent.TILE_LIST[i]) for i in range(34) if m[base + i]]
    cand.sort(key=lambda x: -x[0])
    return [t for _, t in cand]

def process(req):
    global agent, seatWind, zimo, angang
    t = req.split()
    if not t: return 'PASS'
    if t[0] == '0':
        seatWind=int(t[1]); agent=FeatureAgent(seatWind); agent.request2obs('Wind %s'%t[2]); return 'PASS'
    if t[0] == '1': agent.request2obs(' '.join(['Deal',*t[5:]])); return 'PASS'
    if t[0] == '2':
        o = agent.request2obs('Draw %s'%t[1]); r=a2r(o).split()
        if r[0]=='Hu': return 'HU'
        if r[0]=='Play':
            if FAN_MASK:
                packs=[(p[0],p[1]) for p in agent.packs[0]]
                ch=fan_mask.choose_discard(agent.hand, packs, agent.seatWind, agent.prevalentWind, ranked_plays(o))
                return 'PLAY %s'%ch
            return 'PLAY %s'%r[1]
        if r[0]=='Gang': return 'GANG %s'%r[1]
        if r[0]=='BuGang': return 'BUGANG %s'%r[1]
        return 'PASS'
    if t[0] == '3':
        p=int(t[1])
        if t[2]=='DRAW': agent.request2obs('Player %d Draw'%p); zimo=True; return 'PASS'
        if t[2]=='GANG':
            agent.request2obs(('Player %d AnGang %s'%(p,angang)) if (p==seatWind and angang) else ('Player %d AnGang'%p if zimo else 'Player %d Gang'%p)); return 'PASS'
        if t[2]=='BUGANG':
            o=agent.request2obs('Player %d BuGang %s'%(p,t[3])); return 'PASS' if p==seatWind else ('HU' if a2r(o)=='Hu' else 'PASS')
        zimo=False
        if t[2]=='CHI': agent.request2obs('Player %d Chi %s'%(p,t[3]))
        elif t[2]=='PENG': agent.request2obs('Player %d Peng'%p)
        o=agent.request2obs('Player %d Play %s'%(p,t[-1]))
        if p==seatWind: return 'PASS'
        r=a2r(o); rs=r.split()
        if rs[0]=='Hu': return 'HU'
        if rs[0]=='Pass': return 'PASS'
        if rs[0]=='Gang': angang=None; return 'GANG'
        if rs[0] in ('Peng','Chi'):
            o2=agent.request2obs('Player %d '%seatWind+r); d=a2r(o2); agent.request2obs('Player %d Un'%seatWind+r)
            return ' '.join([rs[0].upper(),*rs[1:],d.split()[-1]])
        return 'PASS'
    return 'PASS'

def emit(r):
    print(json.dumps({"response": r}) if JSON_OUT else r, flush=True)
    if not JSON_OUT: print(SENT, flush=True)

def run():
    while True:
        line=sys.stdin.readline()
        if not line: break
        s=line.strip()
        if not s or s=='1': continue
        try: emit(process(s if not s.startswith('{') else (json.loads(s).get('requests') or ['PASS'])[-1]))
        except Exception: emit('PASS')

if __name__=='__main__': run()
