# Botzone deploy entry for the reproduced CNN agent (Chinese Standard Mahjong).
#
# Uses caiest's ORIGINAL, proven per-request handler (0 illegal, +1039 vs r18 in local
# judge benchmarks), with three deploy-safety wrappers:
#   1) LONG-RUNNING: load torch+model ONCE, stay alive across turns (the CNN is too heavy to
#      reload per turn). State persists in module globals; each turn we process the CURRENT
#      request only (requests[-1]) and maintain state via Botzone's echoes (caiest's design).
#   2) JSON I/O: Botzone expects {"response": ...} output (raw text -> "not JSON" error).
#      Output JSON followed by the keep-running marker. Input may be a JSON blob
#      {"requests":[...]} or a raw request line; we handle both, plus the leading "1".
#   3) EOF-safe: read via sys.stdin.readline and BREAK on EOF (input() raised EOFError -> RE).
# Set BOTZONE_JSON=0 to emit raw responses (for the local run_match_kr keep-running harness).
import os, sys, glob, json
import numpy as np
import torch
from feature import FeatureAgent
from model import CNNModel

MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
JSON_OUT = os.environ.get("BOTZONE_JSON", "1") != "0"

def _find_model():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = []
    for base in (os.path.join(here, 'data'), 'data', here, '.'):
        cands += glob.glob(os.path.join(base, '*.pkl'))
    cands = sorted(set(cands), key=lambda p: os.path.getsize(p), reverse=True)
    return cands[0] if cands else None

model = CNNModel()
model.load_state_dict(torch.load(os.environ.get('CAIEST_MODEL') or _find_model(),
                                  map_location=torch.device('cpu')))
model.eval()

agent = None
seatWind = 0
zimo = False
angang = None

def obs2response(obs):
    with torch.no_grad():
        logits = model({'is_training': False,
                        'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                                'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
    return agent.action2response(int(logits.detach().numpy().flatten().argmax()))

def process(request):
    """caiest's original per-request logic, returning the Botzone response string."""
    global agent, seatWind, zimo, angang
    t = request.split()
    if not t:
        return 'PASS'
    if t[0] == '0':
        seatWind = int(t[1]); agent = FeatureAgent(seatWind)
        agent.request2obs('Wind %s' % t[2]); return 'PASS'
    if t[0] == '1':
        agent.request2obs(' '.join(['Deal', *t[5:]])); return 'PASS'
    if t[0] == '2':
        obs = agent.request2obs('Draw %s' % t[1]); r = obs2response(obs).split()
        if r[0] == 'Hu': return 'HU'
        if r[0] == 'Play': return 'PLAY %s' % r[1]
        if r[0] == 'Gang': angang = r[1]; return 'GANG %s' % r[1]
        if r[0] == 'BuGang': return 'BUGANG %s' % r[1]
        return 'PASS'
    if t[0] == '3':
        p = int(t[1])
        if t[2] == 'DRAW':
            agent.request2obs('Player %d Draw' % p); zimo = True; return 'PASS'
        if t[2] == 'GANG':
            if p == seatWind and angang: agent.request2obs('Player %d AnGang %s' % (p, angang))
            elif zimo: agent.request2obs('Player %d AnGang' % p)
            else: agent.request2obs('Player %d Gang' % p)
            return 'PASS'
        if t[2] == 'BUGANG':
            obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
            if p == seatWind: return 'PASS'
            return 'HU' if obs2response(obs) == 'Hu' else 'PASS'
        # PLAY / CHI / PENG by someone
        zimo = False
        if t[2] == 'CHI': agent.request2obs('Player %d Chi %s' % (p, t[3]))
        elif t[2] == 'PENG': agent.request2obs('Player %d Peng' % p)
        obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
        if p == seatWind:
            return 'PASS'
        r = obs2response(obs); rs = r.split()
        if rs[0] == 'Hu': return 'HU'
        if rs[0] == 'Pass': return 'PASS'
        if rs[0] == 'Gang': angang = None; return 'GANG'
        if rs[0] in ('Peng', 'Chi'):
            obs = agent.request2obs('Player %d ' % seatWind + r)
            r2 = obs2response(obs)
            out = ' '.join([rs[0].upper(), *rs[1:], r2.split()[-1]])
            agent.request2obs('Player %d Un' % seatWind + r)
            return out
        return 'PASS'
    return 'PASS'

def emit(resp):
    if JSON_OUT:
        # Botzone simple interaction parses the WHOLE stdout as JSON -> output ONLY the JSON,
        # NO keep-running marker (the marker makes stdout invalid JSON -> "not JSON" / NJ).
        print(json.dumps({"response": resp}), flush=True)
    else:
        print(resp, flush=True)
        print(MARKER, flush=True)   # raw keep-running marker (local run_match_kr only)

def _replay_event(req, resp):
    """Rebuild state for a HISTORICAL (already-answered) request, applying our recorded
    action `resp` for our own moves (own moves are not echoed in JSON requests). Skips
    rt=3 pid==seat echoes. Used for JSON full-history replay (robust to a fresh process)."""
    global zimo, angang
    t = req.split()
    if not t: return
    if t[0] in ('0', '1'):
        process(req); return                      # INIT / Deal: pure state
    if t[0] == '2':                               # my draw, then my recorded action
        agent.request2obs('Draw %s' % t[1]); zimo = False
        rp = resp.split()
        if rp and rp[0] == 'PLAY':   agent.request2obs('Player %d Play %s' % (seatWind, rp[1]))
        elif rp and rp[0] == 'GANG': angang = rp[1]; agent.request2obs('Player %d AnGang %s' % (seatWind, rp[1]))
        elif rp and rp[0] == 'BUGANG': agent.request2obs('Player %d BuGang %s' % (seatWind, rp[1]))
        return
    if t[0] == '3':
        p = int(t[1])
        if p == seatWind:
            return                                # own echo: already applied via recorded resp
        if t[2] == 'DRAW':
            agent.request2obs('Player %d Draw' % p); zimo = True; return
        if t[2] == 'GANG':
            agent.request2obs('Player %d AnGang' % p if zimo else 'Player %d Gang' % p); return
        if t[2] == 'BUGANG':
            agent.request2obs('Player %d BuGang %s' % (p, t[3])); return
        zimo = False
        if t[2] == 'CHI': agent.request2obs('Player %d Chi %s' % (p, t[3]))
        elif t[2] == 'PENG': agent.request2obs('Player %d Peng' % p)
        agent.request2obs('Player %d Play %s' % (p, t[-1]))
        rp = resp.split()                         # my recorded claim on this discard
        if rp and rp[0] == 'PENG':
            agent.request2obs('Player %d Peng' % seatWind); agent.request2obs('Player %d Play %s' % (seatWind, rp[1]))
        elif rp and rp[0] == 'CHI':
            agent.request2obs('Player %d Chi %s' % (seatWind, rp[1])); agent.request2obs('Player %d Play %s' % (seatWind, rp[2]))
        elif rp and rp[0] == 'GANG':
            agent.request2obs('Player %d Gang' % seatWind)

def run_json(blob):
    """Rebuild state from full request/response history, then decide the current request."""
    global agent, seatWind, zimo, angang
    d = json.loads(blob)
    R = d.get('requests') or []; RESP = d.get('responses') or []
    if not R: emit('PASS'); return
    agent = None; zimo = False; angang = None
    process(R[0])                                  # INIT -> init agent
    for i in range(1, len(RESP)):                  # historical answered turns
        _replay_event(R[i], RESP[i])
    resp = process(R[-1]) if len(R) > len(RESP) else 'PASS'   # current decision
    emit(resp)

def run():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        s = line.strip()
        if s.startswith('{') and s.count('{') > s.count('}'):   # multi-line JSON
            while s.count('{') > s.count('}'):
                more = sys.stdin.readline()
                if not more: break
                s += more
        if not s or s == '1':
            continue
        try:
            if s.startswith('{'):
                d = json.loads(s); R = d.get('requests') or []; RESP = d.get('responses') or []
                if agent is None and (len(R) > 1 or RESP):
                    run_json(s)                    # fresh process + full history -> replay
                elif R:
                    emit(process(R[-1]))           # persistent or current-only -> incremental
                else:
                    emit('PASS')
            else:
                emit(process(s))                   # raw incremental (long-running)
        except Exception:
            emit('PASS')

if __name__ == '__main__':
    run()
