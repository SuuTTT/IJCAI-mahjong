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
try:
    import torch                                   # primary path
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False                              # -> pure-NumPy fallback (zero torch-version risk)
from feature import FeatureAgent

MARKER = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
JSON_OUT = os.environ.get("BOTZONE_JSON", "1") != "0"
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_model():
    cands = []
    for base in (os.path.join(_HERE, 'data'), 'data', _HERE, '.'):
        cands += glob.glob(os.path.join(base, '*.pkl'))
    cands = sorted(set(cands), key=lambda p: os.path.getsize(p), reverse=True)
    return cands[0] if cands else None

def _fused_from_sd(sd):
    """Build a ResFused sized from the checkpoint's own keys (robust to any channels/blocks)."""
    from model_resfused import ResFused
    ch = sd['stem.weight'].shape[0]
    blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('body.') and k.endswith('.c1.weight'))
    m = ResFused(channels=ch, blocks=blocks); m.load_state_dict(sd); return m

def _cnn_from_sd(sd):
    from model import CNNModel
    m = CNNModel(); m.load_state_dict(sd); return m

def _load_model():
    """Load whatever checkpoint is present, auto-detecting the architecture from its KEYS so a
    mismatched/renamed upload can't hard-crash. Order: fused ResNet, then 16-block CNNModel.
    Returns (model_or_None, debug_str). On total failure -> None -> legal-fallback play (never RE)."""
    if not HAS_TORCH:
        return None, 'no_torch->numpy'
    path = os.environ.get('CAIEST_MODEL') or _find_model()
    try:
        sd = torch.load(path, map_location=torch.device('cpu'))
    except Exception as e:
        return None, 'load_fail:%s' % str(e)[:60]
    keys = list(sd.keys())
    if any('running_mean' in k for k in keys):         # un-fused BatchNorm net: unsafe on torch 1.4
        return None, 'got_batchnorm_ckpt(need_fused)'
    for builder in (_fused_from_sd, _cnn_from_sd):
        try:
            m = builder(sd); m.eval(); return m, 'ok:%s' % builder.__name__
        except Exception:
            continue
    return None, 'no_arch_matched_keys'

ENS = None                                             # inference-time mixture (P2) — highest priority if set
_ENS_SPEC = os.environ.get('ENSEMBLE_NPZS', '')
if _ENS_SPEC:
    try:
        paths = [p for p in _ENS_SPEC.split(',') if p.strip()]
        from ensemble_infer import Ensemble
        ENS = Ensemble(paths); MODEL_DBG_E = 'ensemble:%d' % ENS.n
    except Exception as e:
        ENS = None; MODEL_DBG_E = 'ensemble_fail:%s' % str(e)[:40]
else:
    MODEL_DBG_E = ''

model, MODEL_DBG = (None, 'ensemble') if ENS is not None else _load_model()
MODEL_DBG += '|' + MODEL_DBG_E if MODEL_DBG_E else ''

NP_MODEL = None                                        # pure-NumPy fallback (used if torch model failed)
if model is None and ENS is None:
    try:
        cands = []
        for base in (os.path.join(_HERE, 'data'), 'data', _HERE, '.'):
            cands += glob.glob(os.path.join(base, '*.npz'))
        if cands:
            npz = sorted(set(cands), key=lambda p: os.path.getsize(p), reverse=True)[0]
            from numpy_resfused import NumpyResFused
            NP_MODEL = NumpyResFused(npz); MODEL_DBG += '|numpy:%s' % os.path.basename(npz)
    except Exception as e:
        MODEL_DBG += '|numpy_fail:%s' % str(e)[:40]

agent = None
seatWind = 0
zimo = False
angang = None

SAFE = os.environ.get('SAFE_DISCARD', '0') == '1'      # opt-in defensive discard filter (A/B only)
try:
    import safe_discard as _sd
except Exception:
    _sd = None

def _pick(lg, mask):
    """argmax, optionally re-ranked through the genbutsu safe-discard filter for Play actions."""
    a = int(np.asarray(lg).flatten().argmax())
    if not (SAFE and _sd is not None and agent is not None):
        return a
    try:
        po = agent.OFFSET_ACT['Play']
        if not (po <= a < po + 34):
            return a
        lgf = np.asarray(lg).flatten()
        legal = [i for i in range(po, po + 34) if mask[i]]
        legal.sort(key=lambda i: -float(lgf[i]))
        t = _sd.choose_discard(agent, [agent.TILE_LIST[i - po] for i in legal])
        return po + agent.OFFSET_TILE[t]
    except Exception:
        return a                                       # filter must never break the bot

def obs2response(obs):
    if ENS is not None:                                # P2: inference-time mixture (NumPy, memory-light)
        lg = ENS.logits(obs['observation'], obs['action_mask'])
        return agent.action2response(_pick(lg, obs['action_mask']))
    if model is not None:                              # primary: torch
        with torch.no_grad():
            logits = model({'is_training': False,
                            'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                                    'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
        return agent.action2response(_pick(logits.detach().numpy(), obs['action_mask']))
    if NP_MODEL is not None:                           # fallback: pure NumPy (no torch)
        lg = NP_MODEL.logits(obs['observation'], obs['action_mask'])
        return agent.action2response(_pick(lg, obs['action_mask']))
    return agent.action2response(int(np.argmax(obs['action_mask'])))   # last resort: first legal action

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

def _replay_event(req, resp, nxt=''):
    """Rebuild state for a HISTORICAL (already-answered) request, applying our recorded
    action `resp` for our own moves (own moves are not echoed in JSON requests). Skips
    rt=3 pid==seat echoes. Used for JSON full-history replay (robust to a fresh process).
    `nxt` = the NEXT request in history: a recorded claim (CHI/PENG/GANG on a discard) is
    applied ONLY if the judge's echo confirms it — a higher-priority claim by another
    player (e.g. their PENG over our CHI) preempts ours, and applying it anyway desyncs
    the hand permanently (sim-7 audit: 5 wrong-HUs, all traced to exactly this)."""
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
        if not (rp and rp[0] in ('PENG', 'CHI', 'GANG')):
            return
        nt = (nxt or '').split()                  # judge echo must confirm the claim went through
        if not (len(nt) >= 3 and nt[0] == '3' and int(nt[1]) == seatWind and nt[2] == rp[0]):
            return                                # preempted by a higher-priority claim: ignore ours
        if rp[0] == 'PENG':
            agent.request2obs('Player %d Peng' % seatWind); agent.request2obs('Player %d Play %s' % (seatWind, rp[1]))
        elif rp[0] == 'CHI':
            agent.request2obs('Player %d Chi %s' % (seatWind, rp[1])); agent.request2obs('Player %d Play %s' % (seatWind, rp[2]))
        elif rp[0] == 'GANG':
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
        _replay_event(R[i], RESP[i], R[i + 1] if i + 1 < len(R) else '')
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
