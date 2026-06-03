# Generic Botzone keep-running bot that serves ANY arch config (for the architecture search).
# Env: CAIEST_MODEL=<path .pkl>, CAIEST_ARCH='{"channels":128,"blocks":16,"head":"flatten"}'
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from feature import FeatureAgent
from model_cfg import CfgCNN

def obs2response(model, agent, obs):
    logits = model({'is_training': False,
                    'obs': {'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
                            'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))}})
    return agent.action2response(int(logits.detach().numpy().flatten().argmax()))

if __name__ == '__main__':
    cfg = json.loads(os.environ.get('CAIEST_ARCH', '{"channels":128,"blocks":16,"head":"flatten"}'))
    model = CfgCNN(**cfg)
    model.load_state_dict(torch.load(os.environ['CAIEST_MODEL'], map_location='cpu'))
    model.eval()
    agent = None
    input()
    while True:
        request = input()
        while not request.strip(): request = input()
        t = request.split()
        if t[0] == '0':
            seatWind = int(t[1]); agent = FeatureAgent(seatWind)
            agent.request2obs('Wind %s' % t[2]); print('PASS')
        elif t[0] == '1':
            agent.request2obs(' '.join(['Deal', *t[5:]])); print('PASS')
        elif t[0] == '2':
            obs = agent.request2obs('Draw %s' % t[1]); r = obs2response(model, agent, obs).split()
            if r[0] == 'Hu': print('HU')
            elif r[0] == 'Play': print('PLAY %s' % r[1])
            elif r[0] == 'Gang': print('GANG %s' % r[1]); angang = r[1]
            elif r[0] == 'BuGang': print('BUGANG %s' % r[1])
        elif t[0] == '3':
            p = int(t[1])
            if t[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p); zimo = True; print('PASS')
            elif t[2] == 'GANG':
                if p == seatWind and angang: agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo: agent.request2obs('Player %d AnGang' % p)
                else: agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif t[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, t[3]))
                print('HU' if (p != seatWind and obs2response(model, agent, obs) == 'Hu') else 'PASS')
            else:
                zimo = False
                if t[2] == 'CHI': agent.request2obs('Player %d Chi %s' % (p, t[3]))
                elif t[2] == 'PENG': agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, t[-1]))
                if p == seatWind: print('PASS')
                else:
                    r = obs2response(model, agent, obs).split()
                    if r[0] == 'Hu': print('HU')
                    elif r[0] == 'Pass': print('PASS')
                    elif r[0] == 'Gang': print('GANG'); angang = None
                    elif r[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d ' % seatWind + ' '.join(r))
                        r2 = obs2response(model, agent, obs)
                        print(' '.join([r[0].upper(), *r[1:], r2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + ' '.join(r))
        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
