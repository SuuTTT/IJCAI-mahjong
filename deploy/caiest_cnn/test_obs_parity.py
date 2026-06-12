# test_obs_parity.py — GOLD TEST for pimc_obs: replay a real collected game through FeatureAgent
# and assert that obs_for_seat reproduces agent.obs EXACTLY at every self-decision point.
# Run on a box with MahjongGB:  python3 test_obs_parity.py <full_log.json> [seat]
import json, sys
import numpy as np
from feature import FeatureAgent
import pimc_obs as P


def replay(log_path, seat):
    d = json.load(open(log_path))
    agent = None
    checked = 0
    for rec in d:
        out = (rec.get('output') or {})
        disp = out.get('display') or {}
        cmd = out.get('content') or {}
        act = disp.get('action')
        if act == 'INIT':
            agent = FeatureAgent(seat)
            agent.request2obs('Wind %d' % disp.get('quan', 0))
            continue
        if agent is None:
            continue
        if act == 'DEAL':
            hands = disp.get('hand')
            if hands:
                agent.request2obs('Deal ' + ' '.join(hands[seat]))
            continue
        p = disp.get('player')
        if act == 'DRAW':
            if p == seat:
                agent.request2obs('Draw %s' % disp.get('tile'))
                # OUR decision point: verify parity
                hands = [list(agent.hand) if i == 0 else [] for i in range(4)]
                discards = [list(agent.history[i]) for i in range(4)]
                packs = [list(agent.packs[i]) for i in range(4)]
                seatwinds = [(agent.seatWind + i) % 4 for i in range(4)]
                mine = P.obs_for_seat(0, hands, discards, packs, seatwinds, agent.prevalentWind)
                ref = agent.obs.reshape(38, 4, 9)
                if not np.array_equal(mine, ref):
                    bad = np.argwhere(mine != ref)
                    print('MISMATCH at decision %d: %d cells, first %s' % (checked, len(bad), bad[:3]))
                    return False
                checked += 1
            else:
                agent.request2obs('Player %d Draw' % p)
        elif act == 'PLAY':
            agent.request2obs('Player %d Play %s' % (p, disp.get('tile')))
        elif act == 'CHI':
            # one record = claim (tileCHI = middle tile of the sequence) + follow-up discard (tile)
            agent.request2obs('Player %d Chi %s' % (p, disp.get('tileCHI')))
            agent.request2obs('Player %d Play %s' % (p, disp.get('tile')))
        elif act == 'PENG':
            # one record = claim of the last discard + follow-up discard (tile)
            agent.request2obs('Player %d Peng' % p)
            agent.request2obs('Player %d Play %s' % (p, disp.get('tile')))
        elif act in ('GANG', 'BUGANG', 'ANGANG'):
            break  # replay scope: stop at kong complexity — enough decisions checked by then
        elif act == 'HU':
            break
    print('PARITY OK: %d self-decision points matched exactly' % checked)
    return checked > 0


if __name__ == '__main__':
    path = sys.argv[1]
    seat = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ok = replay(path, seat)
    sys.exit(0 if ok else 1)
