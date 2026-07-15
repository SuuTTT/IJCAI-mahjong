#!/usr/bin/env python3
"""E2 eval: play a ckpt policy (single model or mean-softmax ensemble) against
the fixed Stockfish node ladder (elo_ladder.run_ladder). Move choice = argmax
of the (ensemble-mean) softmax restricted to legal moves. Output JSON is
written atomically so the keeper's REQUIRE gates only see complete results.
"""
import argparse, json, os, sys, time
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from chess_enc import encode_board, index_to_move, legal_action_mask
from throughput_probe import resnet
from elo_ladder import run_ladder


class EnsPolicy:
    def __init__(self, torch, ckpts, name, dev):
        self.torch, self.dev, self.name = torch, dev, name
        self.nets = []
        for p in ckpts:
            ck = torch.load(p, map_location=dev)
            net = resnet(ck["channels"], ck["blocks"]).to(dev)
            net.load_state_dict(ck["state_dict"])
            net.eval()
            self.nets.append(net)

    def __call__(self, board):
        torch = self.torch
        x = torch.from_numpy(encode_board(board)[None]).float().to(self.dev)
        with torch.no_grad():
            probs = self.nets[0](x).softmax(1)
            for n in self.nets[1:]:
                probs = probs + n(x).softmax(1)
        p = (probs[0] / len(self.nets)).cpu().numpy()
        legal = np.asarray(legal_action_mask(board)).astype(bool)
        p = np.where(legal, p, -1.0)  # any legal move beats every masked one
        return index_to_move(int(p.argmax()), board)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", required=True)
    ap.add_argument("--name", required=True,
                    help="policy label, e.g. single/trioA/trioB/student")
    ap.add_argument("--ckpts", required=True, help="comma-separated ckpt paths")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--levels", default="16,64,256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpts = [p for p in a.ckpts.split(",") if p]
    pol = EnsPolicy(torch, ckpts, f"{a.band}/{a.name}", dev)
    levels = [int(x) for x in a.levels.split(",")]
    t0 = time.time()
    res = run_ladder(pol, levels=levels, n_games=a.games)
    out = {"band": a.band, "name": a.name, "ckpts": ckpts, "device": dev,
           "wall_s": round(time.time() - t0, 1), **res}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, a.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
