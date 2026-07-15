"""Ladder CLI (WO-P0-04).

    python -m arena.ladder.cli run        [--rounds 3 --pairs 16 --dir DIR]
    python -m arena.ladder.cli standings  [--dir DIR]
    python -m arena.ladder.cli head2head A B [--blocks 8 --pairs 16]
    python -m arena.ladder.cli audit MATCH_ID [--dir DIR]
    python -m arena.ladder.cli calibrate  [--set-reference | --unfreeze | --inject-bug] [--dir DIR]
    python -m arena.ladder.cli submit NAME PARAMS.msgpack
"""

from __future__ import annotations

import argparse
import json

from arena.ladder import agents, calibrate as cal, ladder as lad

DEFAULT_DIR = "/root/ludus_ladder"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--dir", default=DEFAULT_DIR)

    p = sub.add_parser("standings")
    p.add_argument("--dir", default=DEFAULT_DIR)

    p = sub.add_parser("head2head")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--pairs", type=int, default=16)

    p = sub.add_parser("audit")
    p.add_argument("match_id")
    p.add_argument("--dir", default=DEFAULT_DIR)

    p = sub.add_parser("calibrate")
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--set-reference", action="store_true")
    p.add_argument("--unfreeze", action="store_true")
    p.add_argument("--inject-bug", action="store_true")

    p = sub.add_parser("submit")
    p.add_argument("name")
    p.add_argument("params")

    args = ap.parse_args()

    if args.cmd == "run":
        L = lad.Ladder(args.dir)
        entrants = list(agents.REGISTRY)
        for r in range(args.rounds):
            L.run_round(entrants, args.pairs, r)
            print(f"round {r + 1}/{args.rounds} done", flush=True)
        print(json.dumps(L.standings(), indent=2))

    elif args.cmd == "standings":
        print(json.dumps(lad.Ladder(args.dir).standings(), indent=2))

    elif args.cmd == "head2head":
        print(json.dumps(lad.head2head(args.a, args.b, args.blocks, args.pairs),
                         indent=2))

    elif args.cmd == "audit":
        print(json.dumps(lad.audit(lad.Ladder(args.dir), args.match_id), indent=2))

    elif args.cmd == "calibrate":
        L = lad.Ladder(args.dir)
        if args.unfreeze:
            L.unfreeze()
            print("unfrozen")
            return
        if args.inject_bug:
            import tempfile
            with tempfile.TemporaryDirectory() as scratch:
                print(json.dumps(cal.inject_bug_drill(scratch), indent=2))
            return
        print(json.dumps(cal.calibrate(L, set_reference=args.set_reference), indent=2))

    elif args.cmd == "submit":
        h = agents.submit(args.name, args.params)
        print(json.dumps({"submitted": args.name, "hash": h}))


if __name__ == "__main__":
    main()
