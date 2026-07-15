"""Generational self-play league (the WO-P0-02 'league-lite', productionized).

Evolutionary hill-climb with CI gates:
  gen k trains by self-play, INITIALIZED from the reigning champion's weights;
  the candidate is then evaluated against the champion in seat-swapped mirrored
  pairs; it is PROMOTED only if its pair-score Wilson lower bound clears 0.5.
Every generation also gets an absolute reading vs the fixed anchors (random_v0,
rule_v0), so the lineage shows both relative and absolute progress.

Artifacts under --dir (default /root/ludus_train/league):
  gen_<k>.msgpack            every candidate
  champion.msgpack           current champion (served live by the play server)
  league.jsonl               one line per event (train/eval/promote), provenance
  standings.json             lineage summary, refreshed each generation

Run:  python -m baselines.league --hours 8.5
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def log_line(f, obj):
    obj["t"] = time.time()
    f.write(json.dumps(obj) + "\n")
    f.flush()
    print(json.dumps(obj), flush=True)


def eval_pair(a_path: Path, b_spec: str, matches: int, out: Path) -> dict:
    r = sh([sys.executable, "-m", "baselines.eval_pair",
            "--a", f"ppo:{a_path}", "--b", b_spec,
            "--matches", str(matches), "--decks", "mirror", "--out", str(out)], cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"eval failed: {r.stderr[-400:]}")
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.5)
    ap.add_argument("--dir", default="/root/ludus_train/league")
    ap.add_argument("--updates-per-gen", type=int, default=6000)
    ap.add_argument("--gate-matches", type=int, default=256)
    ap.add_argument("--anchor-matches", type=int, default=128)
    ap.add_argument("--num-envs", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    lf = open(root / "league.jsonl", "a")
    deadline = time.time() + args.hours * 3600
    champion = root / "champion.msgpack"

    # resume an interrupted league if a champion already exists
    lineage: list[dict] = []
    if (root / "standings.json").exists():
        lineage = json.loads((root / "standings.json").read_text()).get("lineage", [])
    gen = (lineage[-1]["gen"] + 1) if lineage else 0

    log_line(lf, {"event": "league_start", "hours": args.hours, "gen0": gen,
                  "resumed": champion.exists()})

    while time.time() < deadline:
        gen_dir = root / f"train_gen{gen}"
        cand = root / f"gen_{gen}.msgpack"
        t0 = time.time()

        # plateau-breaker: every 3rd challenger explores from a fresh init
        # instead of resuming the champion (gens 5-21 all failed to dethrone
        # gen 4 by cloning it — exploitation alone converged)
        explorer = champion.exists() and gen % 3 == 2
        train_cmd = [sys.executable, "baselines/ppo_selfplay.py",
                     "--decks", "mirror", "--updates", str(args.updates_per_gen),
                     "--num-envs", str(args.num_envs), "--rollout-t", "32",
                     "--eval-every", str(args.updates_per_gen),
                     "--pool-dir", str(root), "--out", str(gen_dir)]
        if champion.exists() and not explorer:
            train_cmd += ["--resume", str(champion)]
        import os
        env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
        r = sh(train_cmd, cwd=REPO, env=env)
        src = gen_dir / "params_latest.msgpack"
        if r.returncode != 0 or not src.exists():
            log_line(lf, {"event": "train_failed", "gen": gen,
                          "stderr": r.stderr[-400:]})
            time.sleep(30)
            continue
        shutil.copy(src, cand)
        train_min = round((time.time() - t0) / 60, 1)

        entry = {"gen": gen, "train_minutes": train_min,
                 "type": "explorer" if explorer else "resume"}
        if not champion.exists():                      # bootstrap
            shutil.copy(cand, champion)
            entry.update({"promoted": True, "bootstrap": True})
            log_line(lf, {"event": "promote", "gen": gen, "bootstrap": True})
        else:
            gate = eval_pair(cand, f"ppo:{champion}", args.gate_matches,
                             root / f"gate_gen{gen}.json")
            promoted = gate["a_win_ci95_lower"] > 0.5
            entry.update({"promoted": promoted,
                          "vs_champion": round(gate["a_win_rate"], 4),
                          "vs_champion_ci": round(gate["a_win_ci95_lower"], 4)})
            log_line(lf, {"event": "gate", "gen": gen, **entry})
            if promoted:
                shutil.copy(cand, champion)
                log_line(lf, {"event": "promote", "gen": gen})

        # absolute anchors for the lineage curve
        for spec, name in (("random", "vs_random"), ("rule", "vs_rule")):
            try:
                res = eval_pair(cand, spec, args.anchor_matches,
                                root / f"anchor_{name}_gen{gen}.json")
                entry[name] = round(res["a_win_rate"], 4)
            except RuntimeError as ex:
                entry[name] = None
                log_line(lf, {"event": "anchor_failed", "gen": gen, "err": str(ex)[-200:]})

        lineage.append(entry)
        (root / "standings.json").write_text(json.dumps({
            "champion_gen": max((e["gen"] for e in lineage if e.get("promoted")),
                                default=0),
            "generations": len(lineage), "lineage": lineage}, indent=2))
        log_line(lf, {"event": "gen_done", **entry})
        shutil.rmtree(gen_dir, ignore_errors=True)     # keep disk bounded
        gen += 1

    log_line(lf, {"event": "league_deadline_reached", "generations": len(lineage)})


if __name__ == "__main__":
    main()
