"""
fleet_monitor.py — non-invasive sidecar for the league_v3 run.

Does two things, on a ~40s loop, until the run finishes:
  1. Reports the LIVE run to the central gpu-fleet dashboard's "Projects" panel
     (project="mahjong-league") via the general client, so the panel stops showing
     only the stale old run. Uses FLEET_CENTER_SSH transport (port 22 only).
  2. Builds an SGT-timestamped, per-instance detail log of what each box is doing:
       /tmp/fleet_instances.json  (machine-readable, the dashboard reads this)
       /tmp/fleet_instances.log   (human tail -f, one block per cycle)
     During the "train" phase it SSH-tails each box's current ppo log (reusing the
     orchestrator's ControlMaster connection, so no extra vast.ai SSH churn) to show
     the live iteration / win% / draw%. Otherwise it reports the phase-derived state.

All times are Singapore time (SGT = UTC+8); the box itself runs in UTC.

Run:  FLEET_CENTER_SSH="ubuntu@54.251.156.216 -i ~/.ssh/aws_fleet_ed25519" \
      FLEET_TOKEN=... FLEET_INGEST_PORT=5056 python3 train/fleet_monitor.py
"""
import os, sys, json, time, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, "/home/coder/gpu-fleet")          # gpu_fleet.client
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from train.pbt_loop import ssh
try:
    from gpu_fleet.client import report
except Exception:
    def report(*a, **k): return "client unavailable"

STATUS    = "/tmp/pbt_status.json"
PROBE     = "/tmp/fleet_probe.json"
INST_JSON = "/tmp/fleet_instances.json"
INST_LOG  = "/tmp/fleet_instances.log"
PROJECT   = "mahjong-league"
SGT_OFF   = 8 * 3600

def sgt(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime((t or time.time()) + SGT_OFF))

def load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default

def box_endpoints():
    return {b["id"]: (b["ip"], b["port"]) for b in load(PROBE, [])}

def tail_log(box, rnd):
    """Best-effort: last ppo line for this box's current round."""
    cmd = (f"tail -n 1 /root/mj/r{rnd}_{box['name']}.log 2>/dev/null")
    rc, out, _ = ssh(box, cmd, timeout=18)
    line = (out or "").strip().splitlines()[-1] if out.strip() else ""
    return line[:160]

def main():
    eps = box_endpoints()
    started = False
    last_phase = None
    while True:
        s = load(STATUS, {})
        phase = s.get("phase"); rnd = s.get("gen"); running = s.get("running", True)
        members = s.get("members", [])
        champion = s.get("champion"); pool_size = s.get("pool_size")
        hist = s.get("history", [])
        n_prom = sum(1 for h in hist if h.get("promoted"))
        boxes_ids = [m["box"] for m in members]

        # ── 1. central Projects-panel report ──  (ingest sets state = event verbatim)
        ev = "done" if phase == "finished" else "running"
        detail = (f"round {rnd} · {phase} · champ {champion} · {n_prom} promotions · "
                  f"{len(members)} boxes / {s.get('total_quota')} cores")
        try:
            report(ev, project=PROJECT, boxes=boxes_ids, detail=detail,
                   note="league_v3 (gen2-anchor, 700-game confirm + exploiters)")
            started = True
        except Exception:
            pass

        # ── 2. per-instance SGT detail ──
        do_tail = (phase == "train")
        live = {}
        if do_tail:
            def grab(m):
                ip_port = eps.get(m["box"])
                if not ip_port:
                    live[m["box"]] = "(no endpoint)"; return
                box = {"id": m["box"], "ip": ip_port[0], "port": ip_port[1], "name": m["name"]}
                live[m["box"]] = tail_log(box, rnd) or "(starting…)"
            ths = [threading.Thread(target=grab, args=(m,)) for m in members]
            for t in ths: t.start()
            for t in ths: t.join()

        now = sgt()
        rows = []
        for m in members:
            c = m.get("config") or {}
            cfg = (f"lr={c.get('lr')} shape={c.get('shape')} ent={c.get('ent')}"
                   if c else "")
            if do_tail:
                act = live.get(m["box"], "")
            elif phase in ("rank", "confirm"):
                act = f"idle — {phase} tournament running on local box"
            elif phase == "done":
                net = m.get("net"); act = f"round {rnd} done" + (f", net={net:+d}" if net is not None else "")
            elif phase == "finished":
                act = "run finished"
            else:
                act = phase or ""
            rows.append({"box": m["box"], "name": m["name"], "role": m.get("role"),
                         "quota": m.get("quota"), "cfg": cfg, "net": m.get("net"),
                         "activity": act, "sgt": now})
        snap = {"sgt": now, "phase": phase, "round": rnd, "champion": champion,
                "promotions": n_prom, "pool_size": pool_size, "running": running,
                "instances": rows}
        tmp = INST_JSON + ".tmp"; json.dump(snap, open(tmp, "w"), indent=2); os.replace(tmp, INST_JSON)

        with open(INST_LOG, "a") as f:
            f.write(f"\n===== {now} SGT · round {rnd} · phase {phase} · "
                    f"champ {champion} · {n_prom} promotions =====\n")
            for r in rows:
                net = "" if r["net"] is None else f" net={r['net']:+d}"
                f.write(f"  [{r['box']} {r['name']:<9} {str(r['role']):<7} q{r['quota']}] "
                        f"{r['activity']}{net}\n")

        if phase == "finished" or not running:
            report("done", project=PROJECT, boxes=boxes_ids,
                   detail=f"finished · {rnd} rounds · {n_prom} promotions · champ {champion}",
                   note="league_v3 done")
            break
        # also stop if the status file has gone stale for >25 min (run died)
        time.sleep(40)

if __name__ == "__main__":
    main()
