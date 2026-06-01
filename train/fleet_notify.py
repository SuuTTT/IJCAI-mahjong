"""
fleet_notify.py — publish mahjong-fleet job status so the central gpu-fleet
dashboard (AWS node 54.251.156.216) can see what we're running and get pinged
when a run finishes.

Two integration paths (we can reach the vast boxes but NOT the AWS node directly;
the AWS node CAN reach this box's public ports), so this writes a local status
file that `fleet_beacon.py` serves over HTTP for the AWS node to poll, AND
optionally PUSHES on state changes via whatever channel is configured:

  • FLEET_AWS_SSH   e.g. "ubuntu@54.251.156.216 -i ~/.ssh/aws_key"
                    -> updates the AWS node's ~/gpu-fleet/assignments.json
                       (tags our box IDs project=mahjong-pbt, note=state) and
                       appends a line to ~/gpu-fleet/notifications.log
  • FLEET_WEBHOOK   e.g. "http://54.251.156.216:5050/ingest" or a Slack URL
                    -> HTTP POST {event, project, boxes, champion, ts}

Both are optional; if neither is set we still publish the local beacon file.

CLI:
  python3 train/fleet_notify.py running  --boxes 36994217,38702735,... --note "PBT gen loop"
  python3 train/fleet_notify.py finished --champion gen2 --detail "gen2 +968 vs poolbig"
"""
import os, sys, json, time, subprocess, argparse, urllib.request

STATE_FILE = "/tmp/fleet_state.json"
PROJECT = "mahjong-pbt"

def _load():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"project": PROJECT, "events": []}

def publish(event, boxes=None, champion=None, note="", detail="", ts=None):
    """Update the local beacon state file. ts passed in (no Date.now in workflows
    context); CLI uses time.time()."""
    ts = ts or time.time()
    s = _load()
    s["project"] = PROJECT
    s["state"] = event                      # running | finished | failed
    s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    s["updated_epoch"] = ts
    if boxes is not None:
        s["boxes"] = boxes
    if champion is not None:
        s["champion"] = champion
    if note:
        s["note"] = note
    s.setdefault("events", []).append(
        {"event": event, "ts": s["updated"], "champion": champion, "detail": detail})
    s["events"] = s["events"][-50:]
    tmp = STATE_FILE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE_FILE)
    return s

def push_ssh(s, detail):
    """If FLEET_AWS_SSH set: tag our boxes in the AWS assignments.json + log it."""
    target = os.environ.get("FLEET_AWS_SSH", "").strip()
    if not target:
        return "ssh: not configured"
    boxes = s.get("boxes", [])
    note = f"{PROJECT} {s['state']} {s['updated']}" + (f" champ={s.get('champion')}" if s.get('champion') else "")
    # remote python: merge our box IDs into assignments.json, append notifications.log
    pyc = (
        "import json,os,sys,time;"
        "f=os.path.expanduser('~/gpu-fleet/assignments.json');"
        "d=json.load(open(f)) if os.path.exists(f) else {};"
        f"ids={json.dumps([str(b) for b in boxes])};note={json.dumps(note)};"
        "[d.__setitem__(i,{'project':'%s','note':note}) for i in ids];" % PROJECT +
        "json.dump(d,open(f,'w'),indent=2);"
        "open(os.path.expanduser('~/gpu-fleet/notifications.log'),'a').write(time.strftime('%Y-%m-%d %H:%M:%S')+' "
        + s["state"] + " " + detail.replace("'", "") + "\\n')"
    )
    parts = target.split()
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=12"] + parts[1:] + [parts[0], f"python3 -c \"{pyc}\""]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        return "ssh: ok" if r.returncode == 0 else f"ssh: rc={r.returncode} {r.stderr[-120:]}"
    except Exception as e:
        return f"ssh: {e}"

def push_webhook(s, detail):
    url = os.environ.get("FLEET_WEBHOOK", "").strip()
    if not url:
        return "webhook: not configured"
    payload = json.dumps({"event": s["state"], "project": PROJECT,
                          "boxes": s.get("boxes", []), "champion": s.get("champion"),
                          "detail": detail, "ts": s["updated"]}).encode()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return f"webhook: {r.status}"
    except Exception as e:
        return f"webhook: {e}"

def notify(event, boxes=None, champion=None, note="", detail=""):
    s = publish(event, boxes=boxes, champion=champion, note=note, detail=detail)
    res = [push_ssh(s, detail), push_webhook(s, detail)]
    return s, res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event", choices=["running", "finished", "failed"])
    ap.add_argument("--boxes", default="")
    ap.add_argument("--champion", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--detail", default="")
    a = ap.parse_args()
    boxes = [b for b in a.boxes.split(",") if b] or None
    s, res = notify(a.event, boxes=boxes, champion=a.champion, note=a.note, detail=a.detail)
    print(f"published state={s['state']} -> {STATE_FILE}")
    for r in res:
        print("  " + r)
