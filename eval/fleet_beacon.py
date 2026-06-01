"""
fleet_beacon.py — tiny HTTP beacon the central gpu-fleet dashboard (AWS node)
can POLL to learn what the mahjong project is running on the fleet and when a
run finishes.

Bind 0.0.0.0:1111 inside this vast container -> publicly reachable at
http://174.115.164.43:20474 (VAST_TCP_PORT_1111=20474).

Endpoints:
  /status        live JSON written by train/fleet_notify.py
  /assignments   a drop-in gpu-fleet assignments.json *fragment* tagging our
                 box IDs project=mahjong-pbt with the current state in the note,
                 so the AWS node can merge it into its assignments.json with a
                 1-line poll (see README banner printed on start).
  /              human-readable summary

Run:  python3 eval/fleet_beacon.py [PORT]   (default 1111)
"""
import json, os, sys, time, http.server, socketserver

STATE_FILE = "/tmp/fleet_state.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1111

def state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"project": "mahjong-pbt", "state": "unknown", "boxes": [], "events": []}

def assignments_fragment(s):
    note = f"mahjong-pbt {s.get('state','?')} {s.get('updated','')}"
    if s.get("champion"):
        note += f" champ={s['champion']}"
    return {str(b): {"project": "mahjong-pbt", "note": note} for b in s.get("boxes", [])}

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        s = state()
        if self.path.startswith("/status"):
            self._send(json.dumps(s, indent=2))
        elif self.path.startswith("/assignments"):
            self._send(json.dumps(assignments_fragment(s), indent=2))
        else:
            ev = "\n".join(f"  {e['ts']}  {e['event']}"
                           + (f"  champ={e['champion']}" if e.get('champion') else "")
                           + (f"  {e['detail']}" if e.get('detail') else "")
                           for e in s.get("events", [])[-12:])
            self._send(
                f"mahjong fleet beacon\nproject : {s.get('project')}\nstate   : {s.get('state')}\n"
                f"updated : {s.get('updated')}\nchampion: {s.get('champion')}\n"
                f"boxes   : {', '.join(str(b) for b in s.get('boxes', []))}\n\nrecent events:\n{ev}\n"
                f"\nendpoints: /status  /assignments\n", "text/plain")
    def log_message(self, *a): pass

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    pub = os.environ.get(f"VAST_TCP_PORT_{PORT}", "?")
    print(f"beacon on :{PORT}  -> public http://174.115.164.43:{pub}/status", flush=True)
    print("AWS-node 1-liner to merge our tags into the dashboard (cron every 1-2 min):", flush=True)
    print(f"""  python3 - <<'PY'
import json,urllib.request,os
f=os.path.expanduser('~/gpu-fleet/assignments.json'); d=json.load(open(f))
frag=json.load(urllib.request.urlopen('http://174.115.164.43:{pub}/assignments',timeout=10))
d.update(frag); json.dump(d,open(f,'w'),indent=2)
PY""", flush=True)
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as srv:
        srv.serve_forever()
