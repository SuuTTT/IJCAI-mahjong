"""
pbt_dashboard.py — live dashboard for the fleet league loop (train/league_v3.py).

Reads two files and renders a self-refreshing page (all times Singapore / SGT):
  /tmp/pbt_status.json        — written by the orchestrator each phase
  /tmp/fleet_instances.json   — written by train/fleet_monitor.py (per-instance,
                                 SGT-timestamped live activity for every box)

Sections: run header (round/phase/champion/deploy file), champion-vs-gen2 margin
curve, the per-INSTANCE table (what each box is doing right now, SGT), round
history (promotions + 700-game confirm nets), opponent pool, orchestrator log.

Run:  python3 eval/pbt_dashboard.py [PORT]   (default 8082)
"""
import json, sys, html, http.server, socketserver

STATUS    = "/tmp/pbt_status.json"
INST_JSON = "/tmp/fleet_instances.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8082


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def curve_svg(history, w=640, h=180):
    pts = [(hh.get("round"), hh.get("vs_gen2")) for hh in history
           if hh.get("round") is not None and hh.get("vs_gen2") is not None]
    if not pts:
        return "<i>no rounds yet — the first 700-game confirm decides round 1</i>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lo, hi = min(ys + [0]), max(ys + [0])
    if hi == lo: hi = lo + 1
    pad = 30
    def px(x): return pad + (x - xs[0]) / max(1, xs[-1] - xs[0]) * (w - 2 * pad)
    def py(v): return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    line = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
    dots = "".join(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="3" '
                   f'fill="{"#5f5" if v>0 else "#f77"}"/>' for x, v in pts)
    zero = f'<line x1="{pad}" y1="{py(0):.1f}" x2="{w-pad}" y2="{py(0):.1f}" stroke="#556" stroke-dasharray="4"/>'
    return f'''<svg width="{w}" height="{h}" style="background:#0b1020;border-radius:8px">
      {zero}<polyline fill="none" stroke="#7af" stroke-width="2" points="{line}"/>{dots}
      <text x="8" y="16" fill="#7af" font-size="12">best candidate margin vs held-out gen2 (700-game confirm)</text>
      <text x="8" y="32" fill="#8ab" font-size="11">above 0 (green) = a real, out-of-pool improvement -> promoted</text>
      <text x="{w-pad}" y="{h-8}" fill="#8ab" font-size="11" text-anchor="end">round {xs[-1]}</text>
    </svg>'''


def page():
    s = load(STATUS); inst = load(INST_JSON)
    if not s:
        return ("<!doctype html><meta http-equiv=refresh content=5>"
                "<body style='font-family:system-ui;background:#0f1226;color:#dde;padding:24px'>"
                "league status not available yet — waiting for the loop…</body>")
    running = s.get("running", True)
    tag = ('<span style="background:#1d6;color:#012;padding:2px 9px;border-radius:6px">RUNNING</span>'
           if running else '<span style="background:#a55;padding:2px 9px;border-radius:6px">STOPPED</span>')
    hist = s.get("history", [])
    n_prom = sum(1 for h in hist if h.get("promoted"))
    sgt = inst.get("sgt", "—")
    P = ['<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=8>',
         '<title>Mahjong fleet league</title>',
         '<style>body{font-family:system-ui;margin:22px;background:#0f1226;color:#dde}'
         'h1{font-size:21px}h2{font-size:15px;color:#9bd;margin-top:22px}'
         'table{border-collapse:collapse;font-size:13px;margin-top:6px;width:100%}'
         'td,th{border:1px solid #2a3050;padding:4px 9px;text-align:left;white-space:nowrap}'
         'th{background:#161c38;color:#9bd}tr:nth-child(even){background:#12172e}'
         'code{color:#8fd}.big{font-size:18px;color:#7f7;margin-top:8px}'
         '.exp{color:#fc6}.main{color:#9cf}.act{white-space:normal;color:#cde;font-family:monospace;font-size:12px}'
         'pre{background:#0b1020;padding:10px;border-radius:8px;font-size:11px;color:#bcd;max-height:240px;overflow:auto}</style>']
    P.append(f"<h1>IJCAI Mahjong — fleet league {tag}</h1>")
    P.append(f"<div>round <b>{s.get('gen')}</b> · phase <b>{html.escape(str(s.get('phase')))}</b> · "
             f"{s.get('n_boxes')} boxes · {s.get('total_quota')} cores quota · "
             f"<b>{n_prom}</b> promotions · <b>{html.escape(str(sgt))} SGT</b></div>")
    champ = s.get("champion")
    if champ:
        P.append(f"<div class=big>★ champion: {html.escape(str(champ))} → deploy "
                 f"<code>train/checkpoints/pbt_champion_fp16.npz</code> "
                 f"(seed = gen2, the held-out anchor)</div>")

    P.append("<h2>Champion progress (vs held-out gen2)</h2>")
    P.append(curve_svg(hist))

    # ── per-instance table (SGT live activity) ──
    P.append(f"<h2>What every instance is doing &middot; SGT {html.escape(str(sgt))}</h2>")
    rows = inst.get("instances")
    if rows:
        P.append("<table><tr><th>box</th><th>member</th><th>role</th><th>quota</th>"
                 "<th>config</th><th>live activity</th><th>last net</th></tr>")
        for r in rows:
            cls = "exp" if r.get("role") == "exploit" else "main"
            net = "" if r.get("net") is None else f"{r['net']:+d}"
            P.append(f"<tr><td>{r['box']}</td><td class={cls}>{html.escape(str(r['name']))}</td>"
                     f"<td class={cls}>{r.get('role')}</td><td>{r.get('quota')}</td>"
                     f"<td><code>{html.escape(str(r.get('cfg','')))}</code></td>"
                     f"<td class=act>{html.escape(str(r.get('activity','')))}</td><td>{net}</td></tr>")
        P.append("</table>")
        P.append("<div style='font-size:11px;color:#8ab'>roles: "
                 "<span class=main>main</span> = train champion vs diverse pool · "
                 "<span class=exp>exploit</span> = attack the champion to find weaknesses</div>")
    else:
        P.append("<i>per-instance monitor not running yet (start train/fleet_monitor.py)</i>")

    # ── round history ──
    if hist:
        P.append("<h2>Round history (700-game confirm)</h2><table>"
                 "<tr><th>round</th><th>result</th><th>best candidate</th><th>role</th>"
                 "<th>vs gen2</th><th>confirm nets</th></tr>")
        for h in reversed(hist):
            cn = h.get("confirm_net", {})
            cn_s = "  ".join(f"{k}:{v:+}" for k, v in cn.items())
            res = ("<b style='color:#5f5'>PROMOTED</b>" if h.get("promoted")
                   else "<span style='color:#9ab'>kept</span>")
            P.append(f"<tr><td>{h.get('round')}</td><td>{res}</td>"
                     f"<td>{html.escape(str(h.get('best_cand')))}</td><td>{h.get('role')}</td>"
                     f"<td>{h.get('vs_gen2'):+}</td><td><code>{html.escape(cn_s)}</code></td></tr>")
        P.append("</table>")

    P.append(f"<h2>Opponent pool ({s.get('pool_size')})</h2>")
    P.append("<pre>" + html.escape("\n".join(s.get("pool", []))) + "</pre>")
    P.append("<h2>Orchestrator log <span style='font-size:11px;color:#8ab'>(timestamps UTC)</span></h2>")
    P.append("<pre>" + html.escape("\n".join(s.get("log", []))) + "</pre>")
    return "".join(P)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as srv:
        print(f"league dashboard at http://localhost:{PORT}")
        srv.serve_forever()
