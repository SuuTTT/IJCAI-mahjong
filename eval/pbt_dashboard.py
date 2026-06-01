"""
pbt_dashboard.py — live dashboard for the PBT/league loop (train/pbt_loop.py).

Reads /tmp/pbt_status.json (written every phase) and renders a self-refreshing
page: current generation/phase, the champion + deploy file, the population table
with per-member hyperparameters and latest tournament net, the champion-vs-frozen-
anchor curve across generations (absolute-progress check), pool contents, per-box
state, and the recent orchestrator log.

Run:  python3 eval/pbt_dashboard.py [PORT]   (default 8082)
Open: http://localhost:8082
"""
import json, os, sys, html, http.server, socketserver

STATUS = "/tmp/pbt_status.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8082


def curve_svg(history, w=620, h=180):
    if not history:
        return "<i>no generations yet</i>"
    gens = [hh["gen"] for hh in history]
    champ = [hh["champion_net"] for hh in history]
    anch = [hh.get("anchor_net", 0) for hh in history]
    allv = champ + anch
    lo, hi = min(allv), max(allv)
    if hi == lo: hi = lo + 1
    pad = 28
    def px(g): return pad + (g - gens[0]) / max(1, gens[-1] - gens[0]) * (w - 2 * pad)
    def py(v): return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    def poly(vals, color):
        pts = " ".join(f"{px(g):.1f},{py(v):.1f}" for g, v in zip(gens, vals))
        dots = "".join(f'<circle cx="{px(g):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>'
                       for g, v in zip(gens, vals))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>{dots}'
    zero = f'<line x1="{pad}" y1="{py(0):.1f}" x2="{w-pad}" y2="{py(0):.1f}" stroke="#445" stroke-dasharray="4"/>' if lo < 0 < hi else ""
    return f'''<svg width="{w}" height="{h}" style="background:#0b1020;border-radius:8px">
      {zero}
      {poly(champ, "#5cf")}
      {poly(anch, "#fd7")}
      <text x="8" y="16" fill="#5cf" font-size="12">champion net (vs population)</text>
      <text x="8" y="32" fill="#fd7" font-size="12">champion vs frozen poolbig anchor (absolute)</text>
      <text x="{w-pad}" y="{h-8}" fill="#8ab" font-size="11" text-anchor="end">gen {gens[-1]}</text>
    </svg>'''


def page():
    try:
        s = json.load(open(STATUS))
    except Exception:
        return "<!doctype html><meta http-equiv=refresh content=5><body style='font-family:system-ui;background:#0f1226;color:#dde;padding:24px'>PBT status not available yet — waiting for the loop to start…</body>"
    run = s.get("running")
    tag = ('<span style="background:#1d6;padding:2px 8px;border-radius:6px">RUNNING</span>'
           if run else '<span style="background:#a55;padding:2px 8px;border-radius:6px">STOPPED</span>')
    P = ['<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=8>',
         '<title>PBT league dashboard</title>',
         '<style>body{font-family:system-ui;margin:22px;background:#0f1226;color:#dde}'
         'h1{font-size:21px}h2{font-size:15px;color:#9bd;margin-top:22px}'
         'table{border-collapse:collapse;font-size:13px;margin-top:6px}'
         'td,th{border:1px solid #2a3050;padding:4px 9px;text-align:left}'
         'th{background:#161c38;color:#9bd}tr:nth-child(even){background:#12172e}'
         'code{color:#8fd}.big{font-size:18px;color:#7f7}'
         'pre{background:#0b1020;padding:10px;border-radius:8px;font-size:11px;color:#bcd;max-height:240px;overflow:auto}</style>']
    P.append(f"<h1>IJCAI Mahjong — PBT / league loop {tag}</h1>")
    P.append(f"<div>generation <b>{s.get('gen')}</b> · phase <b>{html.escape(str(s.get('phase')))}</b>"
             f" · {s.get('n_boxes')} boxes · {s.get('total_quota')} real cores · updated {s.get('updated')}</div>")
    champ = s.get("champion"); cnet = s.get("champion_net")
    if champ:
        P.append(f"<div class=big>★ champion: {html.escape(str(champ))} (net {cnet:+}) "
                 f"→ deploy <code>train/checkpoints/pbt_champion_fp16.npz</code></div>")

    P.append("<h2>Champion progress</h2>")
    P.append(curve_svg(s.get("history", [])))

    P.append("<h2>Population</h2><table><tr><th>member</th><th>box</th><th>quota</th>"
             "<th>lr</th><th>shape</th><th>ent</th><th>add_every</th><th>last net</th></tr>")
    rank = {}
    if s.get("history"):
        for i, (nm, _) in enumerate(s["history"][-1]["ranking"]):
            rank[nm] = i
    for m in sorted(s.get("members", []), key=lambda m: rank.get(m["name"], 99)):
        c = m["config"]; net = m.get("net")
        medal = "🥇" if rank.get(m["name"]) == 0 else ("🥈" if rank.get(m["name"]) == 1 else "")
        P.append(f"<tr><td>{medal} {m['name']}</td><td>{m['box']}</td><td>{m['quota']}</td>"
                 f"<td>{c['lr']:.1e}</td><td>{c['shape']}</td><td>{c['ent']}</td>"
                 f"<td>{c['add_every']}</td><td>{'' if net is None else f'{net:+}'}</td></tr>")
    P.append("</table>")

    hist = s.get("history", [])
    if hist:
        P.append("<h2>Generation results</h2><table><tr><th>gen</th><th>champion</th>"
                 "<th>champ net</th><th>vs anchor</th><th>ranking (net)</th></tr>")
        for hh in reversed(hist):
            rk = "  ".join(f"{n}:{v:+}" for n, v in hh["ranking"] if n != "_anchor")
            margin = hh["champion_net"] - hh.get("anchor_net", 0)
            P.append(f"<tr><td>{hh['gen']}</td><td>{hh['champion']}</td>"
                     f"<td>{hh['champion_net']:+}</td><td>{margin:+}</td>"
                     f"<td><code>{html.escape(rk)}</code></td></tr>")
        P.append("</table>")

    P.append(f"<h2>Opponent pool ({s.get('pool_size')})</h2>")
    P.append("<pre>" + html.escape("\n".join(s.get("pool", []))) + "</pre>")

    P.append("<h2>Orchestrator log</h2>")
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
        print(f"PBT dashboard at http://localhost:{PORT}")
        srv.serve_forever()
