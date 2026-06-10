"""
rl_dashboard.py — live dashboard for the RL-league fine-tune (rl_league.py) + the SL→distill→RL
pipeline & strategy. Reads /tmp/rl_status.json (written by rl_pull.py, which polls ssh8's
rl_league.log) and renders a self-refreshing dark page with training curves, pool composition,
the gauntlet leaderboard, and the strategy narrative.

Run:  python3 eval/rl_dashboard.py [PORT]   (default 8090)
"""
import json, sys, html, http.server, socketserver, time

STATUS = "/tmp/rl_status.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090


def load():
    try:
        return json.load(open(STATUS))
    except Exception:
        return {}


def curve(its, key, color, label, w=560, h=170, pct=False):
    pts = [(d["it"], d[key]) for d in its if key in d and d[key] is not None]
    if not pts:
        return f"<div class=ph>{label}: waiting for data…</div>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    if hi == lo: hi = lo + (abs(lo) or 1) * 0.1 + 1e-6
    pad = 34
    def px(x): return pad + (x - xs[0]) / max(1, xs[-1] - xs[0]) * (w - 2 * pad)
    def py(v): return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
    line = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
    last = ys[-1]
    zero = ""
    if lo < 0 < hi:
        zy = py(0); zero = f'<line x1="{pad}" y1="{zy:.1f}" x2="{w-pad}" y2="{zy:.1f}" stroke="#334" stroke-dasharray="4 4"/>'
    fmt = (lambda v: f"{v:.2f}") if not pct else (lambda v: f"{v*100:.0f}%")
    return f'''<div class=card><div class=ct>{label} <span style="color:{color}">●</span> <b>{fmt(last)}</b></div>
<svg width="{w}" height="{h}" style="background:#0b1020;border-radius:8px">
{zero}<polyline fill="none" stroke="{color}" stroke-width="2" points="{line}"/>
<text x="{pad}" y="14" fill="#8a93a6" font-size="11">{fmt(hi)}</text>
<text x="{pad}" y="{h-6}" fill="#8a93a6" font-size="11">it {xs[0]}</text>
<text x="{w-pad-30}" y="{h-6}" fill="#8a93a6" font-size="11">it {xs[-1]}</text></svg></div>'''


PIPELINE = """
<h2>Training pipeline &amp; strategy</h2>
<div class=grid3>
<div class=stage><div class=sh>1 · Supervised base</div>
<b>distill100b</b> — 40-block BN-ResNet, behavioral-cloning the official + top-player records,
fused (BN-folded) + torch-1.4 legacy serialization for the Botzone deploy box (≤512MB, ≤6s/turn).
This is the proven floor and the current submission.</div>
<div class=stage><div class=sh>2 · Coherent distillation</div>
Single strong teacher (<b>chunjiandu</b>, rank-3, the richest clean source: 1.7k v10 games) ×
12× suit+reflection augmentation, KL-leashed to the frozen base. <i>Coherence &gt; diversity</i>:
a clean single teacher beat the 30-player mix. Date-filtered to current bot versions (ObjectId ts)
to avoid stale-version pollution.</div>
<div class=stage><div class=sh>3 · RL league fine-tune <span class=live>● LIVE</span></div>
PPO from the SL base, <b>KL-leashed</b> so it can't drift off the strong policy, trained vs a
<b>diverse PFSP pool</b>: the frozen SL anchor + 12 real-top-30 imitations + self-snapshots,
oversampling whoever the main currently loses to. The diverse pool is the ingredient our prior
self-play-only RL lacked (it overfit a narrow meta and didn't convert).</div>
</div>
<div class=note><b>Decision rule:</b> the gauntlet (official judge, 2v2 rotated vs 6 strong imitations)
is the play verdict. <b>distill100b stays the locked submission</b>; a candidate is adopted only if it
beats it on the gauntlet <i>and</i> ideally a few real Botzone-ladder games. Agreement-with-teacher is
only a weak proxy — this project has repeatedly seen “agreement ≠ play”.</div>
"""


def page():
    s = load()
    its = s.get("iters", [])
    cur = its[-1]["it"] if its else 0
    tot = s.get("total_iters", "?")
    sec = its[-1].get("sec") if its else None
    eta = ""
    if its and isinstance(tot, int) and sec:
        rem = (tot - cur) * sec
        eta = f" · ETA ~{rem//3600}h{(rem%3600)//60}m"
    P = ['<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=15>',
         '<title>IJCAI-Mahjong · RL league</title>',
         '''<style>
body{background:#070a12;color:#cdd3e0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:1180px}
h1{font-size:22px;margin:0 0 2px} h2{font-size:16px;color:#9fb4ff;border-bottom:1px solid #1c2438;padding-bottom:6px;margin:26px 0 12px}
.sub{color:#7e8aa3;margin-bottom:14px} .row{display:flex;flex-wrap:wrap;gap:16px}
.card{background:#0e1422;border:1px solid #1c2438;border-radius:10px;padding:12px}
.ct{font-size:13px;color:#9aa6bd;margin-bottom:6px} .ph{color:#667;padding:30px;font-style:italic}
.kpi{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 4px}
.kpi div{background:#0e1422;border:1px solid #1c2438;border-radius:10px;padding:10px 16px}
.kpi b{font-size:20px;color:#fff;display:block}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px} .stage{background:#0e1422;border:1px solid #1c2438;border-radius:10px;padding:14px}
.sh{color:#9fb4ff;font-weight:600;margin-bottom:6px} .live{color:#2ecc71;font-size:11px;animation:b 1.4s infinite} @keyframes b{50%{opacity:.3}}
.note{background:#11192b;border-left:3px solid #9fb4ff;padding:10px 14px;border-radius:6px;margin-top:14px;color:#aab4c9}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{padding:6px 10px;border-bottom:1px solid #1c2438;text-align:left}
th{color:#7e8aa3;font-weight:500} .pos{color:#2ecc71} .neg{color:#e74c3c} .best{background:#10241a}
code{background:#0e1422;padding:1px 5px;border-radius:4px;color:#9fb4ff}
@media(max-width:820px){.grid3{grid-template-columns:1fr}}
</style>''']
    P.append(f"<h1>IJCAI Chinese-Standard-Mahjong · training control</h1>")
    P.append(f"<div class=sub>updated {html.escape(str(s.get('updated','—')))} · deadline 2026-06-14 23:55</div>")
    # KPIs
    P.append("<div class=kpi>")
    P.append(f"<div>RL iter<b>{cur}/{tot}{eta}</b></div>")
    P.append(f"<div>base<b>{html.escape(str(s.get('base','—')))}</b></div>")
    P.append(f"<div>pool opponents<b>{s.get('anchors','—')} + {s.get('snapshots',0)} snaps</b></div>")
    if its:
        P.append(f"<div>main reward<b>{its[-1].get('main_r',0):+.2f}</b></div>")
        P.append(f"<div>KL→SL<b>{its[-1].get('kl',0):.2f}</b></div>")
    P.append("</div>")
    # curves
    P.append("<h2>RL league — live training</h2>")
    if not its:
        P.append("<div class=ph>waiting for the first iteration to land in rl_league.log…</div>")
    else:
        P.append("<div class=row>")
        P.append(curve(its, "main_r", "#2ecc71", "main reward (vs shifting pool)"))
        P.append(curve(its, "kl", "#f1c40f", "KL → SL base (leash)"))
        P.append(curve(its, "exp_wr", "#e67e22", "exploiter win-rate vs main", pct=True))
        P.append("</div>")
    # VALIDATION curve — fixed external bar (net vs the 6 top-30 imitations, internal sim)
    val = s.get("val", [])
    P.append("<h2>Validation — net vs 6 top-30 imitations (FIXED bar, internal sim, every ~4min)</h2>")
    if not val:
        P.append("<div class=ph>first validation eval runs ~1min after launch…</div>")
    else:
        P.append("<div class=row>")
        P.append(curve(val, "net", "#5dade2", "validation net — current policy vs 6 top-30 (36 games)", w=900, h=200))
        P.append("</div>")
        P.append("<div class=sub><i>rising = the RL policy is genuinely getting stronger vs a fixed bar "
                 "(main reward can't show this — its pool keeps shifting).</i></div>")
    # gauntlet leaderboard
    g = s.get("gauntlet", [])
    if g:
        P.append("<h2>Gauntlet leaderboard — play verdict (net over 72 games vs 6 strong imitations)</h2>")
        P.append("<table><tr><th>candidate</th><th>net</th><th>stuck</th><th>teacher / method</th></tr>")
        gg = sorted(g, key=lambda x: -x.get("net", -9999))
        for r in gg:
            cls = "best" if r is gg[0] else ""
            nc = "pos" if r.get("net", 0) >= 0 else "neg"
            P.append(f"<tr class={cls}><td>{html.escape(str(r['name']))}</td><td class={nc}>{r.get('net'):+d}</td>"
                     f"<td>{r.get('stuck','')}</td><td>{html.escape(str(r.get('note','')))}</td></tr>")
        P.append("</table>")
    P.append(PIPELINE)
    P.append(f"<div class=sub style='margin-top:20px'>auto-refresh 15s · RL log tail: <code>{html.escape(str(s.get('tail','')))}</code></div>")
    return "".join(P)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), H) as srv:
        print(f"RL dashboard on :{PORT}", flush=True)
        srv.serve_forever()
