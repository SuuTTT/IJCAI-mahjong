"""
dashboard.py — tiny zero-dependency training/eval dashboard.

Serves a self-refreshing web page (default port 8080, which the project's SSH
config forwards to your laptop) showing:
  • live training curves (val-acc / val-loss per epoch) for any *train*.log
  • whether the training process is still running
  • the chained pipeline output (head-to-head + legality) from a status file
  • model checkpoints on disk

Run:  python3 eval/dashboard.py [PORT]
Then open  http://localhost:8080  in your browser.
"""
import os, re, glob, html, http.server, socketserver, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# log files to scan for "ep N/M ... va_loss=.. va_acc=.." lines
TRAIN_LOGS = ["/tmp/v4_train.log"] + glob.glob(os.path.join(ROOT, "train/checkpoints/*train*.log"))
STATUS_FILE = "/tmp/v4_done.txt"

EP_RE = re.compile(r"ep\s+(\d+)/(\d+).*?va_loss=([\d.]+)\s+va_acc=([\d.]+)")
# also match the v2/v3 trainer format "Epoch  N/M ... va_loss=.. va_acc=.."
EP_RE2 = re.compile(r"Epoch\s+(\d+)/(\d+).*?va_loss=([\d.]+)\s+va_acc=([\d.]+)")


def parse(logpath):
    rows = []
    try:
        for line in open(logpath):
            m = EP_RE.search(line) or EP_RE2.search(line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)),
                             float(m.group(3)), float(m.group(4))))
    except FileNotFoundError:
        return None
    return rows


def sparkline_svg(rows, key, color, ymin=None, ymax=None, w=560, h=140):
    """Simple SVG line chart of (epoch, value)."""
    if not rows:
        return "<i>no data yet</i>"
    xs = [r[0] for r in rows]
    ys = [r[2] if key == "loss" else r[3] for r in rows]
    lo = ymin if ymin is not None else min(ys)
    hi = ymax if ymax is not None else max(ys)
    if hi == lo: hi = lo + 1e-6
    pad = 24
    def px(x): return pad + (x - xs[0]) / max(1, (xs[-1]-xs[0])) * (w - 2*pad)
    def py(y): return h - pad - (y - lo) / (hi - lo) * (h - 2*pad)
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    best_i = (min(range(len(ys)), key=lambda i: ys[i]) if key == "loss"
              else max(range(len(ys)), key=lambda i: ys[i]))
    bx, by, bv = px(xs[best_i]), py(ys[best_i]), ys[best_i]
    return f'''<svg width="{w}" height="{h}" style="background:#0b1020;border-radius:8px">
      <polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>
      <circle cx="{bx:.1f}" cy="{by:.1f}" r="4" fill="#ffd166"/>
      <text x="{bx:.1f}" y="{by-8:.1f}" fill="#ffd166" font-size="11" text-anchor="middle">best {bv:.3f}@ep{xs[best_i]}</text>
      <text x="6" y="14" fill="#8ab" font-size="11">{key}  (lo {lo:.3f} – hi {hi:.3f})</text>
    </svg>'''


def page():
    proc_alive = os.popen("pgrep -f train_bc_v4.py").read().strip() != ""
    parts = ['<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=10>',
             '<title>Mahjong AI dashboard</title>',
             '<style>body{font-family:system-ui;margin:24px;background:#0f1226;color:#dde}'
             'h1{font-size:20px}h2{font-size:15px;color:#9bd;margin-top:24px}'
             'pre{background:#0b1020;padding:12px;border-radius:8px;overflow:auto;font-size:12px;color:#cde}'
             '.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px}'
             '.on{background:#1d6}.off{background:#555}</style>']
    parts.append("<h1>IJCAI Mahjong — training / eval dashboard</h1>")
    status = '<span class="tag on">TRAINING</span>' if proc_alive else '<span class="tag off">idle / done</span>'
    parts.append(f"<div>bc_v4 training: {status} &nbsp; (auto-refresh 10s)</div>")

    # training curves (use the first log that has data)
    rows = None; used = None
    for lp in TRAIN_LOGS:
        r = parse(lp)
        if r:
            rows, used = r, lp; break
    if rows:
        last = rows[-1]
        best = min(rows, key=lambda r: r[2])
        parts.append(f"<h2>Training &nbsp;<small>{html.escape(used)}</small></h2>")
        parts.append(f"<div>epoch {last[0]}/{last[1]} &nbsp; latest va_acc={last[3]:.3f} "
                     f"va_loss={last[2]:.3f} &nbsp;|&nbsp; <b>best va_loss={best[2]:.3f} "
                     f"(va_acc={best[3]:.3f}) @ ep{best[0]}</b></div>")
        parts.append("<div style='display:flex;gap:16px;flex-wrap:wrap;margin-top:8px'>")
        parts.append(sparkline_svg(rows, "acc", "#5cf"))
        parts.append(sparkline_svg(rows, "loss", "#f97"))
        parts.append("</div>")

    # PPO self-play progress
    ppo = []
    try:
        for line in open("/tmp/league.log"):
            m = re.search(r"iter\s+(\d+)\s+win%=([\d.]+)\s+draw%=([\d.]+).*?loss=([-\d.]+)", line)
            if m:
                net = None
                mn = re.search(r"vs-baseline net=([+\-\d.]+)", line)
                if mn: net = float(mn.group(1))
                ppo.append((int(m.group(1)), float(m.group(2)), float(m.group(4)), net))
    except FileNotFoundError:
        pass
    if ppo:
        last = ppo[-1]
        nets = [(i, n) for i, _, _, n in ppo if n is not None]
        parts.append("<h2>PPO self-play</h2>")
        parts.append(f"<div>iter {last[0]} &nbsp; win%={last[1]:.1f} &nbsp; loss={last[2]:.3f}"
                     + (f" &nbsp; <b>vs-baseline net={nets[-1][1]:+.0f}</b>" if nets else "")
                     + "</div>")
        # win% sparkline
        rows = [(i, 0, 0.0, w) for i, w, _, _ in ppo]
        parts.append(sparkline_svg(rows, "acc", "#7f7"))
        if len(nets) >= 2:
            nrows = [(i, 0, 0.0, n) for i, n in nets]
            parts.append("<div>vs-baseline net score (higher = PPO beating SL):</div>")
            parts.append(sparkline_svg(nrows, "acc", "#fd7"))

    # pipeline status
    parts.append("<h2>Pipeline (fp16 → head-to-head → legality)</h2>")
    try:
        out = open(STATUS_FILE).read()[-4000:]
        parts.append(f"<pre>{html.escape(out) or '(empty — waiting)'}</pre>")
    except FileNotFoundError:
        parts.append("<pre>(not started)</pre>")

    # checkpoints
    parts.append("<h2>Model checkpoints (.npz)</h2><pre>")
    for f in sorted(glob.glob(os.path.join(ROOT, "train/checkpoints/*.npz"))):
        sz = os.path.getsize(f) / 1e6
        parts.append(f"{sz:6.2f} MB  {os.path.basename(f)}\n")
    parts.append("</pre>")
    return "".join(parts)


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
        print(f"dashboard at http://localhost:{PORT}  (Ctrl-C to stop)")
        srv.serve_forever()
