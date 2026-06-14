import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
M = os.environ.get("METRICS_PATH", "/root/metrics.json")
PAGE = b"""<!doctype html><html><head><meta charset=utf-8><title>CSM RL</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body{font:14px sans-serif;margin:18px;background:#111;color:#eee}canvas{background:#1c1c1c;border-radius:8px;margin:8px}h2{color:#6cf}</style>
</head><body><h2>CSM self-play RL &mdash; live</h2><div id=stat>waiting...</div>
<canvas id=c1 height=80></canvas><canvas id=c2 height=80></canvas><canvas id=c3 height=80></canvas><canvas id=c4 height=80></canvas>
<script>
const mk=(id,l,c)=>new Chart(document.getElementById(id),{type:'line',data:{labels:[],datasets:[{label:l,data:[],borderColor:c,pointRadius:0,borderWidth:1.5}]},options:{animation:false,scales:{x:{ticks:{color:'#888'}},y:{ticks:{color:'#888'}}},plugins:{legend:{labels:{color:'#eee'}}}}});
const w=mk('c1','8-fan win rate %','#6f6'),d=mk('c2','draw rate %','#f96'),e=mk('c3','entropy','#fc6'),g=mk('c4','games/s','#6cf');
async function up(){try{const r=await(await fetch('/data')).json();const h=r.history||[];const L=h.map(x=>x.iter);
w.data.labels=L;w.data.datasets[0].data=h.map(x=>(x.winrate8*100).toFixed(2));w.update();
d.data.labels=L;d.data.datasets[0].data=h.map(x=>(x.draw_rate*100).toFixed(1));d.update();
e.data.labels=L;e.data.datasets[0].data=h.map(x=>x.entropy);e.update();
g.data.labels=L;g.data.datasets[0].data=h.map(x=>x.games_per_s);g.update();
const l=r.latest||{};document.getElementById('stat').innerHTML=`iter <b>${l.iter}</b> | 8-fan win <b>${(l.winrate8*100).toFixed(2)}%</b> | draw ${(l.draw_rate*100).toFixed(0)}% | reward ${(l.mean_reward||0).toFixed(2)} | entropy ${(l.entropy||0).toFixed(2)} | ${l.games_per_s} g/s`;
}catch(e){}}
setInterval(up,3000);up();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/data"):
            try: body=open(M,'rb').read()
            except Exception: body=b'{"history":[],"latest":{}}'
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers(); self.wfile.write(PAGE)
ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT","8080"))), H).serve_forever()
