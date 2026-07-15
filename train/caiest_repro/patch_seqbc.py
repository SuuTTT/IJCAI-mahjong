import io
f = "seq_bc.py"
s = open(f).read()
a1 = '    ap.add_argument("--emb", type=int, default=64); ap.add_argument("--gru", type=int, default=256)'
assert a1 in s, "a1 anchor missing"
if "--gru_layers" not in s:
    add = (a1 + '\n'
           '    ap.add_argument("--gru_layers", type=int, default=1)\n'
           '    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--tf_layers", type=int, default=3)')
    s = s.replace(a1, add)
n1 = '    net = build_seq(a.kind, channels=a.channels, blocks=a.blocks, emb=a.emb, gru=a.gru).to(dev)'
assert n1 in s, "n1 anchor missing"
n1new = ('    _scfg = dict(channels=a.channels, blocks=a.blocks, emb=a.emb, gru=a.gru, '
         'gru_layers=a.gru_layers, heads=a.heads, tf_layers=a.tf_layers)\n'
         '    net = build_seq(a.kind, **_scfg).to(dev)')
s = s.replace(n1, n1new)
e1 = '    ema_net = build_seq(a.kind, channels=a.channels, blocks=a.blocks, emb=a.emb, gru=a.gru).to(dev)'
assert e1 in s, "e1 anchor missing"
s = s.replace(e1, '    ema_net = build_seq(a.kind, **_scfg).to(dev)')
open(f, "w").write(s)
import ast
ast.parse(s)
print("patched + parses OK")
