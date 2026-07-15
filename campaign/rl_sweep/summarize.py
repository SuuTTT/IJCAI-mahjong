import glob, json, os, time
BASE="/root/rl_sweep"
# per-config baseline = that run's own update-0 win_rate (anchor-vs-anchor under this greedy eval harness, ~0.163)
rows=[]
for d in sorted(glob.glob(BASE+"/g*_*")):
    tag=os.path.basename(d)
    jl=sorted(glob.glob(d+"/seed*_jax_results.jsonl"))
    running = any(True for _ in glob.glob(d+"/run.log"))
    recs=[]
    if jl:
        for line in open(jl[-1]):
            line=line.strip()
            if line:
                try: recs.append(json.loads(line))
                except: pass
    if not recs:
        rows.append((tag,"-","-","-","-","-","no-eval-yet")); continue
    first,last=recs[0],recs[-1]
    base=first["win_rate"]
    wr=[r["win_rate"] for r in recs]
    best=max(wr)
    delta=last["win_rate"]-base
    verdict="BEATS(+%.3f)"%delta if last["win_rate"]>base+0.015 else ("~anchor" if abs(delta)<=0.015 else "below(%.3f)"%delta)
    rows.append((tag, last["step"], round(last["win_rate"],3), round(best,3),
                 round(last["mean_score"],3), round(last.get("KL",0),3), verdict))
# write MD
with open(BASE+"/results/RL_SWEEP_STATUS.md","w") as f:
    f.write("# RL Sweep Status (F3 attack on the kdens3 SL-anchor ceiling)\n\n")
    f.write("Updated: %s\n\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    f.write("Strength metric = seat0 (current policy) win_rate vs seats1-3 = FROZEN kdens3 anchor.\n")
    f.write("Baseline = each run's update-0 win_rate (~0.163, anchor-vs-anchor, greedy eval). verdict compares last vs that per-run baseline.\n\n")
    f.write("| config | upd | win_rate | best_wr | mean_score | KL | verdict |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| %s | %s | %s | %s | %s | %s | %s |\n" % r)
    f.write("\nRaw per-config JSONL: g*_*/seed*_jax_results.jsonl (win_rate, mean_score, dealin_rate, KL, env_steps).\n")
print(open(BASE+"/results/RL_SWEEP_STATUS.md").read())
