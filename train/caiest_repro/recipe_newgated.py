import json, os, glob
BASE="/root/IJCAI-mahjong/train/caiest_repro"
GD=os.path.join(BASE,"ckpt/recipe/gates"); RES=os.path.join(BASE,"RECIPE_RESULTS.json")
SEEN="/tmp/rc_newgated_seen"
seen=set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
done=[os.path.basename(f)[:-len("_s511000.json")] for f in glob.glob(GD+"/*_s511000.json")]
newly=[t for t in done if t not in seen and t!="calib"]
if newly and os.path.exists(RES):
    d=json.load(open(RES))
    for t in sorted(newly):
        c=d.get("configs",{}).get(t)
        if c:
            print(f"GATED {t}: val={c.get('val_acc')} pl={c.get('placement_mean')} "
                  f"CI=[{c.get('ci95_lo')},{c.get('ci95_hi')}] margin={c.get('margin_lo')} {c.get('verdict')}")
            seen.add(t)
    open(SEEN,"w").write("\n".join(sorted(seen)))
