"""
f2_aggregate.py — aggregators for the Final2 campaign. t-CI over gate blocks.
  python3 f2_aggregate.py exp1   -> results/F2_CORPUS_KD.json
  python3 f2_aggregate.py jdv2   -> results/JD_V2.json
  python3 f2_aggregate.py score  -> results/SCORE_GATE.json
  python3 f2_aggregate.py value  -> results/F2_VALUE_HEAD.json
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086}


def tci(vals):
    v = np.asarray(vals, np.float64)
    n = len(v)
    if n < 2:
        return dict(mean=round(float(v.mean()), 4) if n else None, ci=None, lo=None, hi=None, n=n)
    t = T975.get(n - 1, 1.96)
    m = v.mean(); s = v.std(ddof=1) / np.sqrt(n)
    return dict(mean=round(float(m), 4), ci=round(float(t * s), 4),
                lo=round(float(m - t * s), 4), hi=round(float(m + t * s), 4), n=n)


def blocks(pattern):
    out = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            out.append(json.load(f))
    return out


def side(name):
    fp = name if os.path.exists(name) else None
    if fp:
        with open(fp) as f:
            return json.load(f)
    return None


def gate_summary(pattern):
    bs = blocks(pattern)
    if not bs:
        return dict(blocks=0, incomplete=True)
    pl = tci([b["placement_pts"] for b in bs])
    sd = tci([b["score_diff"] for b in bs])
    cs = tci([b["cand_score_mean"] for b in bs])
    return dict(blocks=len(bs), games=sum(b["games"] for b in bs),
                placement=pl, score_diff=sd, cand_score=cs,
                block_placements=[b["placement_pts"] for b in bs],
                block_score_diffs=[b["score_diff"] for b in bs])


def exp1():
    arms = {
        "b_mix_all4": dict(gate="results/f2_gate/armb_b*.json", side="ckpt/f2/armb_s*.json"),
        "c_mix_kongmoyu": dict(gate="results/f2_gate/armc_b*.json", side="ckpt/f2/armc_s*.json"),
        "d_pure_bc": dict(gate="results/f2_gate/armd_b*.json", side="ckpt/f2/armd_s*.json"),
    }
    out = dict(
        experiment="FINALIST-CORPUS DISTILLATION: 128x40 students, 60k steps, 3 seeds/arm; "
                   "gates = 3-student deploy ensemble vs KDENS3 ENSEMBLE (kd_128x40_s0/1/2) "
                   "PAIRED duplicate, 12 blocks x 500 seeds (seed0=500000+b*500), dual metric "
                   "(placement pts + raw duplicate score). placement 2.500 = parity with kdens3.",
        corpus="final2_cai_corpus.npz (IJCAI Final2, all-4-finalist decisions, cai encoding)",
        arms={})
    for k, v in arms.items():
        sides = []
        for fp in sorted(glob.glob(os.path.join(HERE, v["side"]))):
            with open(fp) as f:
                sides.append(json.load(f))
        out["arms"][k] = dict(gate=gate_summary(os.path.join(HERE, v["gate"])), train=sides)
    cal = side(os.path.join(R, "f2_gate/CALIBRATION.json"))
    out["calibration_kdens3_vs_kdens3"] = cal
    verdicts = {}
    for k, v in out["arms"].items():
        g = v["gate"]
        if g.get("incomplete"):
            verdicts[k] = "INCOMPLETE"
            continue
        lo, hi = g["placement"]["lo"], g["placement"]["hi"]
        if lo > 2.5:
            verdicts[k] = f"BEATS kdens3 on placement ({g['placement']['mean']}, lo {lo})"
        elif hi < 2.5:
            verdicts[k] = f"LOSES to kdens3 on placement ({g['placement']['mean']}, hi {hi})"
        else:
            verdicts[k] = f"NULL vs kdens3 ({g['placement']['mean']} +- {g['placement']['ci']})"
        sdiff = g["score_diff"]
        verdicts[k + "_score"] = (f"score_diff {sdiff['mean']} [{sdiff['lo']},{sdiff['hi']}] "
                                  + ("(excludes 0)" if sdiff["lo"] and (sdiff["lo"] > 0 or sdiff["hi"] < 0)
                                     else "(includes 0)"))
    out["verdicts"] = verdicts
    with open(os.path.join(R, "F2_CORPUS_KD.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(verdicts, indent=1))


def jdv2():
    out = dict(
        experiment="JD-v2: state-centered danger penalty pen=E[p*(danger-mean_legal danger)]; "
                   "128x40, 60k steps, 2 seeds/lambda; control = existing jd_lam0 (v1 lam=0, "
                   "same trainer lineage) restricted to s0,s1 for ensemble-size parity. "
                   "Gate = 2-student ensemble vs aug_128x40_s0, 8 blocks x 500 seeds "
                   "(seed0=300000+b*500, SAME walls as the v1 JD gate), dual metric; "
                   "deal-in = 6x250 paired games (seeds 900000+e*100000+g, same as v1).",
        danger_std=side(os.path.join(R, "DANGER_STD.json")),
        per_lambda={})
    for lam in ["0", "0.3", "1.0", "3.0"]:
        key = "lam" + lam
        gate = gate_summary(os.path.join(R, f"jdv2_gate/{key}_b*.json"))
        dl = blocks(os.path.join(R, f"jdv2_dealin/{key}_e*.json"))
        dealin = None
        if dl:
            n = sum(x["ngames"] for x in dl)
            di = sum(x["dealins"] for x in dl)
            wi = sum(x["wins"] for x in dl)
            p = di / n
            se = float(np.sqrt(p * (1 - p) / n))
            dealin = dict(n=n, deal_in_rate=round(p, 4), ci95=round(1.96 * se, 4),
                          lo=round(p - 1.96 * se, 4), hi=round(p + 1.96 * se, 4),
                          win_rate=round(wi / n, 4))
        sides = []
        pat = (os.path.join(HERE, "ckpt/jd/jd_lam0_s[01].json") if lam == "0"
               else os.path.join(HERE, f"ckpt/jdv2/jdv2_lam{lam}_s*.json"))
        for fp in sorted(glob.glob(pat)):
            with open(fp) as f:
                sides.append(json.load(f))
        out["per_lambda"][lam] = dict(gate=gate, dealin=dealin, train=sides)
    base = out["per_lambda"].get("0", {})
    verd = {}
    b_dl = (base.get("dealin") or {})
    b_g = (base.get("gate") or {})
    for lam in ["0.3", "1.0", "3.0"]:
        v = out["per_lambda"][lam]
        if not v["dealin"] or v["gate"].get("incomplete") or not b_dl or b_g.get("incomplete"):
            verd[lam] = "INCOMPLETE"
            continue
        ddi = v["dealin"]["deal_in_rate"] - b_dl["deal_in_rate"]
        dpl = v["gate"]["placement"]["mean"] - b_g["placement"]["mean"]
        verd[lam] = (f"deal-in {v['dealin']['deal_in_rate']} vs lam0 {b_dl['deal_in_rate']} "
                     f"(delta {ddi:+.4f}); placement {v['gate']['placement']['mean']} vs "
                     f"lam0 {b_g['placement']['mean']} (delta {dpl:+.4f})")
    out["verdicts"] = verd
    with open(os.path.join(R, "JD_V2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(verd, indent=1))


def score():
    val = gate_summary(os.path.join(R, "score_gate_val/b*.json"))
    bs = blocks(os.path.join(R, "score_gate_val/b*.json"))
    cal_ens = side(os.path.join(R, "score_gate_val/CAL_ens.json"))
    cal_single = side(os.path.join(R, "score_gate_val/CAL_single.json"))
    gph = np.mean([b["games_per_hour"] for b in bs]) if bs else None
    out = dict(
        experiment="SCORE-METRIC DUPLICATE GATE (e12_score_gate.py): Botzone Final Stage-2 "
                   "metric (cumulative raw duplicate score) alongside placement, same paired "
                   "duplicate sim. Validation: kdens3 3-ens vs aug_s0, 12 blocks x 500 seeds.",
        calibration=dict(ens_vs_same_ens=cal_ens, single_vs_same_single=cal_single),
        validation_kdens3_vs_aug_s0=val,
        historical_placement_reference="kdens3 vs aug_s0 = 2.5054 (lo 2.5012) + repl 2.5057",
        throughput_games_per_hour_per_gateproc=round(float(gph), 1) if gph else None,
        final_format_replica_hours_12288_games=(round(12288.0 / gph, 2) if gph else None),
        agreement=None)
    if not val.get("incomplete"):
        pl, sd = val["placement"], val["score_diff"]
        pl_v = "cand better" if pl["lo"] > 2.5 else ("cand worse" if pl["hi"] < 2.5 else "null")
        sd_v = "cand better" if sd["lo"] > 0 else ("cand worse" if sd["hi"] < 0 else "null")
        out["agreement"] = dict(
            placement_verdict=f"{pl_v} ({pl['mean']} [{pl['lo']},{pl['hi']}], parity=2.5)",
            score_verdict=f"{sd_v} (diff {sd['mean']} [{sd['lo']},{sd['hi']}] pts/game, parity=0)",
            agree=(pl_v == sd_v))
    with open(os.path.join(R, "SCORE_GATE.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out.get("agreement"), indent=1))


def value():
    out = dict(
        experiment="SCORE-VALUE HEAD on Final2 corpus: predict actor final duplicate score "
                   "from state; held-out split BY GAME; stages = thirds of decision sequence. "
                   "GRP prior on real data: r=0.829.",
        frozen=side(os.path.join(R, "value_frozen.json")),
        e2e=side(os.path.join(R, "value_e2e.json")))
    fr = out["frozen"]; ee = out["e2e"]
    v = []
    if fr:
        v.append(f"frozen-trunk head: r_all mean {fr.get('r_all_mean')} over {len(fr.get('per_seed', []))} seeds")
    if ee:
        v.append(f"e2e 128x40: r_all {ee.get('metrics', {}).get('r_all')}")
    out["verdict"] = "; ".join(v) if v else "INCOMPLETE"
    with open(os.path.join(R, "F2_VALUE_HEAD.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(out["verdict"])


if __name__ == "__main__":
    {"exp1": exp1, "jdv2": jdv2, "score": score, "value": value}[sys.argv[1]]()
