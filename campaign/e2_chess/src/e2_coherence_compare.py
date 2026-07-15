#!/usr/bin/env python3
"""E2 COHERENCE TEST compare -> results/E2_COHERENCE.json.

Theory refinement under test (E2 grid v3 found the distill-ensemble gap GROWS
with source rating: -0.003 / +0.031 / +0.111 low->high): distill-then-ensemble
gains require a COHERENT policy observed noisily, not merely "noisy data".

Direct test: mixedband = equal parts (by POSITION count) of the 0800-1200 and
2400plus bands in one training set (~24.6M positions, same total size as the
pure 2400plus band). Two different "policies" in one dataset = incoherent
supervision. PREDICTION (recorded before results, see
results/E2_COHERENCE_PREDICTION.json): gap(mixedband) << gap(2400plus)
despite the mixture containing 2400+ data — incoherence, not noise, kills the
gain.

gap convention = same as e2_grid_summary_v4: per Stockfish level,
teacher_ens = games-weighted mean score over the band's teacher trios;
gap = score(student_ens) - teacher_ens, averaged over levels.
"""
import argparse, json, os

MIX_TRIOS = ["trioA", "trioB"]   # teachers s30-32 / s33-35 (disjoint)
MIX_STUDENT = "student"          # students s40-42 (KD from all 6)


def band_gap_from_evals(resdir, band, teacher_trios, student):
    row = {}
    for pol in teacher_trios + [student]:
        f = os.path.join(resdir, f"e2_eval_{band}_{pol}.json")
        if not os.path.exists(f):
            return None, [f]
        with open(f) as fh:
            row[pol] = json.load(fh)
    levels = sorted(row[teacher_trios[0]]["levels"].keys(), key=int)
    per_level, gaps = {}, []
    for lv in levels:
        tw = [(row[p]["levels"][lv]["score"], row[p]["n_games_per_level"])
              for p in teacher_trios]
        t_ens = sum(s * n for s, n in tw) / sum(n for _, n in tw)
        sc = row[student]["levels"][lv]["score"]
        gaps.append(sc - t_ens)
        per_level[lv] = {
            "teacher_trios": {p: row[p]["levels"][lv]["score"]
                              for p in teacher_trios},
            "teacher_ens_weighted": round(t_ens, 4),
            "student_ens": sc,
            "gap": round(sc - t_ens, 4)}
    return {"levels": per_level,
            "mean_gap": round(sum(gaps) / len(gaps), 4)}, []


def pure_band_refs(resdir):
    """Reference gaps for the pure bands: prefer E2_GRID_v4 (all evidence),
    fall back to E2_GRID_v3 mean_gap."""
    v4 = os.path.join(resdir, "E2_GRID_v4.json")
    if os.path.exists(v4):
        with open(v4) as f:
            g = json.load(f)
        key = "gap_by_band_pooled"
        if key in g and all(v is not None for v in g[key].values()):
            return g[key], "E2_GRID_v4.json:gap_by_band_pooled"
    v3 = os.path.join(resdir, "E2_GRID_v3.json")
    if os.path.exists(v3):
        with open(v3) as f:
            g = json.load(f)
        return {b: g["bands"][b]["mean_gap"] for b in g["bands"]}, \
            "E2_GRID_v3.json:mean_gap"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/E2_COHERENCE.json")
    a = ap.parse_args()

    mix, missing = band_gap_from_evals(a.results, "mixedband",
                                       MIX_TRIOS, MIX_STUDENT)
    refs, refsrc = pure_band_refs(a.results)

    out = {"experiment": "E2 coherence test: mixedband (equal-position blend "
                         "of 0800-1200 + 2400plus, ~24.6M positions = pure "
                         "2400plus size) vs pure-band gaps",
           "prediction": "gap(mixedband) << gap(2400plus): distill-ensemble "
                         "gains require a coherent source policy; an "
                         "incoherent two-policy blend kills the gain even "
                         "though it contains the same 2400+ data",
           "mixedband": mix, "missing": missing,
           "pure_band_gaps": refs, "pure_band_gaps_source": refsrc}
    if mix is not None and refs is not None and "2400plus" in refs \
            and refs["2400plus"] is not None:
        gm, gh = mix["mean_gap"], refs["2400plus"]
        gl = refs.get("0800-1200")
        out["gap_mixedband"] = gm
        out["gap_2400plus"] = gh
        out["gap_0800-1200"] = gl
        out["mixed_minus_2400plus"] = round(gm - gh, 4)
        supported = gm < 0.5 * gh
        out["verdict"] = (
            f"SUPPORTED: mixedband gap {gm:+.4f} < half of pure-2400plus gap "
            f"{gh:+.4f} — incoherence kills the distill-ensemble gain"
            if supported else
            f"NOT SUPPORTED: mixedband gap {gm:+.4f} vs pure-2400plus "
            f"{gh:+.4f} — the blend retains >=half of the pure-band gain")
        if gl is not None:
            out["note_vs_lowband"] = (f"mixedband {gm:+.4f} vs pure low-band "
                                      f"{gl:+.4f}")
    else:
        out["verdict"] = "INCOMPLETE: missing eval files or reference gaps"

    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, a.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
