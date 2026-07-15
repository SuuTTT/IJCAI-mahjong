"""3-way gate: over ONE fixed seed set, evaluate mean payoff of
  (i) single student, (ii) 3-teacher seed-ensemble, (iii) 3-student distill-ensemble,
plus a ref==ref calibration (delta should be 0). Writes a JSON."""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dou_gate import EnsembleAgent, play

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--single", required=True, help="1 pkl")
    ap.add_argument("--teacher_ens", required=True, help="comma 3 teacher pkls")
    ap.add_argument("--student_ens", required=True, help="comma 3 student pkls")
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--nseeds", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(10000, 10000 + a.nseeds))
    t0 = time.time()

    single_pkls = [a.single]
    teacher_pkls = a.teacher_ens.split(",")
    student_pkls = a.student_ens.split(",")

    single_ag = EnsembleAgent(single_pkls, a.hidden, a.layers, dev)
    tea_ag = EnsembleAgent(teacher_pkls, a.hidden, a.layers, dev)
    stu_ag = EnsembleAgent(student_pkls, a.hidden, a.layers, dev)
    ref2_ag = EnsembleAgent(single_pkls, a.hidden, a.layers, dev)  # calibration copy

    single_mp = play(single_ag, a.seat, seeds, dev)
    tea_mp = play(tea_ag, a.seat, seeds, dev)
    stu_mp = play(stu_ag, a.seat, seeds, dev)
    ref2_mp = play(ref2_ag, a.seat, seeds, dev)

    res = {
        "regime": a.regime, "seat": a.seat, "nseeds": a.nseeds, "dev": dev,
        "single_pkl": a.single, "teacher_ens": teacher_pkls, "student_ens": student_pkls,
        "single_mean_payoff": round(single_mp, 5),
        "teacher_ens3_mean_payoff": round(tea_mp, 5),
        "student_ens3_mean_payoff": round(stu_mp, 5),
        "calibration_ref_vs_ref_delta": round(abs(single_mp - ref2_mp), 6),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"WROTE {a.out}", flush=True)

if __name__ == "__main__":
    main()
