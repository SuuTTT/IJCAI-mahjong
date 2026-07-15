import json
sw = json.load(open("results/PIMC_SWEEP.json"))
g = json.load(open("results/G6_N50_K6_SCORE.json"))
lo, hi = g["ci95"]
cell = dict(belief=False, leaf="score", K=g["k_cutoff"], N=g["n_worlds"],
            blocks=g["n_blocks"], seeds_per_block=g["seeds_per_block"], n_games=g["n_games"],
            block_mean_placement=g["block_mean_placement"], block_sd=g["block_sd"],
            ci95=g["ci95"], ci_lower=lo, clears_2p5=bool(lo > 2.5),
            override_fraction=g["override_fraction"], reject_fraction=g["reject_fraction"],
            n_decisions=g["n_search_decisions"], seconds=g["seconds"],
            games_per_hour=g["games_per_hour"])
# avoid duplicate if rerun
sw["cells"] = [c for c in sw["cells"] if not (c["N"] == 50 and c["K"] == 6 and c["leaf"] == "score" and not c["belief"])]
sw["cells"].append(cell)
sw["ranked"] = sorted(sw["cells"], key=lambda x: -x["block_mean_placement"])
sw["best"] = sw["ranked"][0]
sw["any_clears_2p5"] = any(c["clears_2p5"] for c in sw["cells"])
sw["n_cells_done"] = len(sw["cells"])
sw["note"] = ("Full belief x leaf x K grid at N=20 (8 cells) + N=50 baseline. Every upgrade "
              "(belief-weighted determinization, placement-head leaf, deeper K=12) DEGRADES "
              "placement vs the plain uniform+score+K6 baseline. N=50 (more worlds) is the only "
              "lever that helps: baseline 2.48 (N=20) -> 2.52 (N=50). No moderate cell's CI "
              "lower bound clears 2.5.")
json.dump(sw, open("results/PIMC_SWEEP.json", "w"), indent=1)
print("merged; ranked:")
for c in sw["ranked"]:
    print(f"  belief={int(c['belief'])} leaf={c['leaf']:9s} K={c['K']:2d} N={c['N']}: "
          f"{c['block_mean_placement']:.4f} CI{c['ci95']} clears2.5={c['clears_2p5']}")
