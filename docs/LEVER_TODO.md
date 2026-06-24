# Lever TODO — test one-by-one via the pipeline (gate → real-field A/B)

**Pipeline per lever:** build/train → **local placement gate** (`frontier_gate.py` / `parity_gate.py`,
calibrated no-op=2.5000/+0.000, N≥400×2 families, per-seat, no-op-trap proof) → **real-field A/B** only
if it beats the reference. Reference to beat: moyu / raw-aggressive base (frontier winner) ≈ +1/g noise band.
**Always keep GPU busy** (training levers) + CPU (gate/controller levers) overlapping.

Status: 🔄 running · ⬜ queued · ✅ done · ❌ null

| # | lever | mechanism | prior | resource | status |
|---|---|---|---|---|---|
| 1 | **Bigger model (256×40, 320×40)** | use 256MB headroom; ~4× capacity | low (capacity documented flat) | GPU | 🔄 `add5ac98` |
| 2 | **Fan-weighted targeted defense** | fold ONLY when P(deal-in)×E[fan] high (dodge the −40 hands) | modest (scoring asymmetry justifies it; blanket fold was null) | CPU | 🔄 `aca718d7` |
| 3 | **Within-game win-rate controller** | value critic (R²≈0.99) → live win-prob → placement-calibrated push/fold (deployable; NOT cross-tournament) | modest-low (value_search=noise, placement-RL≈moyu, but placement-calibrated within-hand rule distinct) | CPU | ⬜ |
| 4 | **Aggressive+conservative ensemble + controller** | hierarchical: controller switches sub-policies by within-game state/win-prob | modest-low (ensemble null, but state-dependent switching distinct) | CPU+GPU | ⬜ |
| 5 | **Reward/loss-eng aggression** | train MORE aggressive than raw (risk-weighted objective) — the one direction the inference overlay couldn't reach; also supplies #4's aggressive arm | low-modest (frontier: raw is peak in tested dir; >raw untested) | GPU | ⬜ |
| — | frontier aggression sweep | global push/fold knob | — | — | ✅ BALANCE, no sweet spot (don't fold broadly) |
| — | feature engineering | richer input planes | — | — | ❌ null (≤0.005 val, no play gain) |
| — | real-field A/B {A,B,C} | ship-decision | — | — | ✅ TIE (keep deployed bot; no upgrade) |

## Execution order (keep GPU busy)
- **GPU queue:** [1 running] → **5** (reward-eng aggression training) → (any bigger-model follow-ups).
- **CPU queue (overlap):** [2 running] → **3** (winrate controller) → **4** (ensemble+controller, needs 5's aggressive arm + 3's controller).
- Gate everything on **placement** (the contest metric), not score/val-acc. Real-field A/B only for gate-winners.

## Honest framing
All of 1–5 are **modest-to-low prior** — the campaign's record is that levers regress to ≈parity, and
the frontier already showed defense loses / aggression is the peak. #2 and #3/#4 are the most-motivated
(the scoring asymmetry + within-game win-prob are genuinely untested in these precise forms). Test each
cleanly; a clean null is a valid result. moyu/deployed bot stays the submission until something passes a
real-field A/B.
</content>
