# Tutorial: Why Our Mahjong Bot Drew Every Game (for non-Mahjong researchers)

This explains, from zero, two phrases from the project notes:

> "peng/chi/gang decisions are a **fan-blind shanten heuristic**, so it never melds
> toward **high-value patterns** (碰碰和, 清一色, 混一色)."

and what it means to make those decisions **model-driven**. No Mahjong background assumed.

---

## 1. The 60-second rules you need

- There are **34 tile types**, 4 copies each (136 tiles). Three "suits" of number tiles
  1–9 (characters **W**, dots **B**, bamboo **T**), plus honor tiles (winds, dragons).
- Your **hand** holds 13 tiles; on your turn you draw a 14th and discard one.
- A **winning hand** = **4 "sets" + 1 "pair"**. A set is either:
  - a **triplet** (three identical tiles, e.g. `W2 W2 W2`), or
  - a **sequence** (three consecutive in one suit, e.g. `T4 T5 T6`).
  - (a **kong** = four identical, counts as a special set.)
- You can complete a set using a tile **another player discards**, by *claiming* it:
  - **Peng (碰)** — take a discard to finish a **triplet** (you hold 2, claim the 3rd).
  - **Chi (吃)** — take a discard to finish a **sequence** (only from the player to your left).
  - **Gang (杠)** — take a discard (or your own tile) to make a **kong** (four of a kind).
  - Claiming exposes that set face-up and skips you straight to discarding.

When you claim, you reveal a set ("meld") and get closer to the 4-sets-+-pair goal faster.

---

## 2. The catch that makes Mahjong hard: the 8-fan floor

In Chinese Standard Mahjong you **cannot win with just any 4-sets-+-pair**. The hand must
also score at least **8 fan (番)** — fan are points awarded for *patterns*. Declaring a win
worth fewer than 8 fan is an **illegal move** (a −30 penalty). So there are really two goals:

1. **Be "ready"** (one tile away from 4 sets + 1 pair). Distance-to-ready is called **shanten**:
   shanten 0 = ready (called *tenpai*), shanten 1 = one useful tile away, etc.
2. **Be worth ≥ 8 fan** when you get there.

These two goals **pull in different directions**, and that is the whole story below.

### What patterns score big?

A plain hand of mixed sequences is worth ~0–4 fan — *not enough to win*. The points come
from **structured** hands, e.g.:

| Pattern | What it is | Fan |
|---|---|---|
| 清一色 (full flush) | every tile in **one suit** | 24 |
| 组合龙 (knitted straight) | a 1-4-7 / 2-5-8 / 3-6-9 pattern across suits | 12 |
| 混一色 (half flush) | **one suit + honor tiles** only | 6 |
| 碰碰和 (all triplets) | the hand is **all triplets**, no sequences | 6 |

Reaching ≥ 8 fan almost always means *committing early* to one of these shapes.

---

## 3. What "shanten heuristic" means, and why it is "fan-blind"

A **heuristic** here is a hand-written rule. Our claim rule was:

> *"Peng/Chi/Gang a discard only if doing so lowers my shanten (gets me closer to ready)."*

This is **greedy toward speed**. It asks one question — *"does this claim get me to ready
faster?"* — and **never asks** *"is the hand I'm rushing toward worth 8 fan?"* That is what
**fan-blind** means: blind to the score, optimizing only the distance-to-ready.

### Why that loses every game

Consider you hold `W2 W2` and someone discards `W2`. Peng-ing it instantly gives you a
triplet and lowers shanten — the heuristic happily takes it. But suppose your other tiles
are a jumble of three suits. You now rocket to "ready" on a hand worth maybe **2 fan**.
You're ready… but you **can't legally win** (2 < 8). You sit there discarding until the wall
runs out. **Every game ends in a draw with nobody scoring** — which is exactly what our logs
showed: clean play, 0 illegal moves, and `canHu` topping out at 4 fan.

The fix a human expert applies: *don't peng that W2.* Instead, keep collecting one suit and
aim for 清一色 (24 fan), or peng only tiles that build toward 碰碰和 (all-triplets, 6 fan).
The expert sacrifices **speed** for **value**, because a fast hand under 8 fan is worthless.

A shanten-only rule cannot make that trade-off, because it does not know what any pattern is
worth. It needs a notion of **value**, not just **distance**.

---

## 4. What "model-driven" means

Instead of a hand-written rule, we use a **neural network policy** — the "model." It was
trained by **imitation learning** on the official dataset of **98,209 games played by strong
AIs**: ~1.3–5.1 million real (situation → action) examples.

- **Input**: a 240-number summary of the current situation (your hand, what's been discarded,
  the exposed melds, the winds, etc.).
- **Output**: a score for each of the 235 possible actions (pass, win, discard each tile,
  peng, chi, gang, …). We take the highest-scoring **legal** action.

Because it learned from experts, the model has implicitly absorbed *value*: in positions
where experts pass a tempting peng to chase 清一色, the model also learned to pass; where they
peng to build 碰碰和, it pengs. It is **not fan-blind** — it encodes "what good players do,"
which already accounts for the 8-fan economy. Formally, for legal action set \(\mathcal{A}(o)\):

$$ \pi(a \mid o) = \frac{\exp z_a(o)\,\mathbb{1}[a \in \mathcal{A}(o)]}{\sum_{a' \in \mathcal{A}(o)} \exp z_{a'}(o)}, \qquad a^\* = \arg\max_a \pi(a \mid o) $$

### The bug in our deployment

We *had* this model, but we were only using it to pick **which tile to discard**. The
**peng/chi/gang** decisions still went through the old fan-blind shanten rule. So the part of
the game that most determines whether you ever reach 8 fan — *what to claim and what to build
toward* — was being made by the weakest component. The fix is to let the model decide claims
too, while keeping two safety rails:

1. **HU stays fan-gated.** We only declare a win when an exact fan calculation confirms ≥ 8 —
   we never trust the model to keep a win legal.
2. **Every claim is legality-checked** (`verify_claim`) before it leaves the bot, so a
   model mistake can at worst become a harmless `PASS`, never an illegal `−30`.

---

## 5. Summary

| | fan-blind shanten heuristic | model-driven |
|---|---|---|
| Decides claims by | "does it speed me to ready?" | "what would a strong player do here?" |
| Knows pattern value? | No | Yes (learned from data) |
| Builds 清一色 / 碰碰和? | Only by accident | On purpose |
| Typical result vs peers | ready but < 8 fan → **draw** | commits to a scoring shape → **can win** |

The next change makes claims model-driven, keeps the legality rails, and is validated locally
with `eval/gen_log.py` (which replays full games through the official judge) before any upload.
