# Follow-ups from talk-deck prep (2026-08-09) — for later discussion

While building the award-talk decks and blogs, a round of deep source-verification Q&A
surfaced five open items that didn't get resolved in the moment. Logged as GitHub issues
(not fixed here) so they can be picked up deliberately, either as 2027-competition
engineering work or as paper material. **Not fully fleshed out — flagged for a follow-up
discussion.**

## Issues opened

| # | Title | Competition angle | Paper angle |
|---|---|---|---|
| [#1](https://github.com/SuuTTT/IJCAI-mahjong/issues/1) | Feature encoding was vendored, never redesigned | try richer/danger planes head-to-head on the real gate | if it still ties, strengthens "representation-bound, not data-bound" ceiling claim |
| [#2](https://github.com/SuuTTT/IJCAI-mahjong/issues/2) | Augmentation's known 推不倒 residual | cheap fix: skip-augment on 推不倒-eligible hands | `aug_verify.py`'s oracle-verification pattern is a reusable methods contribution |
| [#3](https://github.com/SuuTTT/IJCAI-mahjong/issues/3) | kdens3's shipped gain was mostly winner's-curse noise | enforce disjoint selection/confirmation in the gate harness by construction | strongest first-person case study for the evaluation-gap paper's §4 |
| [#4](https://github.com/SuuTTT/IJCAI-mahjong/issues/4) | Source-conditioned multi-corpus BC came back null | re-test with real-strength gating + a pure scale-up before concluding the mechanism failed | another concrete val-acc≠strength datapoint for §5.3 |
| [#5](https://github.com/SuuTTT/IJCAI-mahjong/issues/5) | Meld rate (副露率) is unmeasured | add it to the real-field measurement harness | mechanistic explanation for the win-conversion gap, if it correlates |

## Why these five

Each came out of a "wait, is that actually true / actually verified?" question during deck
prep — not from a planning session. #2 and #3 both trace back to the augmentation and
distill-then-ensemble mechanics that the concise deck explains; #1 and #5 came from
questions about *why* specific numbers in the input representation are what they are;
#4 came from a direct "have we tried X" question that turned up a documented
self-correction in `moyu_MODEL_CARD.md`.

#3 is probably the highest-value one to prioritize: it's a fully-formed, already-audited
case study (not speculative) that directly demonstrates the evaluation-gap paper's central
claim, on the paper's own model, after the paper's own verification discipline was applied
and still missed it one level up.
