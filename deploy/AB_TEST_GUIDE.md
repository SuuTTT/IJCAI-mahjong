# Ladder A/B Guide — SHARED user data/ (2026-06-08)

Botzone Storage `data/` is SHARED across all your bots, so you canNOT differentiate bots by Storage.
Instead, each bot's CODE zip carries a `model.cfg` (one line) naming which file to load from the shared
data/. Upload all models to data/ ONCE; switch models by swapping the tiny per-bot zip.

## One-time: upload all candidate models to the shared data/
Upload these to 用户存储空间 data/ (keep all — they coexist):
  cnn_distill100b.pkl (7e45c413)  cnn_v1.pkl (9c1863e3)  cnn_raven.pkl (66858558)
(optional cnn_base2025.pkl d69c9de9)

## Per bot: upload the matching zip (each reads its own file from shared data/)
| Bot | upload this zip | loads | debug shows |
|---|---|---|---|
| A floor | bot_distill100b.zip | data/cnn_distill100b.pkl | [cnn_distill100b.pkl md5=7e45c413] |
| B | bot_v1.zip | data/cnn_v1.pkl | [cnn_v1.pkl md5=9c1863e3] |
| C | bot_raven.zip | data/cnn_raven.pkl | [cnn_raven.pkl md5=66858558] |

VERIFY: run a debug match; the log `debug` field must show the EXPECTED md5 prefix for that bot.
If you have only ONE bot and test sequentially: upload all pkls once, then just swap the 17KB zip
between batches (fast — no need to re-upload the 57MB models).

Fallback if no model.cfg: bot reads data/cnn.pkl, else the largest *.pkl.
