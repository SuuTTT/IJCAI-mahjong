# Deploy candidates (all fused, BN-free, torch-1.4-safe, hardened loader)

Same code zip for all: **`deploy/caiest_cnn_bot.zip`**. Only the Storage `data/cnn.pkl` differs.
The bot auto-detects arch from the checkpoint keys; a wrong upload plays legal-fallback (never RE).

| Upload as data/cnn.pkl | md5 | gauntlet avg net | notes |
|---|---|---|---|
| **distill100b (PRIMARY)** = current `deploy/caiest_cnn/data/cnn.pkl` | `7e45c41309502865b824f90b41a0a537` | +113 | champion-imitation; gauntlet top-tier |
| dense  = `deploy/cnn_dense_fused.pkl` | `d93d4b44c2947374c32008728a8a9ac2` | +142 | dense-reward RL; gauntlet co-top |
| resbn40 base = `deploy/cnn_resbn40_fused.pkl` | `418b6413a91300b75c12892b284b44e5` | −94 | old "best"; gauntlet LAST (A/B control) |

Recommended: deploy **distill100b** as the main bot; A/B vs **dense** and the **base** control on the
ladder (the gauntlet is a proxy — the ladder is the real yardstick). Verify md5 before each upload
(the non-fused resbn40.pkl, md5 07136e8…, is a near-identical 57MB file that causes RE — do NOT upload it).
