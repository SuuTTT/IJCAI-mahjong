# Deploy Guide — IJCAI Mahjong (durable reference for every milestone)

## CURRENT DEPLOYED STATE (2026-06-10): 2-bot shared-data A/B
Botzone Storage `data/` is SHARED across all of a user's bots, so bots are differentiated by a
`model.cfg` (one line = model filename) baked into each bot's CODE zip:
| Bot | Code zip (md5) | model.cfg → loads | model md5 |
|---|---|---|---|
| A (floor) | `bot_distill100b.zip` (`db86fd5b…`) | `cnn_distill100b.pkl` | `7e45c413…` |
| B (candidate) | `bot_lad_chunjiandu.zip` (`ad016476…`) | `cnn_lad_chunjiandu.pkl` | `d517e6a9…` |
Both `.pkl`s live together in the shared Storage `data/`. Verify each bot picked its own model via
the first-turn debug line `[<file> md5=…]`. **The ladder A/B between these two decides the final lock.**

## The deploy payload (in `deploy/ship/`, regenerated each milestone)
| File | What | Upload to Botzone as | md5 |
|---|---|---|---|
| `caiest_cnn_bot.zip` | bot CODE (WH-fixed, no model.cfg → falls back to `cnn.pkl`) | **bot source** | `c591bfff99cee6c737267472541ea808` |
| `bot_*.zip` | per-bot CODE (same code + `model.cfg`) | **bot source** | see `MD5SUMS.txt` |
| `cnn_*.pkl` | models (fused, torch-1.4 legacy serialization) | Storage `data/<same name>` | see `MD5SUMS.txt` |
`MD5SUMS.txt` is the single source of truth — regenerate it whenever ship/ changes; always
`md5sum -c MD5SUMS.txt` after download and after Botzone upload.

## Botzone upload steps
1. My Bots → the bot → upload the bot's zip as the **Python source** (it has `__main__.py` at root).
2. 用户存储空间 (Storage) → upload the chosen `*.pkl` keeping its filename (matching the zip's
   `model.cfg`), or as `data/cnn.pkl` for the no-cfg fallback.
3. Smoke test: run one manual match. Expect legal play + occasional `HU`, and check the debug line
   shows the EXPECTED model md5. (If it only ever PLAY/PASS, MahjongGB import failed — check the log.)
4. For an A/B: a SECOND bot with a different `model.cfg` zip — NOT a different Storage (it's shared).

---

## Transfer methods (research-os EC2 ⇄ your laptop)

### Option 1 — scp (zero-setup, most secure, USE NOW)
You already scp data *to* research-os; just reverse it. From your **laptop**:
```bash
mkdir -p ~/Downloads/mahjong-deploy
scp -r research-os:~/IJCAI-mahjong/deploy/ship/ ~/Downloads/mahjong-deploy/
cd ~/Downloads/mahjong-deploy/ship && md5sum -c MD5SUMS.txt   # must say OK
```
Nothing leaves your controlled infra; no third party; no credentials. **Best default.**

### Option 2 — AWS S3 (browser-downloadable links, shareable, repeatable)
Natural fit (we're on EC2, `boto3` is installed). **You provide ONE of:**
- **(Recommended, no secrets) Attach an IAM role to this instance.** In AWS console (region **ap-southeast-1**):
  IAM → Roles → Create role → EC2 → attach policy `AmazonS3FullAccess` (or scope to one bucket) →
  then EC2 → instance **`i-0f0b456c34261f868`** → Actions → Security → Modify IAM role → select it.
  Tell me the **bucket name** (or let me create one). boto3 then auths automatically — I never see a secret.
- **(Alternative) An access key + secret** for an IAM user with S3 access. Less ideal (secret lives on the box).

Then my flow each milestone:
```python
# upload + 7-day browser link (presigned = private + expiring, good for model weights)
python3 deploy/s3_ship.py up <bucket> deploy/ship           # prints a download URL per file
```
You click the link (or `aws s3 cp s3://<bucket>/ship/ . --recursive`), `md5sum -c`, upload to Botzone.

**You → me (data) via S3:** upload your zip to the bucket and send me a **presigned URL** (or just the
`s3://bucket/key` if I have the role) — I pull it with `wget`/boto3. A presigned URL needs nothing on my side.

### Option 3 — Google Cloud / Drive (data direction only, no setup for me)
If your data lives in GCS/Drive, just send a **public or signed URL** — I `wget` it. No creds needed on my side.
(For me→you, S3 or scp is cleaner since we're on EC2.)

---

## Recommendation
- **Right now:** Option 1 (scp) — deploy the WH-fixed zip + chosen model immediately, no setup.
- **Repeatable channel:** set up Option 2 (S3 + IAM role) once → then every milestone is one `s3_ship.py up`
  from me + one click from you, both directions, private expiring links.
- For **model weights** prefer presigned/private links over public hosts (don't hand competitors our weights pre-final).

## Each future milestone (the loop)
1. I retrain/validate → regenerate `deploy/ship/` (new pkl + refreshed `MD5SUMS.txt`) + update `deploy/CANDIDATES.md` with the verdict.
2. I push via the agreed channel (scp target or `s3_ship.py up`).
3. You download, `md5sum -c`, upload to Botzone, smoke test, freeze.
4. You send new data as a presigned/public URL; I `wget` and start the next cycle.
