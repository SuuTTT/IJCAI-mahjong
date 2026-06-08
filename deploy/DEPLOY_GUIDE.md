# Deploy Guide — IJCAI Mahjong (durable reference for every milestone)

## The deploy payload (in `deploy/ship/`, regenerated each milestone)
| File | What | Upload to Botzone as | md5 |
|---|---|---|---|
| `caiest_cnn_bot.zip` | bot CODE (WH-fixed) | **bot source** | `064a49cbbf67674bfaacb8c3061e1a09` |
| `cnn_v1.pkl` | V1 model (candidate) | Storage `data/cnn.pkl` | `9c1863e3b59923c5215b332bd483682c` |
| `cnn_distill100b.pkl` | distill100b (floor) | Storage `data/cnn.pkl` | `7e45c41309502865b824f90b41a0a537` |
`MD5SUMS.txt` ships alongside — always `md5sum -c MD5SUMS.txt` after download and after Botzone upload.

## Botzone upload steps
1. My Bots → the bot → upload `caiest_cnn_bot.zip` as the **Python source** (it has `__main__.py` at root).
2. 用户存储空间 (Storage) → upload the chosen `*.pkl` so its path is `data/cnn.pkl` (the bot auto-loads the largest `.npz`/`.pkl` in `data/`).
3. Smoke test: run one manual match. Expect legal play + occasional `HU`. (If it only ever PLAY/PASS, MahjongGB import failed — check the log.)
4. For an A/B: a SECOND bot, same zip, the other `.pkl` as its Storage.

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
