# botzone_collect.py — automated top-bot game creation (run on YOUR laptop)

Automates the repetitive "创建游戏桌 → CSM → add 4 bots by ID → 开始游戏 → grab log" loop on your
own Botzone account, so the games exist for your hourly collector to harvest.

## Why it runs on your laptop, not the server
The captcha (every ~5–10 creations) needs a human to see and type it. The script opens a **visible**
Chromium and **pauses** for you to clear the captcha, then resumes. Headless on the research box
can't do that. It also **waits out the add-bot cooldown** rather than evading it (no multi-account).

## Run
```bash
pip install playwright && playwright install chromium
export BOTZONE_USER='1015011749@qq.com'
export BOTZONE_PASS='your-password'        # treat as exposed — consider rotating after
python3 botzone_collect.py --games 300 --out ~/botzone_logs
```
Leave the window visible. When a captcha pops, type it in the browser — the script auto-continues.
Logs are saved to `~/botzone_logs/manual_*.json` as a backup; the hourly collector also catches
the games on Botzone directly.

## First-run tuning (selectors are best-effort)
The DOM selectors in `SEL{}` are written from your description; if a step doesn't click, do that one
step manually once and adjust the matching `SEL[...]` string (use the browser inspector). The 4 bot
IDs are in `BOTS[]` (default = chunjiandu / SelfRegPO / yigeiwoligiaogiao / QwQ).

## Getting the logs to the pipeline
Either let the hourly collector harvest them, or:
```bash
scp ~/botzone_logs/*.json research-os:~/IJCAI-mahjong/others/ladder_top30_score1216/future_hourly/exactly3_top30/logs/
```
Then the re-distill loop picks them up next cycle (extract → distill-from-lad_chunjiandu → gauntlet).

## What it deliberately does NOT do
- Solve/bypass the captcha (you do it).
- Create extra accounts to dodge the cooldown (it waits instead).
These are Botzone's anti-automation controls; the script stays within "automate my own account."
