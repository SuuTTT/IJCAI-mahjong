#!/usr/bin/env python3
"""
botzone_collect.py — automate LAUNCHING Chinese-Standard-Mahjong tables of top-ranker bots on
botzone.org.cn, on YOUR OWN account. We do NOT scrape the JSON — your hourly collector harvests
all global games played; this just makes the top-bot games EXIST. Fire-and-forget: create table →
add 4 bots by ID → 开始游戏 → next.

DESIGN BOUNDARIES (intentional):
  * Human-in-the-loop captcha (every ~5-10 creations): the script PAUSES and waits for YOU to type
    it in the visible browser, then resumes. It does not read or bypass the captcha.
  * Cooldown-respecting: when "add bot" is rate-limited it WAITS the backoff out — it does NOT
    create extra accounts to evade the limit.
  * Runs HEADED on your machine (the captcha needs a visible browser). Headless is only for a
    selector smoke-test (--shots), which will likely hit the captcha wall sooner.

SETUP (your laptop):
  pip install playwright && playwright install chromium
  python3 botzone_collect.py --games 300 --user 1015011749@qq.com --pass '...'
  # (or set BOTZONE_USER / BOTZONE_PASS env; --user/--pass override them — use ANY of your own accounts)
"""
import os, sys, time, argparse, pathlib, re

USER_DEFAULT = os.environ.get("BOTZONE_USER", "1015011749@qq.com")
BASE = "https://botzone.org.cn/"
GAME = "Chinese-Standard-Mahjong"
BOTS = [                                          # (label, bot-ID) — 4 seats; edit freely
    ("chunjiandu",        "69ef25bf83ee0a54c189cd9e"),
    ("SelfRegPO",         "68620536a4349e61674f0a0e"),
    ("yigeiwoligiaogiao", "667648e12e524945e73126bf"),
    ("QwQ",               "6284ebbe3a897766fbbcddf6"),
]
COOLDOWN_WAIT = 90
CAPTCHA_POLL = 3

# Selectors CONFIRMED by a 2026-06-10 smoke test (login + table-create verified working). The
# bot-assignment dialog uses these REAL labels: an `ID` button, a `提交` submit, `开始对局！` to start,
# and a per-seat <select>. The exact per-seat click SEQUENCE is the one thing to eyeball on your
# first headed run (the dialog is obvious on screen) — adjust add_bot_by_id() if needed.
SEL = {
    "login_user":  "input[name='email']:visible, #txtEmail:visible",   # NOT #txtEmail_reg (register form)
    "login_pass":  "input[name='password']:visible, #txtPassword:visible",
    "create_link": "text=创建游戏桌",
    "game_select": "select",                       # game-type / seat selects live in the dialog
    "id_btn":      "text=ID",                       # 'specify bot by ID' button (per seat)
    "id_input":    "input[type='text']:visible, input[placeholder*='ID']:visible",
    "submit_btn":  "text=提交",                     # confirm the bot ID
    "start_btn":   "text=开始对局",                  # NOT 开始游戏
    "captcha_box": "input[placeholder*='验证']:visible, .captcha:visible, #captcha:visible",
    "cooldown_txt":"text=/冷却|稍后|频繁|too frequent|cooldown/i",
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def shot(page, shots, name):
    if shots:
        try: page.screenshot(path=str(pathlib.Path(shots) / f"{name}.png"))
        except Exception: pass


def wait_for_captcha(page):
    try:
        cap = page.locator(SEL["captcha_box"]).first
        if cap.count() and cap.is_visible():
            log("🧩 CAPTCHA — solve it in the browser; I'll continue automatically.")
            while cap.is_visible(): time.sleep(CAPTCHA_POLL)
            log("captcha cleared.")
    except Exception:
        pass


def login(page, user, pw, shots):
    page.goto(BASE); page.wait_for_timeout(2000); shot(page, shots, "01_home")
    if not pw:
        log("no password — log in MANUALLY in the window, then press Enter here."); input(); return
    try:
        page.fill(SEL["login_user"], user); page.fill(SEL["login_pass"], pw)
        page.keyboard.press("Enter"); page.wait_for_timeout(2500)
        wait_for_captcha(page); shot(page, shots, "02_after_login")
        log("login submitted.")
    except Exception as e:
        log(f"auto-login failed ({e}); log in MANUALLY then press Enter."); shot(page, shots, "02_login_fail"); input()


def add_bot_by_id(page, i, label, bid):
    # Per-seat: click the i-th `ID` button → fill the ID → `提交`. (Verify this sequence on your
    # first headed run; if the dialog uses a seat <select> first, set it via SEL['game_select'].)
    for _ in range(20):
        try:
            page.locator(SEL["id_btn"]).nth(i).click(timeout=4000)
            page.locator(SEL["id_input"]).last.fill(bid)
            page.locator(SEL["submit_btn"]).last.click(timeout=4000)
            page.wait_for_timeout(800); wait_for_captcha(page)
            if page.locator(SEL["cooldown_txt"]).count():
                log(f"⏳ cooldown — waiting {COOLDOWN_WAIT}s (not evading)…"); time.sleep(COOLDOWN_WAIT); continue
            log(f"  seat {i}: {label}"); return True
        except Exception as e:
            log(f"  seat {i} retry ({str(e)[:40]})"); page.wait_for_timeout(1500)
    return False


def launch_one(page, n, shots):
    page.goto(BASE); page.wait_for_timeout(1200)
    page.locator(SEL["create_link"]).first.click(); page.wait_for_timeout(2000); shot(page, shots, f"g{n}_03_dialog")
    # the create dialog has a game-type select + per-seat config; pick CSM if a select offers it
    try: page.locator(SEL["game_select"]).first.select_option(label=re.compile("Mahjong"))
    except Exception: pass
    shot(page, shots, f"g{n}_04_dialog2")
    for i, (label, bid) in enumerate(BOTS):
        if not add_bot_by_id(page, i, label, bid):
            log(f"  couldn't add seat {i}; skipping table."); shot(page, shots, f"g{n}_05_addfail"); return False
    shot(page, shots, f"g{n}_06_botsadded")
    page.locator(SEL["start_btn"]).first.click(); page.wait_for_timeout(1500); wait_for_captcha(page)
    shot(page, shots, f"g{n}_07_started")
    log(f"game {n}: launched (fire-and-forget; hourly collector will harvest it).")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--user", default=USER_DEFAULT)
    ap.add_argument("--pass", dest="pw", default=os.environ.get("BOTZONE_PASS", ""))
    ap.add_argument("--shots", default="", help="dir to save per-step screenshots (debug/smoke-test)")
    ap.add_argument("--headless", action="store_true", help="no visible browser (smoke-test only; captcha will wall it)")
    a = ap.parse_args()
    if a.shots: pathlib.Path(a.shots).mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=a.headless)
        page = browser.new_context().new_page()
        login(page, a.user, a.pw, a.shots)
        ok = 0
        for n in range(1, a.games + 1):
            try:
                if launch_one(page, n, a.shots): ok += 1
            except Exception as e:
                log(f"game {n} errored ({str(e)[:80]})"); shot(page, a.shots, f"g{n}_ERR"); page.wait_for_timeout(3000)
            log(f"progress: {ok}/{n} (target {a.games})")
        log(f"DONE: launched {ok} games. The hourly collector harvests them — nothing to copy.")
        browser.close()


if __name__ == "__main__":
    main()
