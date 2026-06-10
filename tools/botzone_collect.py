#!/usr/bin/env python3
"""
botzone_collect.py — automate CREATING Chinese-Standard-Mahjong tables of top-ranker bots on
botzone.org.cn, on YOUR OWN account, so the games exist and your hourly collector harvests them
(it also saves each game's log JSON locally as a backup).

DESIGN BOUNDARIES (intentional):
  * Human-in-the-loop captcha: when Botzone shows the 1-char captcha (every ~5-10 creations), the
    script PAUSES and waits for YOU to type it in the visible browser, then continues. It does not
    attempt to read or bypass the captcha.
  * Cooldown-respecting: when "add bot" is on cooldown, it WAITS the backoff out (configurable),
    it does NOT create extra accounts to evade the limit.
  * Runs HEADED on your local machine (you must see the browser to solve captchas). Not headless.

SETUP (on your laptop):
  pip install playwright && playwright install chromium
  export BOTZONE_USER='1015011749@qq.com'   BOTZONE_PASS='...'      # or edit CONFIG below
  python3 botzone_collect.py --games 300 --out ~/botzone_logs

The 4 bots are added by ID (用ID指定bot). Default = the top table you've been running.
Edit BOTS / SELECTORS below if the UI differs — selectors are best-effort (verify once, then it runs).
"""
import os, sys, json, time, argparse, pathlib, re

# ---- CONFIG ---------------------------------------------------------------
USER = os.environ.get("BOTZONE_USER", "1015011749@qq.com")
PASS = os.environ.get("BOTZONE_PASS", "")        # set via env; do NOT hardcode in a shared repo
BOTS = [                                          # (label, bot-ID) — 4 seats
    ("chunjiandu",        "69ef25bf83ee0a54c189cd9e"),
    ("SelfRegPO",         "68620536a4349e61674f0a0e"),
    ("yigeiwoligiaogiao", "667648e12e524945e73126bf"),
    ("QwQ",               "6284ebbe3a897766fbbcddf6"),
]
GAME = "Chinese-Standard-Mahjong"
BASE = "https://botzone.org.cn/"
COOLDOWN_WAIT = 90          # seconds to back off when "add bot" is rate-limited, then retry
CAPTCHA_POLL = 3            # seconds between checks while waiting for you to solve a captcha

# Selectors are best-effort against the described UI; adjust after one manual run if needed.
SEL = {
    "login_user":  "input[name='email'], input[type='email'], #email",
    "login_pass":  "input[name='password'], input[type='password'], #password",
    "create_table_link": "text=创建游戏桌",
    "game_select": "select",                       # the game-type dropdown on the create page
    "create_btn":  "text=创建",
    "use_id_btn":  "text=用ID指定Bot",              # per-seat "specify bot by ID"
    "id_input":    "input[placeholder*='ID'], input[type='text']",
    "add_btn":     "text=添加",
    "start_btn":   "text=开始游戏",
    "debug_toggle":"text=调试模式",
    "log_tool":    "text=log查看工具, text=Log查看工具",
    "captcha_box": "input[placeholder*='验证'], .captcha, #captcha",
    "cooldown_txt":"text=/冷却|稍后|频繁|too frequent|cooldown/i",
}
# ---------------------------------------------------------------------------


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_for_captcha(page):
    """If a captcha is visible, pause until the human clears it (it disappears / add succeeds)."""
    try:
        cap = page.locator(SEL["captcha_box"]).first
        if cap.count() and cap.is_visible():
            log("🧩 CAPTCHA visible — solve it in the browser window; I'll continue automatically.")
            while cap.is_visible():
                time.sleep(CAPTCHA_POLL)
            log("captcha cleared — continuing.")
    except Exception:
        pass


def login(page):
    page.goto(BASE)
    if not PASS:
        log("BOTZONE_PASS not set — log in MANUALLY in the window now; press Enter here when done.")
        input()
        return
    try:
        page.fill(SEL["login_user"], USER)
        page.fill(SEL["login_pass"], PASS)
        page.keyboard.press("Enter")
        page.wait_for_timeout(2500)
        wait_for_captcha(page)
        log("logged in.")
    except Exception as e:
        log(f"auto-login failed ({e}); log in MANUALLY then press Enter."); input()


def add_bot_by_id(page, seat_idx, label, bot_id):
    """Add one bot to a seat by ID, handling cooldown + captcha. Returns True on success."""
    for attempt in range(20):
        try:
            page.locator(SEL["use_id_btn"]).nth(seat_idx).click(timeout=4000)
            box = page.locator(SEL["id_input"]).last
            box.fill(bot_id)
            page.locator(SEL["add_btn"]).last.click(timeout=4000)
            page.wait_for_timeout(800)
            wait_for_captcha(page)
            # cooldown detection
            if page.locator(SEL["cooldown_txt"]).count():
                log(f"⏳ add-bot cooldown — backing off {COOLDOWN_WAIT}s (NOT evading; just waiting)…")
                time.sleep(COOLDOWN_WAIT); continue
            log(f"  seat {seat_idx}: {label} added.")
            return True
        except Exception as e:
            log(f"  seat {seat_idx} add retry ({str(e)[:50]})"); page.wait_for_timeout(1500)
    return False


def grab_log(page, out_dir, n):
    """Switch to debug mode, open the log tool, copy the JSON, save it."""
    try:
        if page.locator(SEL["debug_toggle"]).count():
            page.locator(SEL["debug_toggle"]).first.click(); page.wait_for_timeout(800)
        page.locator(SEL["log_tool"]).first.click(); page.wait_for_timeout(1200)
        # the log tool dumps JSON into a textarea / pre — grab the largest blob on the page
        blobs = page.locator("textarea, pre").all_inner_texts()
        js = max((b for b in blobs if b.strip().startswith(("{", "["))), key=len, default="")
        if js:
            p = pathlib.Path(out_dir) / f"manual_{int(time.time())}_{n}.json"
            p.write_text(js); log(f"  saved log -> {p.name}"); return True
    except Exception as e:
        log(f"  log grab failed ({str(e)[:60]}) — the hourly collector will still catch this game.")
    return False


def create_one_game(page, out_dir, n):
    page.goto(BASE)
    page.locator(SEL["create_table_link"]).first.click(); page.wait_for_timeout(1500)
    try: page.locator(SEL["game_select"]).first.select_option(label=re.compile("Mahjong"))
    except Exception:
        try: page.locator(SEL["game_select"]).first.select_option(value=GAME)
        except Exception: log("  ⚠ select the game type manually if not auto-selected")
    page.locator(SEL["create_btn"]).first.click(); page.wait_for_timeout(2000)
    for i, (label, bid) in enumerate(BOTS):
        if not add_bot_by_id(page, i, label, bid):
            log(f"  could not add seat {i} ({label}); skipping this table."); return False
    page.locator(SEL["start_btn"]).first.click(); page.wait_for_timeout(1500)
    wait_for_captcha(page)
    log(f"game {n}: started; waiting for it to finish…")
    # wait for the game to end (log tool / debug toggle appears)
    for _ in range(120):
        if page.locator(SEL["log_tool"]).count() or page.locator(SEL["debug_toggle"]).count():
            break
        page.wait_for_timeout(2000)
    grab_log(page, out_dir, n)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--out", default=os.path.expanduser("~/botzone_logs"))
    a = ap.parse_args()
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)        # HEADED so you can solve captchas
        page = browser.new_context().new_page()
        login(page)
        ok = 0
        for n in range(1, a.games + 1):
            try:
                if create_one_game(page, a.out, n): ok += 1
            except Exception as e:
                log(f"game {n} errored ({str(e)[:80]}); continuing."); page.wait_for_timeout(3000)
            log(f"progress: {ok}/{n} created (target {a.games})")
        log(f"DONE: {ok} games created -> {a.out}. scp them to research-os "
            f"~/IJCAI-mahjong/others/ladder_top30_score1216/future_hourly/exactly3_top30/logs/ "
            f"(or just let the hourly collector harvest them).")
        browser.close()


if __name__ == "__main__":
    main()
