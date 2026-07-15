"""Eyeball-loop tooling (docs/09 §4): screenshot a live page via headless Chrome.

Usage: python scene_shot.py <url> <out.png> [wait_seconds]
"""
import sys

from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
wait = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

with sync_playwright() as p:
    b = p.chromium.launch(channel="chromium", args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--no-sandbox"])
    page = b.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(int(wait * 1000))
    page.screenshot(path=out)
    b.close()
    print("JS_ERRORS:", errors[:3] if errors else "none")
    print("saved", out)
