"""Render assets/poster.html to assets/poster.png (1280x640, 2x sharp).

One-time setup:
    pip install playwright
    playwright install chromium

Run from the repo root:
    python tools/render_poster.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "poster.html"
OUT = ROOT / "assets" / "poster.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 640}, device_scale_factor=2)
    page.goto(SRC.as_uri())
    page.wait_for_timeout(300)  # let fonts and gradients settle
    page.screenshot(path=str(OUT))
    browser.close()

print(f"saved {OUT}")
