"""Build self-contained single-file versions of the screensavers (page + recorded games in one .html) to share.

    python build_single_html.py

Output: dist/snake397-cartoon.html and dist/snake397-classic.html. They need no other file, no internet and no Python:
open them in any modern browser (phone included). Tap toggles sound; the screen is kept awake where the browser allows it.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
REPLAYS = HERE / "replays_wins.js"
PAGES = {
    "snake397-cartoon.html": HERE / "cartoon_screensaver" / "snake_cartoon.html",
    "snake397-classic.html": HERE / "snake_classic.html",
}
TAG = '<script src="replays_wins.js"></script>'


def main():
    """Replace the external replay <script> with the file's content inline and write the results to dist/."""
    data = REPLAYS.read_text(encoding="utf-8")
    if "</script" in data.lower():
        raise SystemExit("replays_wins.js contains '</script', cannot inline it safely")
    DIST.mkdir(exist_ok=True)
    for name, page in PAGES.items():
        html = page.read_text(encoding="utf-8")
        if html.count(TAG) != 1:
            raise SystemExit(f"{page.name}: expected exactly one {TAG}")
        out = DIST / name
        out.write_text(
            html.replace(TAG, "<script>\n" + data.strip() + "\n</script>"),
            encoding="utf-8",
        )
        print(f"{out}: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
