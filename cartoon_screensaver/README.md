# Cartoon screensaver

The cartoon look of the Snake 397 screensaver. Plain 2D canvas, no 3D, no internet, no Python needed on the viewer's side.

| File | What it is |
|---|---|
| `snake_cartoon.html` | The page: sky, clouds, wooden frame, checkered lawn, cartoon snake and apple, numbers |
| `replays_wins.js` | The recorded 397-point games it replays (a copy of the one in the repository root) |
| `SnakeCartoon.scr` | The Windows screensaver launcher (built from `../screensaver/SnakeSaver.cs`, the file name selects this page) |
| `install_cartoon.bat` / `uninstall_cartoon.bat` | Per-user install and removal (no admin rights) |

- **Install:** double-click `install_cartoon.bat` (or `install_cartoon.bat 10` for 10 idle minutes).
- **Body colours:** exactly four (`BODY_COLORS` in `snake_cartoon.html`), 10 cells each, counted from the head and repeating (`STRIPE`).
- **Speed:** 16 steps per second; change the default in `Q("sps", 16)`.
- **Numbers:** same text and layout as the green version (SCORE, STEPS, GAME TIME, clock and date).
- **Share it:** from the repository root run `python package_zips.py cartoon`, then send `dist/SnakeScreensaver-cartoon.zip`.
- **Rebuild the launcher:** `python build_screensaver.py` from the repository root.
