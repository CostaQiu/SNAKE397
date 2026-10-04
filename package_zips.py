"""Build the zip files to share with friends (no Python needed on their side).

    python package_zips.py            # both
    python package_zips.py cartoon    # only the cartoon one
    python package_zips.py classic    # only the classic one

Output: dist/SnakeScreensaver-classic.zip and dist/SnakeScreensaver-cartoon.zip
"""

import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
CARTOON = HERE / "cartoon_screensaver"

CLASSIC_README = """SNAKE 397 SCREENSAVER, classic green retro-monitor look (Windows 10/11, no Python, no GPU needed, works offline)
=====================================================================================================
Install:   double-click install_screensaver.bat       (optional: install_screensaver.bat 10  = 10 idle minutes, default 5)
Preview:   %LOCALAPPDATA%\\SnakeSaver\\app\\SnakeSaver.scr /s     (any key / mouse move quits)
Settings:  Settings > Personalization > Lock screen > Screen saver
Remove:    double-click uninstall_screensaver.bat

It replays recorded perfect games (397 points) of a snake AI, one full-screen window per monitor, all monitors in sync.
Needs Microsoft Edge (already part of Windows). SmartScreen may warn about SnakeSaver.scr because it is not code-signed.
Source code: https://github.com/CostaQiu/SNAKE397

中文说明
--------
安装：双击 install_screensaver.bat（可加参数设置闲置分钟数，如 install_screensaver.bat 10，默认 5 分钟）
预览：%LOCALAPPDATA%\\SnakeSaver\\app\\SnakeSaver.scr /s （按任意键或动鼠标退出）
修改/关闭：设置 > 个性化 > 锁屏界面 > 屏幕保护程序
卸载：双击 uninstall_screensaver.bat
内容：回放贪吃蛇 AI 录制的 397 分满分对局，多块显示器同步显示。需要 Edge 浏览器（Windows 自带），不需要 Python，也不需要显卡。
因为没有代码签名，Windows 可能弹出 SmartScreen 警告。
"""

CARTOON_README = """SNAKE 397 SCREENSAVER, cartoon look (Windows 10/11, no Python, no GPU needed, works offline)
==========================================================================================
Install:   double-click install_cartoon.bat           (optional: install_cartoon.bat 10  = 10 idle minutes, default 5)
Preview:   %LOCALAPPDATA%\\SnakeSaver\\app\\SnakeCartoon.scr /s     (any key / mouse move quits)
Settings:  Settings > Personalization > Lock screen > Screen saver
Remove:    double-click uninstall_cartoon.bat

It replays recorded perfect games (397 points) of a snake AI, one full-screen window per monitor, all monitors in sync.
Plain 2D drawing (no 3D, no internet), kept light on the graphics card. The body has three colours in blocks of 15 cells.
Needs Microsoft Edge (already part of Windows). SmartScreen may warn about SnakeCartoon.scr because it is not code-signed.
Source code: https://github.com/CostaQiu/SNAKE397

中文说明
--------
安装：双击 install_cartoon.bat（可加参数设置闲置分钟数，如 install_cartoon.bat 10，默认 5 分钟）
预览：%LOCALAPPDATA%\\SnakeSaver\\app\\SnakeCartoon.scr /s （按任意键或动鼠标退出）
修改/关闭：设置 > 个性化 > 锁屏界面 > 屏幕保护程序
卸载：双击 uninstall_cartoon.bat
内容：回放贪吃蛇 AI 录制的 397 分满分对局，多块显示器同步显示。2D 卡通画面，不用 3D、不用联网，尽量减轻显卡压力；
蛇身 3 种颜色，每种连续 15 格。需要 Edge 浏览器（Windows 自带），不需要 Python，也不需要显卡。
因为没有代码签名，Windows 可能弹出 SmartScreen 警告。
"""


def write_zip(out, root_name, files, readme):
    """files maps a path inside the zip folder to a source file; readme is added as README.txt (CRLF line endings)."""
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dst, src in files.items():
            if not src.exists():
                raise SystemExit(f"missing file: {src}")
            z.write(src, f"{root_name}/{dst}")
        z.writestr(f"{root_name}/README.txt", readme.replace("\n", "\r\n"))
    print(f"{out}: {out.stat().st_size / 1024:.0f} KB")


def main():
    """Refresh the cartoon folder's replay copy, then build the requested zips."""
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    shared = HERE / "replays_wins.js"
    if shared.read_bytes() != (CARTOON / "replays_wins.js").read_bytes():
        shutil.copyfile(shared, CARTOON / "replays_wins.js")
        print("refreshed cartoon_screensaver/replays_wins.js")
    if which in ("all", "classic"):
        write_zip(
            DIST / "SnakeScreensaver-classic.zip",
            "SnakeScreensaver-classic",
            {
                "install_screensaver.bat": HERE / "install_screensaver.bat",
                "uninstall_screensaver.bat": HERE / "uninstall_screensaver.bat",
                "snake_classic.html": HERE / "snake_classic.html",
                "replays_wins.js": shared,
                "screensaver/SnakeSaver.scr": HERE / "screensaver" / "SnakeSaver.scr",
            },
            CLASSIC_README,
        )
    if which in ("all", "cartoon"):
        write_zip(
            DIST / "SnakeScreensaver-cartoon.zip",
            "SnakeScreensaver-cartoon",
            {
                name: CARTOON / name
                for name in (
                    "install_cartoon.bat",
                    "uninstall_cartoon.bat",
                    "SnakeCartoon.scr",
                    "snake_cartoon.html",
                    "replays_wins.js",
                )
            },
            CARTOON_README,
        )


if __name__ == "__main__":
    main()
