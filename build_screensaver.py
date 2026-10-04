"""Build screensaver/SnakeSaver.scr (classic green) and cartoon_screensaver/SnakeCartoon.scr (2D cartoon) with the C# compiler
that ships with Windows (no downloads). Both come from the same source; the file name decides which page is shown."""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "screensaver" / "SnakeSaver.cs"
OUTS = [HERE / "screensaver" / "SnakeSaver.scr", HERE / "cartoon_screensaver" / "SnakeCartoon.scr"]
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")


def main():
    """Substitute the project path into the source and compile it once per output name."""
    if not CSC.exists():
        raise SystemExit(f"C# compiler not found: {CSC}")
    text = SRC.read_text(encoding="utf-8").replace("@PROJECT_DIR@", str(HERE))
    built = HERE / "screensaver" / "_SnakeSaver.build.cs"
    built.write_text(text, encoding="utf-8")
    try:
        for out in OUTS:
            cmd = [str(CSC), "/nologo", "/target:winexe", f"/out:{out}", "/r:System.Windows.Forms.dll",
                   "/r:System.Drawing.dll", "/r:System.Management.dll", str(built)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout + result.stderr)
            if result.returncode != 0:
                raise SystemExit(f"build failed: {out.name}")
            print(f"built {out} ({out.stat().st_size / 1024:.0f} KB)")
    finally:
        built.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
