"""Build screensaver/SnakeSaver.scr with the C# compiler that ships with Windows (no downloads)."""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "screensaver" / "SnakeSaver.cs"
OUT = HERE / "screensaver" / "SnakeSaver.scr"
CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")


def main():
    """Substitute the project path into the source and compile it."""
    if not CSC.exists():
        raise SystemExit(f"C# compiler not found: {CSC}")
    text = SRC.read_text(encoding="utf-8").replace("@PROJECT_DIR@", str(HERE))
    built = HERE / "screensaver" / "_SnakeSaver.build.cs"
    built.write_text(text, encoding="utf-8")
    cmd = [str(CSC), "/nologo", "/target:winexe", f"/out:{OUT}", "/r:System.Windows.Forms.dll",
           "/r:System.Drawing.dll", "/r:System.Management.dll", str(built)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    built.unlink(missing_ok=True)
    print(result.stdout + result.stderr)
    if result.returncode != 0:
        raise SystemExit("build failed")
    print(f"built {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
