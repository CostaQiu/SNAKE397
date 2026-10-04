// Windows screensaver launcher for the 3D snake page.
// /s  = run: one full-screen Edge window per monitor, quit on any keyboard/mouse input
// /p  = preview pane (not supported, exits)   /c = settings (shows a short info box)
// /s /test N = run for N seconds and ignore input (used for automated checks)
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;

static class SnakeSaver
{
    // The file name picks the look: SnakeCartoon.scr shows the 2D cartoon page, anything else the classic green page.
    static bool Cartoon
    {
        get { return Path.GetFileNameWithoutExtension(Process.GetCurrentProcess().MainModule.FileName).IndexOf("Cartoon", StringComparison.OrdinalIgnoreCase) >= 0; }
    }
    static string PageName { get { return Cartoon ? "snake_cartoon.html" : "snake_classic.html"; } }

    // Page + replays are looked up next to the .scr first (so a copied folder works anywhere); the compiled-in path is the fallback.
    static string ProjectDir
    {
        get
        {
            string here = AppDomain.CurrentDomain.BaseDirectory;
            if (File.Exists(Path.Combine(here, PageName))) return here;      // installed copy: all files in one folder
            string up = Path.GetFullPath(Path.Combine(here, ".."));
            if (File.Exists(Path.Combine(up, PageName))) return up;          // repo / zip layout: the .scr sits in a subfolder
            return @"@PROJECT_DIR@";
        }
    }

    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] static extern bool SetProcessDpiAwarenessContext(IntPtr value);
    [StructLayout(LayoutKind.Sequential)] struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

    delegate bool EnumProc(IntPtr h, IntPtr l);
    [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc p, IntPtr l);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr h, int index);
    const int GWL_EXSTYLE = -20, WS_EX_TOPMOST = 0x8;
    static readonly bool NoTop = Environment.GetEnvironmentVariable("SNAKESAVER_NOTOP") == "1";  // diagnostics only
    // raise a window once; calling SetWindowPos again on every scan would make the browser re-composite it each time
    static void RaiseOnce(IntPtr h)
    {
        if (NoTop) return;
        if ((GetWindowLong(h, GWL_EXSTYLE) & WS_EX_TOPMOST) != 0) return;
        SetWindowPos(h, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }
    static readonly HashSet<uint> OwnPids = new HashSet<uint>();  // browser processes this launcher started
    static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
    const uint SWP_NOSIZE = 0x1, SWP_NOMOVE = 0x2, SWP_NOACTIVATE = 0x10;

    static readonly Regex TitleRe = new Regex(@"SnakeScr slot=(\d+) seed=(\d+) step=(\d+)(?: audio=(\w+))?");
    static readonly Dictionary<int, string> AudioState = new Dictionary<int, string>();

    // The page writes "slot/seed/step" into its window title every second; read the latest values.
    static void ScanTitles(Dictionary<int, string> progress)
    {
        EnumWindows((h, l) =>
        {
            if (!IsWindowVisible(h)) return true;
            // Our own windows go on top no matter what their title says: a window hidden behind another one
            // stops running its page (the browser pauses occluded windows), so the title can never be read.
            uint wpid; GetWindowThreadProcessId(h, out wpid);
            if (OwnPids.Contains(wpid)) RaiseOnce(h);
            var sb = new StringBuilder(300);
            GetWindowText(h, sb, 300);
            var m = TitleRe.Match(sb.ToString());
            if (m.Success)
            {
                progress[int.Parse(m.Groups[1].Value)] = m.Groups[2].Value + " " + m.Groups[3].Value;
                AudioState[int.Parse(m.Groups[1].Value)] = m.Groups[4].Success ? m.Groups[4].Value : "n/a";
                RaiseOnce(h);  // stay above other windows
            }
            return true;
        }, IntPtr.Zero);
    }

    static void SaveProgress(string file, Dictionary<int, string> progress)
    {
        try { File.WriteAllLines(file, progress.Select(kv => kv.Key + " " + kv.Value).ToArray()); } catch { }
    }

    static Dictionary<int, string> LoadProgress(string file)
    {
        var d = new Dictionary<int, string>();
        try
        {
            foreach (var line in File.ReadAllLines(file))
            {
                var parts = line.Split(' ');
                if (parts.Length == 3) d[int.Parse(parts[0])] = parts[1] + " " + parts[2];
            }
        }
        catch { }
        return d;
    }

    static string LogFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SnakeSaver", "launcher.log");
    static void Log(string msg)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogFile));
            if (File.Exists(LogFile) && new FileInfo(LogFile).Length > 200000) File.Delete(LogFile);
            File.AppendAllText(LogFile, DateTime.Now.ToString("HH:mm:ss.fff") + "  " + msg + Environment.NewLine);
        }
        catch { }
    }

    static string HeartFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SnakeSaver", "heartbeat.txt");
    static void Heartbeat(string what)
    {
        try { File.WriteAllText(HeartFile, DateTime.Now.ToString("HH:mm:ss.fff") + "  " + what + Environment.NewLine); } catch { }
    }

    static uint LastInput()
    {
        var info = new LASTINPUTINFO();
        info.cbSize = (uint)Marshal.SizeOf(typeof(LASTINPUTINFO));
        GetLastInputInfo(ref info);
        return info.dwTime;
    }

    static string FindEdge()
    {
        string[] candidates = {
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), @"Microsoft\Edge\Application\msedge.exe"),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), @"Microsoft\Edge\Application\msedge.exe"),
        };
        return candidates.FirstOrDefault(File.Exists);
    }

    static void KillTree(int pid)
    {
        try
        {
            var p = Process.Start(new ProcessStartInfo("taskkill", "/PID " + pid + " /T /F") { CreateNoWindow = true, UseShellExecute = false });
            if (p != null) p.WaitForExit(3000);
        }
        catch { }
    }

    // Edge may leave helper processes behind; remove everything that uses our profile folders.
    static void CleanupLeftovers()
    {
        try
        {
            using (var q = new ManagementObjectSearcher("SELECT ProcessId FROM Win32_Process WHERE Name='msedge.exe' AND CommandLine LIKE '%SnakeSaver%'"))
                foreach (ManagementObject o in q.Get()) KillTree((int)(uint)o["ProcessId"]);
        }
        catch { }
    }

    [STAThread]
    static int Main(string[] args)
    {
        try { if (!SetProcessDpiAwarenessContext(new IntPtr(-4))) SetProcessDPIAware(); }
        catch { try { SetProcessDPIAware(); } catch { } }

        string first = args.Length > 0 ? args[0].ToLowerInvariant().TrimStart('/', '-') : "c";
        char mode = first.Length > 0 ? first[0] : 'c';
        if (mode == 'p') return 0;
        if (mode != 's')
        {
            MessageBox.Show("Snake screensaver (perfect 397-point games).\nThere is nothing to configure here.\nChange the idle time in Windows Settings > Personalization > Lock screen > Screen saver.",
                "Snake Screensaver", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return 0;
        }

        int testSeconds = 0;
        int ti = Array.FindIndex(args, a => a.ToLowerInvariant() == "/test");
        if (ti >= 0 && ti + 1 < args.Length) int.TryParse(args[ti + 1], out testSeconds);

        Heartbeat("process started, args=" + string.Join(" ", args) + " pid=" + Process.GetCurrentProcess().Id);
        bool created;
        using (var mutex = new Mutex(true, "SnakeSaverSingleton", out created))
        {
            if (!created) { Heartbeat("another instance owns the mutex, exiting"); return 0; }
            Heartbeat("mutex acquired");

            string edge = FindEdge();
            if (edge == null) { MessageBox.Show("Microsoft Edge was not found.", "Snake Screensaver"); return 1; }
            string pageFile = Path.Combine(ProjectDir, PageName);
            if (!File.Exists(pageFile)) { MessageBox.Show("Missing file: " + pageFile, "Snake Screensaver"); return 1; }

            // left-to-right order keeps each screen's saved progress attached to the same physical position
            var screens = Screen.AllScreens.OrderBy(s => s.Bounds.X).ToArray();
            string profiles = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SnakeSaver");
            string progressFile = Path.Combine(profiles, "progress.txt");
            Directory.CreateDirectory(profiles);
            var progress = LoadProgress(progressFile);
            string[] resume = null;  // all screens show one game, so any saved entry will do
            foreach (var kv in progress.OrderBy(k => k.Key)) { resume = kv.Value.Split(' '); break; }
            long t0Ms = (long)(DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds + 4000;
            var procs = new List<Process>();
            for (int i = 0; i < screens.Length; i++)
            {
                var b = screens[i].Bounds;
                string profile = Path.Combine(profiles, "slot" + i);
                Directory.CreateDirectory(profile);
                string url = new Uri(pageFile).AbsoluteUri + "#saver&host=scr&slot=" + i;
                string extra = Environment.GetEnvironmentVariable("SNAKESAVER_EXTRA");  // test hook, e.g. "&fps=0&pr=0"
                if (!string.IsNullOrEmpty(extra)) url += extra;
                url += "&t0=" + t0Ms;                       // shared clock: all screens show the same frame
                if (screens[i].Primary) url += "&snd=1";    // only the primary screen makes sound
                if (resume != null) url += "&seed=" + resume[0] + "&step=" + resume[1];  // everybody continues the same game
                string argLine = string.Format(
                    "--kiosk \"{0}\" --edge-kiosk-type=fullscreen --no-first-run --no-default-browser-check " +
                    "--disable-session-crashed-bubble --noerrdialogs --autoplay-policy=no-user-gesture-required " +
                    "--lang=en-US --disable-features=Translate,TranslateUI,msEdgeTranslate " +
                    "--user-data-dir=\"{1}\" --window-position={2},{3}",
                    url, profile, b.X + 100, b.Y + 100);
                string edgeFlags = Environment.GetEnvironmentVariable("SNAKESAVER_EDGEFLAGS");  // test hook for extra browser flags
                if (!string.IsNullOrEmpty(edgeFlags)) argLine += " " + edgeFlags;
                try { var pr = Process.Start(new ProcessStartInfo(edge, argLine) { UseShellExecute = false }); procs.Add(pr); if (pr != null) OwnPids.Add((uint)pr.Id); }
                catch { }
            }

            Heartbeat("edge windows launched: " + procs.Count);
            Log("started, mode=s test=" + testSeconds + ", launched " + procs.Count + " edge windows");
            string exitReason = "?";
            try
            {
                // wait for the first input after a short grace period (the user may still be touching the mouse)
                for (int w = 0; w < 10; w++) { Thread.Sleep(250); ScanTitles(progress); }  // lift the new windows to the top right away
                uint baseline = LastInput();
                var started = DateTime.UtcNow;
                var lastScan = DateTime.UtcNow;
                bool audioLogged = false;
                while (true)
                {
                    Thread.Sleep(100);
                    if ((DateTime.UtcNow - lastScan).TotalSeconds >= 1)
                    {
                        lastScan = DateTime.UtcNow;
                        Heartbeat("main loop alive, windows titled so far: " + progress.Count + ", edge procs: " + procs.Count);
                        ScanTitles(progress);
                        SaveProgress(progressFile, progress);
                        if (!audioLogged && (DateTime.UtcNow - started).TotalSeconds >= 5)
                        {
                            audioLogged = true;
                            Log("audio state per screen: " + string.Join("; ", AudioState.OrderBy(k => k.Key).Select(kv => "slot" + kv.Key + "=" + kv.Value).ToArray()));
                        }
                    }
                    if (testSeconds > 0)
                    {
                        if ((DateTime.UtcNow - started).TotalSeconds >= testSeconds) { exitReason = "test time over"; break; }
                        continue;
                    }
                    if (LastInput() != baseline) { exitReason = "user input"; break; }
                }
            }
            catch (Exception ex) { exitReason = "EXCEPTION " + ex.Message; }
            finally
            {
                ScanTitles(progress);
                SaveProgress(progressFile, progress);
                foreach (var p in procs) { try { if (p != null) KillTree(p.Id); } catch { } }
                CleanupLeftovers();
                Log("exit (" + exitReason + "), progress: " + string.Join("; ", progress.Select(kv => kv.Key + "=" + kv.Value).ToArray()));
            }
        }
        return 0;
    }
}
