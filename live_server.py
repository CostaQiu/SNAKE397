"""Live server: the AI plays in Python while snake3d.html shows it in real time.

Every request for moves runs the trained network plus the food-directed search
on the spot (nothing is pre-recorded). The same server also serves the static
files, so open  http://127.0.0.1:8770/snake3d.html#live  after starting it.

Usage:
    python live_server.py [--port 8770] [--model rl_snake_best.pth]

API (JSON, all GET):
    /api/ping                      -> {"ok": true, ...}
    /api/new?seed=123              -> {"sid", "seed", "start", "foods"}
    /api/step?sid=...&n=100        -> {"moves", "foods", "done", "result", "score", "total", "ms"}
"""

import argparse
import json
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from rl_search import choose_action_goal
from rl_train import QNet
from snake_env import ACTION_TURN, SnakeEnv

HERE = Path(__file__).resolve().parent
STUCK_AFTER = 300  # moves without food before a live game is called stuck
MAX_SESSIONS = 20
MAX_BATCH = 6000

NET = None
MODEL_NAME = ""
SESSIONS = {}
LOCK = threading.Lock()


def load_model(path):
    """Load the trained Q-network onto the CPU (single-state inference)."""
    global NET, MODEL_NAME
    ck = torch.load(path, map_location="cpu")
    net = QNet(n_in=ck["n_in"])
    net.load_state_dict(ck["model"])
    net.eval()
    NET, MODEL_NAME = net, Path(path).name


class Session:
    """One live game: environment plus bookkeeping for the replay protocol."""

    def __init__(self, seed):
        self.seed = seed
        self.env = SnakeEnv(seed=seed)
        self.env.timeout_limit = lambda: 3000
        self.state = self.env.state()
        self.start = [list(c) for c in self.env.snake]
        self.first_food = list(self.env.food)
        self.moves_made = 0
        self.last_food_at = 0
        self.done = False
        self.result = None
        self.lock = threading.Lock()

    def advance(self, n):
        """Play up to n moves; returns (moves string, new foods, ms spent)."""
        t0 = time.perf_counter()
        env, moves, foods = self.env, [], []
        while len(moves) < n and not self.done:
            with torch.no_grad():
                q = NET(torch.as_tensor(self.state).unsqueeze(0))[0].numpy()
            action = choose_action_goal(env, q, self.state)
            heading = (env.d + ACTION_TURN[action]) % 4
            before = env.score
            _, finished = env.step(action)
            moves.append(heading)
            self.moves_made += 1
            if env.score > before:
                self.last_food_at = self.moves_made
                if not env.won:
                    foods.append(list(env.food))
            if finished:
                self.done = True
                self.result = "win" if env.won else "dead"
            elif self.moves_made - self.last_food_at > STUCK_AFTER:
                self.done, self.result = True, "stuck"
            else:
                self.state = env.state()
        ms = (time.perf_counter() - t0) * 1000
        return "".join(map(str, moves)), foods, ms


class Handler(SimpleHTTPRequestHandler):
    """Static files from the project folder plus the /api/* endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/ping":
                return self._json({"ok": True, "model": MODEL_NAME})
            if url.path == "/api/new":
                seed = int(q.get("seed", int(time.time()) % 1_000_000))
                s = Session(seed)
                sid = uuid.uuid4().hex[:12]
                with LOCK:
                    SESSIONS[sid] = s
                    while len(SESSIONS) > MAX_SESSIONS:
                        SESSIONS.pop(next(iter(SESSIONS)))
                return self._json(
                    {
                        "sid": sid,
                        "seed": seed,
                        "start": s.start,
                        "foods": [s.first_food],
                    }
                )
            if url.path == "/api/step":
                s = SESSIONS.get(q.get("sid", ""))
                if s is None:
                    return self._json({"error": "unknown session"}, 404)
                n = max(1, min(MAX_BATCH, int(q.get("n", 100))))
                with s.lock:
                    moves, foods, ms = s.advance(n)
                    return self._json(
                        {
                            "moves": moves,
                            "foods": foods,
                            "done": s.done,
                            "result": s.result,
                            "score": s.env.score,
                            "total": s.moves_made,
                            "ms": round(ms, 1),
                        }
                    )
            return self._json({"error": "not found"}, 404)
        except (
            Exception
        ) as exc:  # report to the page instead of dropping the connection
            return self._json({"error": repr(exc)}, 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model", default=str(HERE / "rl_snake_best.pth"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    load_model(args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"live server ready: http://127.0.0.1:{args.port}/snake3d.html#live "
        f"(model {MODEL_NAME})",
        flush=True,
    )
    server.serve_forever()
