"""Record AI games (network + food-directed search) for the 3D viewer.

Each game is stored as a compact replay: the start body, the food positions in
the order they appear, and one absolute heading per move (0=up, 1=right,
2=down, 3=left). Output goes to replays.js so snake3d.html also works when
opened straight from disk.

Usage:
    python record_games.py [--games 32] [--keep 8]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from rl_search import choose_action_goal
from rl_train import QNet
from snake_env import ACTION_TURN, SnakeEnv

HERE = Path(__file__).resolve().parent
STUCK_AFTER = 500  # moves without food before we call the game stuck
STUCK_TAIL = 120  # moves of circling kept after the last food, for the replay


def record(args):
    """Play one game with the search policy and return its replay dict."""
    seed, model_path = args
    torch.set_num_threads(1)
    ck = torch.load(model_path, map_location="cpu")
    net = QNet(n_in=ck["n_in"])
    net.load_state_dict(ck["model"])
    net.eval()

    env = SnakeEnv(seed=seed)
    env.timeout_limit = lambda: 3000
    state = env.state()
    start = [list(c) for c in env.snake]
    foods = [list(env.food)]
    moves = []
    last_food_at = 0
    result = "stuck"
    while True:
        with torch.no_grad():
            q = net(torch.as_tensor(state).unsqueeze(0))[0].numpy()
        action = choose_action_goal(env, q, state)
        heading = (env.d + ACTION_TURN[action]) % 4
        before = env.score
        _, done = env.step(action)
        moves.append(heading)
        if env.score > before:
            last_food_at = len(moves)
            if not env.won:
                foods.append(list(env.food))
        if done:
            result = "win" if env.won else "dead"
            break
        if len(moves) - last_food_at > STUCK_AFTER:
            moves = moves[: last_food_at + STUCK_TAIL]
            break
        state = env.state()
    return {
        "seed": seed,
        "score": env.score,
        "result": result,
        "start": start,
        "foods": foods,
        "moves": "".join(str(m) for m in moves),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(HERE / "rl_snake_best.pth"))
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--seed0", type=int, default=5000)
    parser.add_argument(
        "--wins",
        type=int,
        default=0,
        help="keep recording until this many perfect games exist (wins only)",
    )
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--out", default=str(HERE / "replays.js"))
    args = parser.parse_args()

    if args.wins:
        # Perfect games are rare (~3%), so record in batches until enough exist.
        found, tried, seed = [], 0, args.seed0
        with Pool(args.procs) as pool:
            while len(found) < args.wins and tried < args.max_games:
                batch = [(seed + i, args.model) for i in range(args.procs * 4)]
                seed += len(batch)
                results = pool.map(record, batch, chunksize=1)
                tried += len(results)
                found += [g for g in results if g["result"] == "win"]
                print(
                    f"tried {tried} games, perfect so far {len(found)}/{args.wins}",
                    flush=True,
                )
        kept = found[: args.wins]
        print(f"perfect games: {len(kept)} out of {tried} tried")
        for g in kept:
            print(f"  seed {g['seed']}: score {g['score']} ({len(g['moves'])} moves)")
        out = Path(args.out)
        out.write_text(
            "window.REPLAYS_WINS = " + json.dumps(kept) + ";" + chr(10),
            encoding="utf-8",
        )
        print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
        raise SystemExit(0 if len(kept) == args.wins else 1)

    jobs = [(args.seed0 + i, args.model) for i in range(args.games)]
    with Pool(args.procs) as pool:
        games = pool.map(record, jobs, chunksize=1)

    # Perfect games first, then the rest in seed order (not cherry-picked).
    wins = [g for g in games if g["result"] == "win"]
    rest = [g for g in games if g["result"] != "win"]
    kept = (wins + rest)[: args.keep]
    scores = [g["score"] for g in games]
    print(
        f"recorded {len(games)} games: mean {np.mean(scores):.1f}, "
        f"wins {len(wins)}, kept {len(kept)}"
    )
    for g in kept:
        print(
            f"  seed {g['seed']}: score {g['score']} {g['result']} "
            f"({len(g['moves'])} moves)"
        )
    out = Path(args.out)
    out.write_text("window.REPLAYS = " + json.dumps(kept) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
