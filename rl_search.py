"""Lookahead search on top of the trained Q-network.

The network ranks the three moves. Before playing the best-ranked one we check,
with a depth-limited search on a copy of the board, that the snake can keep
playing for `depth` more moves while the tail stays reachable after every move.
If the move fails that check we try the next-best one. Future food positions
are unknown, so eating the food ends that branch (counted as a success if the
tail is still reachable afterwards).

Usage:
    python rl_search.py --depths 0 15 40 --games 32
"""

import argparse
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from rl_train import QNet
from snake_env import SnakeEnv, _flood, _is_blocked, DX, DY, ACTION_TURN

HERE = Path(__file__).resolve().parent
DXL = [int(v) for v in DX]
DYL = [int(v) for v in DY]


def _dfs(occ, body, d, depth, food, w, h, budget, hole):
    """True if some line of `depth` moves keeps the tail reachable each step.

    Mutates occ/body with make/unmake and restores them before returning.
    budget[0] counts remaining nodes; when it runs out we optimistically say True.
    """
    if budget[0] <= 0:
        return True
    hx, hy = body[0]
    tx, ty = body[-1]
    for turn in ACTION_TURN:
        nd = (d + turn) % 4
        nx, ny = hx + DXL[nd], hy + DYL[nd]
        eating = (nx, ny) == food
        if _is_blocked(occ, nx, ny, tx, ty, eating, w, h):
            continue
        budget[0] -= 1
        if not eating:
            body.pop()
            occ[ty, tx] = 0
        body.appendleft((nx, ny))
        occ[ny, nx] = 1
        ntx, nty = body[-1]
        area, tail_r, _ = _flood(occ, nx, ny, ntx, nty, food[0], food[1], w, h)
        ok = tail_r == 1
        if ok and len(body) >= hole and area != w * h - len(body):
            ok = False  # a free cell got sealed off from the head
        if ok and not (eating or depth <= 1):
            ok = _dfs(occ, body, nd, depth - 1, food, w, h, budget, hole)
        body.popleft()
        occ[ny, nx] = 0
        if not eating:
            body.append((tx, ty))
            occ[ty, tx] = 1
        if ok:
            return True
    return False


def move_viable(env, action, depth, budget=3000, hole=10**9):
    """Does playing `action` now leave a safe line of `depth` further moves?"""
    d = (env.d + ACTION_TURN[action]) % 4
    hx, hy = env.snake[0]
    tx, ty = env.snake[-1]
    nx, ny = hx + DXL[d], hy + DYL[d]
    eating = (nx, ny) == env.food
    if _is_blocked(env.occ, nx, ny, tx, ty, eating, env.w, env.h):
        return False
    occ = env.occ.copy()
    body = deque(env.snake)
    if not eating:
        body.pop()
        occ[ty, tx] = 0
    body.appendleft((nx, ny))
    occ[ny, nx] = 1
    ntx, nty = body[-1]
    area, tail_r, _ = _flood(
        occ, nx, ny, ntx, nty, env.food[0], env.food[1], env.w, env.h
    )
    if tail_r != 1:
        return False
    if len(body) >= hole and area != env.w * env.h - len(body):
        return False
    if eating or depth <= 1:
        return True
    return _dfs(occ, body, d, depth - 1, env.food, env.w, env.h, [budget], hole)


def goal_viable(env, action, hole=10**9):
    """Can the snake, starting with `action`, eat the food and stay healthy?

    Finds the shortest path to the food that respects the moving body (a body
    cell is free once enough steps have passed for the tail to vacate it),
    replays it on a copy of the board, and then requires that the tail is still
    reachable and, once the body has `hole` cells or more, that no free cell
    was sealed off from the head.
    """
    w, h = env.w, env.h
    body = list(env.snake)
    length = len(body)
    d = (env.d + ACTION_TURN[action]) % 4
    hx, hy = body[0]
    sx, sy = hx + DXL[d], hy + DYL[d]
    if not (0 <= sx < w and 0 <= sy < h):
        return False
    # block[y][x] = first step at which the cell may be entered
    block = {(x, y): length - i for i, (x, y) in enumerate(body)}
    if block.get((sx, sy), 0) > 1:
        return False
    food = env.food
    parent = {(sx, sy): None}
    frontier = deque([((sx, sy), 1)])
    end = (sx, sy) if (sx, sy) == food else None
    while frontier and end is None:
        (cx, cy), t = frontier.popleft()
        for k in range(4):
            nx, ny = cx + DXL[k], cy + DYL[k]
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in parent:
                continue
            if block.get((nx, ny), 0) > t + 1:
                continue
            parent[(nx, ny)] = (cx, cy)
            if (nx, ny) == food:
                end = (nx, ny)
                break
            frontier.append(((nx, ny), t + 1))
    if end is None:
        return False
    path = []
    cell = end
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    # replay the path on a copy; the last cell is the food (snake grows)
    occ = env.occ.copy()
    snake = deque(body)
    for i, (px, py) in enumerate(path):
        last = i == len(path) - 1
        if not last:
            tx, ty = snake.pop()
            occ[ty, tx] = 0
        snake.appendleft((px, py))
        occ[py, px] = 1
    ntx, nty = snake[-1]
    hx2, hy2 = snake[0]
    area, tail_r, _ = _flood(occ, hx2, hy2, ntx, nty, food[0], food[1], w, h)
    if tail_r != 1:
        return False
    if len(snake) >= hole and area != w * h - len(snake):
        return False
    return True


def choose_action_goal(env, q, state, hole=10**9, wait_depth=15, budget=3000):
    """Prefer a shield-safe move that has a healthy way to eat the food.

    If no move does, fall back to the plain lookahead ("wait") policy.
    """
    safe = safe_mask_np(state)
    order = [int(a) for a in np.argsort(-q) if safe[a]]
    for a in order:
        if goal_viable(env, a, hole):
            return a
    return choose_action(env, q, state, wait_depth, budget)


def safe_mask_np(s):
    """numpy twin of rl_train.safe_mask for a single state."""
    blocked = s[0:3] > 0.5
    ok = (~blocked) & (s[17:20] > 0.5)
    if not ok.any():
        ok = ~blocked
    if not ok.any():
        ok = np.ones(3, dtype=bool)
    return ok


def choose_action(env, q, state, depth, budget=3000, hole=10**9):
    """Best-Q move among shield-safe moves that also pass the lookahead."""
    safe = safe_mask_np(state)
    order = [a for a in np.argsort(-q) if safe[a]]
    if depth > 0:
        for h in (depth, depth // 2, depth // 4):
            if h < 1:
                break
            for a in order:
                if move_viable(env, int(a), h, budget, hole):
                    return int(a)
    return int(order[0])


def play_game(args):
    """Worker: play one full game; returns (score, steps, cause, seconds)."""
    seed, depth, budget, model_path, timeout, hole, goal = args
    torch.set_num_threads(1)
    ck = torch.load(model_path, map_location="cpu")
    net = QNet(n_in=ck["n_in"])
    net.load_state_dict(ck["model"])
    net.eval()
    env = SnakeEnv(seed=seed)
    if timeout:  # search plays more carefully, so allow longer food-less stretches
        env.timeout_limit = lambda: timeout
    state = env.state()
    t0 = time.time()
    while True:
        with torch.no_grad():
            q = net(torch.as_tensor(state).unsqueeze(0))[0].numpy()
        if goal:
            action = choose_action_goal(env, q, state, hole, depth or 15, budget)
        else:
            action = choose_action(env, q, state, depth, budget, hole)
        _, done = env.step(action)
        if done:
            if env.won:
                cause = "win"
            elif env.since_food > env.timeout_limit():
                cause = "timeout"
            else:
                cause = "dead"
            return env.score, env.steps, cause, time.time() - t0
        state = env.state()


def evaluate_depth(
    depth,
    games,
    budget,
    model_path,
    procs,
    timeout=0,
    hole=10**9,
    goal=False,
    seed0=5000,
):
    """Play `games` games at one search depth in parallel and summarise."""
    jobs = [
        (seed0 + i, depth, budget, str(model_path), timeout, hole, goal)
        for i in range(games)
    ]
    with Pool(procs) as pool:
        res = pool.map(play_game, jobs, chunksize=1)
    sc = np.array([r[0] for r in res])
    causes = [r[2] for r in res]
    return {
        "depth": depth,
        "hole": hole,
        "goal": goal,
        "mean": round(float(sc.mean()), 1),
        "median": int(np.median(sc)),
        "max": int(sc.max()),
        "ge350": int((sc >= 350).sum()),
        "wins": causes.count("win"),
        "timeouts": causes.count("timeout"),
        "sec_per_game": round(float(np.mean([r[3] for r in res])), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(HERE / "rl_snake_best.pth"))
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 15, 40])
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--procs", type=int, default=16)
    parser.add_argument("--goal", action="store_true", help="food-directed search")
    parser.add_argument(
        "--holes",
        type=int,
        nargs="+",
        default=[10**9],
        help="body length from which sealed free cells are forbidden",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="steps without food before cutoff (0 = env default)",
    )
    parser.add_argument("--out", default=str(HERE / "rl_search_results.txt"))
    args = parser.parse_args()
    for depth in args.depths:
        for hole in args.holes:
            r = evaluate_depth(
                depth,
                args.games,
                args.budget,
                args.model,
                args.procs,
                args.timeout,
                hole,
                args.goal,
            )
            line = str(r)
            print(line, flush=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(line + chr(10))
