"""Headless Snake trainer: Double + Dueling DQN, n-step returns, parallel envs.

Differences from the older dqn scripts:
  * space-aware features (see snake_env.py) instead of 3 local danger flags
  * many envs stepped in lock-step, one batched forward pass per iteration
  * Ape-X style per-env epsilons: some envs are almost greedy, so the replay
    buffer always contains long-snake situations
  * a training update every few env steps (not one batch per game)
  * vectorised Double-DQN targets, Huber loss, soft target updates
  * no rendering, so it is limited by compute and not by clock.tick(60)

Usage:
    python rl_train.py --steps 3000000
Outputs (next to this file): rl_snake_best.pth, rl_snake_last.pth,
rl_train_log.csv, rl_train.done.flag
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snake_env import SnakeEnv, N_FEATURES

HERE = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
N_STEP = 3
LR = 5e-4
BATCH = 256
UPDATES_PER_ITER = 2
TAU = 0.005
REPLAY_SIZE = 500_000
WARMUP = 20_000


class QNet(nn.Module):
    """Dueling MLP: shared trunk, separate value and advantage heads."""

    def __init__(self, n_in=N_FEATURES, hidden=256, n_out=3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, n_out)
        )

    def forward(self, x):
        z = self.trunk(x)
        a = self.adv(z)
        return self.value(z) + a - a.mean(dim=1, keepdim=True)


def safe_mask(x):
    """Boolean (B, 3) mask of moves that keep the snake alive and not trapped.

    A move is safe if it is not blocked and the tail stays reachable after it.
    (Adding an 'area >= body' alternative was tested and made scores worse.) If nothing qualifies we
    fall back to any unblocked move, then to all moves.
    """
    blocked = x[:, 0:3] > 0.5
    ok = (~blocked) & (x[:, 17:20] > 0.5)
    ok = torch.where(ok.any(1, keepdim=True), ok, ~blocked)
    return torch.where(ok.any(1, keepdim=True), ok, torch.ones_like(ok))


def masked_q(net, x):
    """Q-values with unsafe moves set to -inf."""
    return net(x).masked_fill(~safe_mask(x), -1e9)


class Replay:
    """Uniform replay buffer stored entirely on the GPU."""

    def __init__(self, capacity, n_in):
        self.cap = capacity
        self.s = torch.zeros((capacity, n_in), device=device)
        self.s2 = torch.zeros((capacity, n_in), device=device)
        self.a = torch.zeros(capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(capacity, device=device)
        self.disc = torch.zeros(capacity, device=device)  # gamma^k, 0 if terminal
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, disc):
        n = len(a)
        idx = (torch.arange(n, device=device) + self.pos) % self.cap
        self.s[idx] = torch.as_tensor(np.asarray(s), device=device)
        self.s2[idx] = torch.as_tensor(np.asarray(s2), device=device)
        self.a[idx] = torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)
        self.r[idx] = torch.as_tensor(np.asarray(r, dtype=np.float32), device=device)
        self.disc[idx] = torch.as_tensor(
            np.asarray(disc, dtype=np.float32), device=device
        )
        self.pos = (self.pos + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch):
        i = torch.randint(0, self.size, (batch,), device=device)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.disc[i]


def learn(net, target, opt, replay):
    """One Double-DQN update on a sampled batch."""
    s, a, r, s2, disc = replay.sample(BATCH)
    with torch.no_grad():
        best = masked_q(net, s2).argmax(dim=1, keepdim=True)
        q_next = target(s2).gather(1, best).squeeze(1)
        y = r + disc * q_next
    q = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    with torch.no_grad():  # soft target update
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.mul_(1 - TAU).add_(TAU * p)
    return loss.item()


@torch.no_grad()
def evaluate(net, n_games=20, seed=12345):
    """Run greedy games in lock-step; returns the list of final scores."""
    envs = [SnakeEnv(seed=seed + i) for i in range(n_games)]
    states = np.stack([e.state() for e in envs])
    active = list(range(n_games))
    while active:
        q = masked_q(net, torch.as_tensor(states[active], device=device))
        acts = q.argmax(dim=1).cpu().numpy()
        still = []
        for k, i in enumerate(active):
            _, done = envs[i].step(int(acts[k]))
            if not done:
                states[i] = envs[i].state()
                still.append(i)
        active = still
    return [e.score for e in envs]


def save(net, path, extra=None):
    torch.save({"model": net.state_dict(), "n_in": N_FEATURES, **(extra or {})}, path)


def train(total_steps, n_envs, eval_every, max_minutes):
    net = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    replay = Replay(REPLAY_SIZE, N_FEATURES)

    envs = [SnakeEnv(seed=i) for i in range(n_envs)]
    states = np.stack([e.state() for e in envs])
    # Ape-X epsilons: env 0 explores most, last env is almost greedy.
    eps = np.array([0.4 ** (1 + 7 * i / max(1, n_envs - 1)) for i in range(n_envs)])
    nbuf = [deque() for _ in range(n_envs)]  # per-env (s, a, r) for n-step returns

    log_path = HERE / "rl_train_log.csv"
    log_path.write_text("steps,minutes,games,train_mean100,eval_mean,eval_max,loss\n")
    flag_path = HERE / "rl_train.done.flag"
    if flag_path.exists():
        flag_path.unlink()

    recent = deque(maxlen=100)
    games = 0
    steps = 0
    next_eval = eval_every
    best_eval = -1.0
    loss = float("nan")
    t0 = time.time()
    status = "ok"

    try:
        while steps < total_steps:
            if (time.time() - t0) / 60 > max_minutes:
                status = "time_limit"
                break

            with torch.no_grad():
                xs = torch.as_tensor(states, device=device)
                greedy = masked_q(net, xs).argmax(dim=1).cpu().numpy()
                # random exploration is also restricted to safe moves
                noise = torch.rand((n_envs, 3), device=device)
                rand = noise.masked_fill(~safe_mask(xs), -1).argmax(dim=1).cpu().numpy()
            acts = np.where(np.random.rand(n_envs) < eps, rand, greedy)

            out_s, out_a, out_r, out_s2, out_d = [], [], [], [], []

            def emit(buf_items, s_next, discount):
                """Push the oldest n-step transition built from buf_items."""
                s0, a0, _ = buf_items[0]
                ret = 0.0
                for j, (_, _, rj) in enumerate(buf_items):
                    ret += (GAMMA**j) * rj
                out_s.append(s0)
                out_a.append(a0)
                out_r.append(ret)
                out_s2.append(s_next)
                out_d.append(discount)

            for i, env in enumerate(envs):
                s = states[i].copy()
                reward, done = env.step(int(acts[i]))
                nbuf[i].append((s, int(acts[i]), reward))
                if done:
                    s_next = s  # unused: discount is 0 for terminal transitions
                    items = list(nbuf[i])
                    for k in range(len(items)):
                        emit(items[k:], s_next, 0.0)
                    nbuf[i].clear()
                    recent.append(env.score)
                    games += 1
                    states[i] = env.reset()
                else:
                    s_next = env.state()
                    if len(nbuf[i]) == N_STEP:
                        emit(list(nbuf[i]), s_next, GAMMA**N_STEP)
                        nbuf[i].popleft()
                    states[i] = s_next
            steps += n_envs

            if out_a:
                replay.add(out_s, out_a, out_r, out_s2, out_d)

            if replay.size >= WARMUP:
                for _ in range(UPDATES_PER_ITER):
                    loss = learn(net, target, opt, replay)

            if steps >= next_eval:
                next_eval += eval_every
                scores = evaluate(net)
                mean_eval, max_eval = float(np.mean(scores)), int(np.max(scores))
                minutes = (time.time() - t0) / 60
                train_mean = float(np.mean(recent)) if recent else 0.0
                print(
                    f"steps {steps:>9,} | {minutes:5.1f} min | games {games:>6} | "
                    f"train100 {train_mean:6.1f} | eval mean {mean_eval:6.1f} "
                    f"max {max_eval:>3} | loss {loss:.4f}",
                    flush=True,
                )
                with log_path.open("a") as f:
                    f.write(
                        f"{steps},{minutes:.2f},{games},{train_mean:.2f},"
                        f"{mean_eval:.2f},{max_eval},{loss:.5f}\n"
                    )
                save(net, HERE / "rl_snake_last.pth")
                if mean_eval > best_eval:
                    best_eval = mean_eval
                    save(net, HERE / "rl_snake_best.pth", {"eval_mean": mean_eval})
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:  # leave a flag so the next session knows it failed
        status = f"fail: {exc!r}"
        raise
    finally:
        save(net, HERE / "rl_snake_last.pth")
        minutes = (time.time() - t0) / 60
        flag_path.write_text(
            f"{status}\nsteps={steps} games={games} "
            f"best_eval_mean={best_eval:.2f} minutes={minutes:.1f}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--max-minutes", type=float, default=240)
    args = parser.parse_args()
    train(args.steps, args.envs, args.eval_every, args.max_minutes)
