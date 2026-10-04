"""Watch the trained RL snake play (greedy policy, pygame window).

Usage:
    python rl_demo.py [--model rl_snake_best.pth] [--fps 30] [--search-depth 15]
Press Q or close the window to quit.
"""

import argparse
from pathlib import Path

import numpy as np
import pygame
import torch

from rl_search import choose_action_goal, safe_mask_np
from rl_train import QNet, device
from snake_env import SnakeEnv

HERE = Path(__file__).resolve().parent
CELL = 30
GRID = 20


def draw(screen, env, font, record, games):
    """Render the board, snake, food and a status line."""
    screen.fill((0, 0, 0))
    for x, y in list(env.snake)[1:]:
        pygame.draw.rect(
            screen, (0, 100, 255), (x * CELL, y * CELL, CELL - 1, CELL - 1)
        )
    hx, hy = env.snake[0]
    pygame.draw.rect(screen, (0, 255, 0), (hx * CELL, hy * CELL, CELL - 1, CELL - 1))
    fx, fy = env.food
    pygame.draw.rect(screen, (200, 0, 0), (fx * CELL, fy * CELL, CELL - 1, CELL - 1))
    text = f"Score {env.score}  Record {record}  Game {games}"
    screen.blit(font.render(text, True, (255, 255, 255)), (8, 6))


def demonstrate(model_path, fps, search_depth):
    """Load a checkpoint and play games forever."""
    path = Path(model_path)
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    ckpt = torch.load(path, map_location=device)
    net = QNet(n_in=ckpt["n_in"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    pygame.init()
    screen = pygame.display.set_mode((GRID * CELL, GRID * CELL))
    pygame.display.set_caption("RL Snake")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    env = SnakeEnv(w=GRID, h=GRID)
    if search_depth > 0:  # a careful snake needs longer food-less stretches
        env.timeout_limit = lambda: 3000
    state = env.state()
    record, games = 0, 1
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                running = False
        with torch.no_grad():
            q = net(torch.as_tensor(state, device=device).unsqueeze(0))[0].cpu().numpy()
        # same safety rule as training, plus optional lookahead search
        if search_depth > 0:
            action = choose_action_goal(env, q, state, wait_depth=search_depth)
        else:  # network + safety rule only
            action = int(np.argmax(np.where(safe_mask_np(state), q, -1e9)))
        _, done = env.step(action)
        record = max(record, env.score)
        draw(screen, env, font, record, games)
        pygame.display.flip()
        clock.tick(fps)
        if done:
            pygame.time.wait(800)
            state = env.reset()
            games += 1
        else:
            state = env.state()
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(HERE / "rl_snake_best.pth"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--search-depth", type=int, default=15, help="food-directed search on top of the network (0 = network + safety rule only)")
    args = parser.parse_args()
    demonstrate(args.model, args.fps, args.search_depth)
