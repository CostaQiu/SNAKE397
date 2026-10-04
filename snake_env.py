"""Fast headless Snake environment with space-aware features for RL.

The old state only exposed "is there danger one cell ahead/left/right", so a
long snake could not see that it was sealing itself into a dead end. The
features here add flood-fill information (reachable free area, can the tail be
reached, can the food be reached) for each of the three possible moves.

Actions are relative to the current heading: 0 = straight, 1 = turn right,
2 = turn left. Headings are 0=up, 1=right, 2=down, 3=left (clockwise).
"""

import numpy as np
from collections import deque
from numba import njit

DX = np.array([0, 1, 0, -1], dtype=np.int64)
DY = np.array([-1, 0, 1, 0], dtype=np.int64)
ACTION_TURN = (0, 1, 3)  # heading offset for straight / right / left

N_FEATURES = 27


@njit(cache=True)
def _flood(occ, sx, sy, tx, ty, fx, fy, w, h):
    """Flood fill from (sx, sy) over free cells.

    The tail cell is passable (it will have moved by the time we arrive) but is
    not expanded and not counted in the area.

    Returns (free_area, tail_reached, food_reached).
    """
    seen = np.zeros(w * h, dtype=np.uint8)
    stack = np.empty(w * h, dtype=np.int64)
    top = 0
    stack[top] = sy * w + sx
    top += 1
    seen[sy * w + sx] = 1
    area = 0
    tail_reached = 0
    food_reached = 1 if (sx == fx and sy == fy) else 0
    while top > 0:
        top -= 1
        cur = stack[top]
        cx = cur % w
        cy = cur // w
        for k in range(4):
            nx = cx + DX[k]
            ny = cy + DY[k]
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            idx = ny * w + nx
            if seen[idx] == 1:
                continue
            if nx == tx and ny == ty:
                seen[idx] = 1
                tail_reached = 1
                continue
            if occ[ny, nx] == 1:
                continue
            seen[idx] = 1
            area += 1
            if nx == fx and ny == fy:
                food_reached = 1
            stack[top] = idx
            top += 1
    return area, tail_reached, food_reached


@njit(cache=True)
def _is_blocked(occ, nx, ny, tx, ty, eating, w, h):
    """True if moving the head into (nx, ny) kills the snake."""
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        return True
    if occ[ny, nx] == 1:
        # The tail cell is vacated on a normal move, but not when eating.
        if nx == tx and ny == ty and not eating:
            return False
        return True
    return False


@njit(cache=True)
def _ray(occ, hx, hy, d, w, h):
    """Number of free cells in direction d before a wall or body segment."""
    n = 0
    x = hx + DX[d]
    y = hy + DY[d]
    while 0 <= x < w and 0 <= y < h and occ[y, x] == 0:
        n += 1
        x += DX[d]
        y += DY[d]
    return n


@njit(cache=True)
def compute_features(occ, hx, hy, d, tx, ty, fx, fy, length, w, h, out):
    """Fill `out` (length N_FEATURES) with the egocentric feature vector."""
    size = max(w, h)
    free_total = max(1, w * h - length)
    i = 0

    # Per-move block: danger, ray length (relative to heading: S, R, L)
    for a in range(3):
        nd = (d + (0 if a == 0 else (1 if a == 1 else 3))) % 4
        nx = hx + DX[nd]
        ny = hy + DY[nd]
        eating = nx == fx and ny == fy
        out[i + a] = 1.0 if _is_blocked(occ, nx, ny, tx, ty, eating, w, h) else 0.0
        out[3 + a] = _ray(occ, hx, hy, nd, w, h) / size
    i = 6

    # Food in the snake's own frame: forward / right components
    dx = fx - hx
    dy = fy - hy
    rd = (d + 1) % 4
    fwd = dx * DX[d] + dy * DY[d]
    rgt = dx * DX[rd] + dy * DY[rd]
    out[i] = fwd / size
    out[i + 1] = rgt / size
    out[i + 2] = 1.0 if fwd > 0 else 0.0
    out[i + 3] = 1.0 if rgt > 0 else 0.0
    out[i + 4] = 1.0 if rgt < 0 else 0.0
    i = 11

    # Flood-fill block per move: area ratio, fits-body, tail reachable,
    # manhattan distance to food after the move, food reachable
    for a in range(3):
        nd = (d + (0 if a == 0 else (1 if a == 1 else 3))) % 4
        nx = hx + DX[nd]
        ny = hy + DY[nd]
        eating = nx == fx and ny == fy
        blocked = _is_blocked(occ, nx, ny, tx, ty, eating, w, h)
        dist = (abs(fx - nx) + abs(fy - ny)) / (w + h)
        if blocked:
            out[i + a] = 0.0
            out[i + 3 + a] = 0.0
            out[i + 6 + a] = 0.0
            out[i + 9 + a] = dist
            out[i + 12 + a] = 0.0
        else:
            area, tail_r, food_r = _flood(occ, nx, ny, tx, ty, fx, fy, w, h)
            out[i + a] = area / free_total
            out[i + 3 + a] = min(area / max(1, length), 1.0)
            out[i + 6 + a] = tail_r
            out[i + 9 + a] = dist
            out[i + 12 + a] = food_r
    i = 26
    out[i] = length / (w * h)


class SnakeEnv:
    """Single Snake game with relative actions and shaped rewards."""

    FOOD_REWARD = 1.0
    DEATH_REWARD = -1.0
    SHAPE = 0.02  # reward for moving closer to / farther from food

    def __init__(self, w=20, h=20, seed=None):
        self.w = w
        self.h = h
        self.rng = np.random.default_rng(seed)
        self.occ = np.zeros((h, w), dtype=np.uint8)
        self._feat = np.zeros(N_FEATURES, dtype=np.float32)
        self.reset()

    def reset(self):
        """Start a new game: 3-segment snake in the middle heading right."""
        self.occ[:] = 0
        cx, cy = self.w // 2, self.h // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.d = 1
        for x, y in self.snake:
            self.occ[y, x] = 1
        self.score = 0
        self.steps = 0
        self.since_food = 0
        self.done = False
        self.won = False
        self._place_food()
        return self.state()

    def _place_food(self):
        """Put food on a random free cell; returns False if the board is full."""
        free = np.flatnonzero(self.occ.ravel() == 0)
        if len(free) == 0:
            return False
        idx = int(self.rng.choice(free))
        self.food = (idx % self.w, idx // self.w)
        return True

    def timeout_limit(self):
        """Max steps without eating before the game is cut off (anti-looping)."""
        return 150 + 3 * len(self.snake)

    def state(self):
        """Return the egocentric feature vector (a reused float32 array copy)."""
        hx, hy = self.snake[0]
        tx, ty = self.snake[-1]
        compute_features(
            self.occ,
            hx,
            hy,
            self.d,
            tx,
            ty,
            self.food[0],
            self.food[1],
            len(self.snake),
            self.w,
            self.h,
            self._feat,
        )
        return self._feat.copy()

    def step(self, action):
        """Advance one move. Returns (reward, done)."""
        self.d = (self.d + ACTION_TURN[action]) % 4
        hx, hy = self.snake[0]
        tx, ty = self.snake[-1]
        nx, ny = hx + int(DX[self.d]), hy + int(DY[self.d])
        self.steps += 1
        self.since_food += 1

        eating = (nx, ny) == self.food
        if _is_blocked(self.occ, nx, ny, tx, ty, eating, self.w, self.h):
            self.done = True
            return self.DEATH_REWARD, True

        old_dist = abs(self.food[0] - hx) + abs(self.food[1] - hy)
        if not eating:
            self.snake.pop()
            self.occ[ty, tx] = 0
        self.snake.appendleft((nx, ny))
        self.occ[ny, nx] = 1

        if eating:
            self.score += 1
            self.since_food = 0
            if not self._place_food():  # board full: perfect game
                self.done = True
                self.won = True
                return self.FOOD_REWARD, True
            return self.FOOD_REWARD, False

        if self.since_food > self.timeout_limit():
            self.done = True
            return self.DEATH_REWARD, True

        new_dist = abs(self.food[0] - nx) + abs(self.food[1] - ny)
        reward = self.SHAPE if new_dist < old_dist else -self.SHAPE
        return reward, False
