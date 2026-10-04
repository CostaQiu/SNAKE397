"""Unit tests for snake_env."""

import numpy as np
from snake_env import SnakeEnv, N_FEATURES, compute_features


def _set_snake(env, cells, d, food):
    """Force the env into a given configuration (head first)."""
    from collections import deque

    env.occ[:] = 0
    env.snake = deque(cells)
    for x, y in cells:
        env.occ[y, x] = 1
    env.d = d
    env.food = food
    env.done = False
    env.since_food = 0


def test_feature_shape_and_range():
    env = SnakeEnv(seed=0)
    s = env.state()
    assert s.shape == (N_FEATURES,)
    assert np.all(np.isfinite(s))
    assert s.min() >= -1.0 and s.max() <= 1.0


def test_wall_death():
    env = SnakeEnv(w=5, h=5, seed=0)
    _set_snake(env, [(4, 2), (3, 2), (2, 2)], d=1, food=(0, 0))
    reward, done = env.step(0)  # straight into the right wall
    assert done and reward == env.DEATH_REWARD


def test_moving_into_vacating_tail_is_legal():
    env = SnakeEnv(w=5, h=5, seed=0)
    # 2x2 loop: head (1,1) heading up->left would hit tail (1,2)? build a ring
    _set_snake(env, [(2, 2), (2, 3), (1, 3), (1, 2)], d=3, food=(4, 4))
    # head heading left goes to (1,2) which is the tail cell: legal
    reward, done = env.step(0)
    assert not done


def test_eating_grows_and_scores():
    env = SnakeEnv(w=6, h=6, seed=0)
    _set_snake(env, [(2, 2), (1, 2), (0, 2)], d=1, food=(3, 2))
    reward, done = env.step(0)
    assert reward == env.FOOD_REWARD and not done
    assert len(env.snake) == 4 and env.score == 1


def test_pocket_area_is_exact():
    """Head above a full-width body row sees only the 5 free cells of row 0."""
    env = SnakeEnv(w=6, h=6, seed=0)
    _set_snake(env, [(5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1)], d=0, food=(3, 4))
    out = np.zeros(N_FEATURES, dtype=np.float32)
    compute_features(env.occ, 5, 1, 0, 0, 1, 3, 4, 6, 6, 6, out)
    assert out[1] == 1.0 and out[2] == 1.0          # right wall, left body: blocked
    assert out[0] == 0.0                            # straight (5,0) is free
    assert abs(out[11] - 5 / 30) < 1e-6             # area / free cells
    assert abs(out[14] - 5 / 6) < 1e-6              # fits-body ratio
    assert out[17] == 1.0                           # tail reachable
    assert out[23] == 0.0                           # food sealed off


def test_random_play_terminates():
    env = SnakeEnv(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(5000):
        _, done = env.step(int(rng.integers(3)))
        if done:
            break
    assert env.done

