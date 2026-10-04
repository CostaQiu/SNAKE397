"""Check that the recorded replays obey the game rules of snake_env.

snake3d.html re-simulates each replay in JavaScript. If a replay is not a legal
game in the Python environment, the 3D viewer and the AI would disagree about
what happened, so every replay must reproduce its recorded score here.
"""

import json
from pathlib import Path

import pytest

from snake_env import ACTION_TURN, SnakeEnv

HERE = Path(__file__).resolve().parent
REPLAYS_JS = HERE / "replays.js"
WINS_JS = HERE / "replays_wins.js"


def _load_replays(path=REPLAYS_JS):
    """Parse a replays file (`window.NAME = [...]`) into a list."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("=") + 1 :].strip().rstrip(";"))


def _action_for_heading(env, heading):
    """Relative action that turns the snake to an absolute heading."""
    for action, turn in enumerate(ACTION_TURN):
        if (env.d + turn) % 4 == heading:
            return action
    raise AssertionError(f"heading {heading} is a U-turn from {env.d}")


@pytest.mark.parametrize("path", [REPLAYS_JS, WINS_JS], ids=lambda p: p.name)
def test_every_replay_is_a_legal_game_with_the_recorded_score(path):
    if not path.exists():
        pytest.skip(f"run record_games.py first ({path.name} missing)")
    replays = _load_replays(path)
    assert replays, f"{path.name} is empty"
    for rep in replays:
        env = SnakeEnv(seed=0)
        # force the recorded start position and food order
        env.snake.clear()
        env.occ[:] = 0
        for x, y in rep["start"]:
            env.snake.append((x, y))
            env.occ[y, x] = 1
        (hx, hy), (nx, ny) = rep["start"][0], rep["start"][1]
        env.d = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}[(hx - nx, hy - ny)]
        foods = iter(rep["foods"])
        env.food = tuple(next(foods))
        env._place_food = lambda foods=foods, env=env: _next_food(env, foods)
        env.timeout_limit = lambda: 10**9
        env.score = env.steps = env.since_food = 0
        env.done = env.won = False

        done = False
        for heading in map(int, rep["moves"]):
            assert not done, f"seed {rep['seed']}: game ended before the last move"
            _, done = env.step(_action_for_heading(env, heading))
        assert env.score == rep["score"], f"seed {rep['seed']}: score mismatch"
        if path == WINS_JS:  # the screensaver file must hold perfect games only
            assert rep["result"] == "win" and rep["score"] == 397
        if rep["result"] == "win":
            assert env.won and done
        else:
            assert not done, f"seed {rep['seed']}: a non-final replay must not die"


def _next_food(env, foods):
    """Replacement for SnakeEnv._place_food: feed the recorded food positions."""
    nxt = next(foods, None)
    if nxt is None:
        return False
    env.food = tuple(nxt)
    return True
