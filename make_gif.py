"""Render a time-lapse GIF of one recorded perfect game, drawn like snake_classic.html (docs/snake397.gif)."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DX, DY = [0, 1, 0, -1], [-1, 0, 1, 0]
W = H = 20
INK_LIGHT = (226, 230, 214)
INK, HEAD, TAIL = (180, 185, 168), (51, 211, 74), (229, 56, 59)
SPS = 16  # steps per second shown by the screensaver


def load_replays(path):
    """Read replays_wins.js (a `window.REPLAYS_WINS = [...]` file) into a list of dicts."""
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text[text.index("[") : text.rindex("]") + 1])


def tri(d, x0, y0, cell, direction, color, g):
    """Draw a triangle in one cell; direction 0=up 1=right 2=down 3=left (same shapes as the web page)."""
    lo, hi, mid = g, cell - g, cell / 2
    shapes = [
        [(mid, lo), (lo, hi), (hi, hi)],
        [(hi, mid), (lo, lo), (lo, hi)],
        [(mid, hi), (lo, lo), (hi, lo)],
        [(lo, mid), (hi, lo), (hi, hi)],
    ]
    d.polygon([(x0 + px, y0 + py) for px, py in shapes[direction]], fill=color)


def render(game, step, score, food, body, size, fonts, blink_on=True):
    """Draw one frame of the screensaver look at the given step."""
    w, h = size
    img = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    cell = int(h * 0.92 / W)
    ox, oy = (w - W * cell) // 2, (h - H * cell) // 2
    pad = max(3, int(cell * 0.35))
    d.rectangle(
        [ox - pad, oy - pad, ox + W * cell + pad, oy + H * cell + pad],
        outline=INK,
        width=2,
    )
    g = max(1, round(cell * 0.07))
    if blink_on:
        for i, (x, y) in enumerate(body[1:-1], start=1):
            shade = INK_LIGHT if (i // 5) % 2 == 0 else INK  # blocks of 5 light / 5 dark counted from the head, as in the page
            d.rectangle(
                [
                    ox + x * cell + g,
                    oy + y * cell + g,
                    ox + (x + 1) * cell - g - 1,
                    oy + (y + 1) * cell - g - 1,
                ],
                fill=shade,
            )
        hd = game["moves"][step - 1] if step > 0 else 1
        tri(d, ox + body[0][0] * cell, oy + body[0][1] * cell, cell, int(hd), HEAD, g)
        t, u = body[-1], body[-2]
        td = next(k for k in range(4) if (DX[k], DY[k]) == (t[0] - u[0], t[1] - u[1]))
        tri(d, ox + t[0] * cell, oy + t[1] * cell, cell, td, TAIL, g)
    if food:
        f = max(g + 1, round(cell * 0.28))
        d.rectangle(
            [
                ox + food[0] * cell + f,
                oy + food[1] * cell + f,
                ox + (food[0] + 1) * cell - f - 1,
                oy + (food[1] + 1) * cell - f - 1,
            ],
            fill=INK,
        )
    free = (w - h * 0.92) / 2
    big, small = fonts
    d.text(
        (free / 2, h / 2 - 12), "SNAKE 397", font=big, fill=(255, 255, 255), anchor="mm"
    )
    d.text((free / 2, h / 2 + 12), "no Hamiltonian", font=small, fill=INK, anchor="mm")
    d.text((free / 2, h / 2 + 30), "cycle used", font=small, fill=INK, anchor="mm")
    secs = int(step / SPS)
    rows = [
        ("SCORE", str(score)),
        ("STEPS", f"{step:,}"),
        ("TIME", f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"),
    ]
    x_l, x_r = w - free * 0.92, w - free * 0.08
    for i, (k, v) in enumerate(rows):
        y = h / 2 - 22 + i * 24
        d.text((x_l, y), k, font=small, fill=INK, anchor="lm")
        d.text((x_r, y), v, font=small, fill=(255, 255, 255), anchor="rm")
    return img


def main():
    """Replay one game and save every N-th step as a GIF frame, then blink the finished snake."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=0)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--out", default=str(HERE / "docs" / "snake397.gif"))
    ap.add_argument("--size", default="720x405")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    game = load_replays(HERE / "replays_wins.js")[a.game]
    fonts = (ImageFont.truetype("courbd.ttf", 22), ImageFont.truetype("cour.ttf", 15))
    moves = [int(m) for m in game["moves"]]
    game["moves"] = moves
    stride = max(1, len(moves) // a.frames)
    body, foods, score = [tuple(c) for c in game["start"]], game["foods"], 0
    fi, food, frames = 0, foods[0], []
    for step in range(len(moves) + 1):
        if step % stride == 0 or step == len(moves):
            frames.append(render(game, step, score, food, body, size, fonts))
        if step == len(moves):
            break
        d = moves[step]
        nh = (body[0][0] + DX[d], body[0][1] + DY[d])
        eat = food is not None and nh == tuple(food)
        body.insert(0, nh)
        if not eat:
            body.pop()
        else:
            score, fi = score + 1, fi + 1
            food = foods[fi] if fi < len(foods) else None
    durations = [70] * len(frames)
    for blink in range(6):  # finished: blink like the screensaver does
        frames.append(
            render(
                game,
                len(moves),
                score,
                food,
                body,
                size,
                fonts,
                blink_on=blink % 2 == 1,
            )
        )
        durations.append(450)
    durations[len(durations) - 7] = 900
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    pal[0].save(
        a.out,
        save_all=True,
        append_images=pal[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        f"{a.out}: {len(frames)} frames, final score {score}, {Path(a.out).stat().st_size / 1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
