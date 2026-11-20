"""The GIFs of the VLA-JEPA section of the README: rollouts, the world-model head at runtime, and the architecture.

    uv run --no-sync python scripts/vla_jepa/gifs.py --tasks 0 3 4 --map-task 0

Writes to `outputs/vla_jepa_gifs/`:

    rollouts.gif       closed-loop episodes of the unchanged policy on the given tasks, both cameras (every third
                       step, in real time)
    world_model.gif    for one task, at every policy call: the real frames that followed, V-JEPA2's embedding of them,
                       the embedding the world-model head predicts from the two latest frames, and the "nothing
                       changes" prediction (the context repeated). Embeddings are drawn as colour maps (first three
                       principal components of each camera's 1024 features, fitted on the real ones)
    architecture.gif   how the policy acts and where the world-model head sits
"""

import argparse
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw, ImageFont

from lerobot.configs.eval import EvalPipelineConfig
from predictor import REPO, RuntimePredictor, launch, run_episode, setup, unit_embeddings

TOKENS = 16  # the 256 tokens of a position form a 16 x 16 grid
VIEW_DIM = 1024  # V-JEPA2 ViT-L features of one camera; the two cameras are concatenated

WHITE, GREY, DIM, DARK = (255, 255, 255), (150, 150, 150), (95, 95, 95), (24, 24, 24)
YELLOW, BLUE, GREEN, RED = (255, 210, 0), (90, 170, 255), (110, 220, 130), (255, 120, 110)


def font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_gif(path: Path, frames: list[np.ndarray], fps: float, diagram: bool = False) -> None:
    """RGB frames [H, W, 3] to an optimised GIF through ffmpeg's palette filters. `diagram` for flat colours: the
    palette of videos is built from frame differences, which loses the few colours of a diagram."""
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    palette = ("split[a][b];[a]palettegen[p];[b][p]paletteuse" if diagram else
               "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-filter_complex", palette, "-loop", "0", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)", flush=True)


def to_image(frame: torch.Tensor, size: int) -> Image.Image:
    """A camera frame [1, 3, H, W] in [0, 1] as a square image."""
    array = (frame[0].float().clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array).resize((size, size), Image.LANCZOS)


# -- rollouts ------------------------------------------------------------------------------------------------------
def rollout_gif(episodes: dict, keys: list[str], out: Path, size: int = 160, cap: int = 260) -> None:
    """Both cameras of every episode, one episode above the other, every third step (6.7 fps is real time at 20 Hz),
    at most `cap` steps."""
    panels = {}
    for task_id, (frames, task, success) in episodes.items():
        panels[task_id] = ([[to_image(f[k], size) for k in keys] for f in frames[:cap][::3]], task, success, len(frames))
    n = max(len(p[0]) for p in panels.values())
    head, small, big = 32, font(11), font(12)
    out_frames = []
    for i in range(n):
        canvas = Image.new("RGB", (size * len(keys), (size + head) * len(panels)), DARK)
        draw = ImageDraw.Draw(canvas)
        for row, (task_id, (imgs, task, success, steps)) in enumerate(panels.items()):
            y = row * (size + head)
            finished = i >= len(imgs) - 1
            for c, im in enumerate(imgs[min(i, len(imgs) - 1)]):
                canvas.paste(im, (c * size, y + head))
            for li, line in enumerate(textwrap.wrap(f"task {task_id}: {task}", 54)[:2]):
                draw.text((4, y + 2 + li * 13), line, font=small, fill=WHITE)
            if not finished:
                status = f"step {3 * i}"
            elif steps > cap:
                status = f"step {cap}, failed ({steps} steps)"
            else:
                status = f"{'success' if success else 'failed'} at step {steps}"
            x = size * len(keys) - 4 - draw.textlength(status, font=big)
            draw.rectangle([x - 4, y + head + 2, size * len(keys), y + head + 22], fill=DARK)  # readable on any frame
            draw.text((x, y + head + 4), status, font=big, fill=(GREEN if success else RED) if finished else WHITE)
        out_frames.append(np.array(canvas))
    write_gif(out, out_frames + out_frames[-1:] * 7, fps=20 / 3)


# -- world model ---------------------------------------------------------------------------------------------------
def colour_maps(real: torch.Tensor, others: dict[str, torch.Tensor]) -> dict[str, list[np.ndarray]]:
    """Colour maps [views][positions, 16, 16, 3] of embeddings [positions, 256, 2048]. One PCA per camera, fitted on
    the real embeddings (`real`) and applied to all of them, so that colours are comparable between rows."""
    maps = {name: [] for name in ("real", *others)}
    for v in range(2):
        view = slice(v * VIEW_DIM, (v + 1) * VIEW_DIM)
        x = real[..., view].reshape(-1, VIEW_DIM).float()
        mean = x.mean(0)
        basis = torch.linalg.svd(x - mean, full_matrices=False)[2][:3].T
        z = (x - mean) @ basis
        lo, hi = torch.quantile(z, 0.01, dim=0), torch.quantile(z, 0.99, dim=0)

        def paint(e: torch.Tensor) -> np.ndarray:
            p = (e[..., view].reshape(-1, VIEW_DIM).float() - mean) @ basis
            rgb = ((p - lo) / (hi - lo)).clamp(0, 1) * 255
            return rgb.byte().reshape(e.shape[0], TOKENS, TOKENS, 3).cpu().numpy()

        maps["real"].append(paint(real))
        for name, e in others.items():
            maps[name].append(paint(e))
    return maps


def world_model_gif(rp: RuntimePredictor, frames: list[dict], task: str, out: Path, call_every: int) -> None:
    s, h, cell = rp.stride, rp.horizon, 76
    cols, label_w, row_h, head = h + 1, 118, cell + 16, 60
    big, small = font(12), font(10)
    keys = rp.keys
    out_frames = []
    for t in range(0, len(frames), call_every):
        u = unit_embeddings(rp, frames, t, task)
        if u is None:
            continue
        real = u["full"][1:]
        context = u["context"].reshape(1, TOKENS * TOKENS, -1)
        copy = context.expand(h, -1, -1)
        # maps["real"] holds position 0 (from the full clip) followed by the h real future positions
        maps = colour_maps(u["full"], {"pred": u["runtime"], "copy": copy, "ctx": context})
        canvas = Image.new("RGB", (label_w + cols * 2 * cell + 8, head + 4 * row_h + 22), DARK)
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"{task}", font=big, fill=WHITE)
        draw.text((6, 22), f"policy call at step {t}: the latest two frames predict the embeddings of the next {2 * h} frames", font=small, fill=GREY)
        rows = ["real frames", "V-JEPA2 of the\nreal frames", "predicted from\nthe 2-frame buffer", "copy of the\ncurrent frames"]
        for r, name in enumerate(rows):
            draw.multiline_text((6, head + r * row_h + 22), name, font=small, fill=(WHITE, WHITE, YELLOW, GREY)[r])
        for c in range(cols):
            x = label_w + c * 2 * cell
            draw.text((x + 4, head - 14), "now" if c == 0 else f"+{2 * s * c} steps", font=small, fill=GREY)
            idx = t if c == 0 else t + 2 * s * c  # the last of the two frames of the position
            for v, key in enumerate(keys):
                canvas.paste(to_image(frames[idx][key], cell), (x + v * cell, head))
            for r, kind in enumerate(("real", "pred", "copy"), start=1):
                if c == 0 and r > 1:
                    continue
                m = maps["ctx"] if (c == 0 and r == 1) else maps[kind]
                for v in range(2):
                    arr = m[v][0] if c == 0 else m[v][c - (0 if kind == "real" else 1)]
                    im = Image.fromarray(arr).resize((cell, cell), Image.NEAREST)
                    canvas.paste(im, (x + v * cell, head + r * row_h))
                if c > 0 and kind != "real":
                    pred = u["runtime"][c - 1] if kind == "pred" else copy[c - 1]
                    cos = F.cosine_similarity(pred.float(), real[c - 1].float(), dim=-1).mean().item()
                    draw.text((x + 4, head + r * row_h + cell + 2), f"cosine to real {cos:.2f}", font=small, fill=YELLOW if kind == "pred" else GREY)
        out_frames.append(np.array(canvas))
    write_gif(out, out_frames + out_frames[-1:] * 3, fps=4)


# -- architecture --------------------------------------------------------------------------------------------------
def dashed_rect(draw: ImageDraw.ImageDraw, xy, color, width=2, dash=6) -> None:
    x0, y0, x1, y1 = xy
    for x in range(x0, x1, 2 * dash):
        draw.line([x, y0, min(x + dash, x1), y0], fill=color, width=width)
        draw.line([x, y1, min(x + dash, x1), y1], fill=color, width=width)
    for y in range(y0, y1, 2 * dash):
        draw.line([x0, y, x0, min(y + dash, y1)], fill=color, width=width)
        draw.line([x1, y, x1, min(y + dash, y1)], fill=color, width=width)


def arrow(draw: ImageDraw.ImageDraw, p0, p1, color, width=2) -> None:
    draw.line([p0, p1], fill=color, width=width)
    (x0, y0), (x1, y1) = p0, p1
    d = np.array([x1 - x0, y1 - y0], dtype=float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    tip = np.array([x1, y1], dtype=float)
    draw.polygon([tuple(tip), tuple(tip - 9 * d + 5 * n), tuple(tip - 9 * d - 5 * n)], fill=color)


def architecture_gif(out: Path) -> None:
    W, H = 760, 440
    big, small = font(13), font(11)
    # (key, x, y, w, h, title, detail lines)
    boxes = {
        "in": (14, 150, 150, 84, "inputs", ["2 cameras (224 px)", "the instruction"]),
        "qwen": (206, 150, 140, 84, "Qwen3-VL-2B", ["reads the images and", "the instruction"]),
        "emb": (402, 38, 150, 56, "32 embodied tokens", ["Qwen hidden states"]),
        "dit": (590, 38, 156, 56, "DiT-B action head", ["4 flow-matching steps", "also reads the state (8)"]),
        "act": (590, 128, 156, 50, "action chunk", ["7 steps x 7 dims -> robot"]),
        "tok": (402, 270, 150, 56, "24 action tokens", ["8 per predicted position"]),
        "enc": (206, 330, 150, 56, "V-JEPA2 encoder", ["frozen, embeds the frames"]),
        "pred": (590, 270, 156, 56, "V-JEPA2 predictor", ["trained, 162M parameters"]),
        "out": (590, 346, 156, 46, "next embeddings", ["256 x 2048 per position"]),
    }
    arrows = {  # key -> (from box, side, to box, side)
        "in-qwen": ("in", "r", "qwen", "l"), "qwen-emb": ("qwen", "t", "emb", "l"), "emb-dit": ("emb", "r", "dit", "l"),
        "dit-act": ("dit", "b", "act", "t"), "qwen-tok": ("qwen", "b", "tok", "l"), "tok-pred": ("tok", "r", "pred", "l"),
        "enc-pred": ("enc", "r", "pred", "bl"), "in-enc": ("in", "b", "enc", "t"), "pred-out": ("pred", "b", "out", "t"),
    }  # fmt: skip

    def anchor(key: str, side: str) -> tuple[int, int]:
        x, y, w, h = boxes[key][:4]
        return {"l": (x, y + h // 2), "r": (x + w, y + h // 2), "t": (x + w // 2, y), "b": (x + w // 2, y + h),
                "bl": (x + 30, y + h)}[side]  # fmt: skip

    # phases: which boxes and arrows are lit, in which colour, with which caption
    acting = (["in", "qwen", "emb", "dit", "act"], ["in-qwen", "qwen-emb", "emb-dit", "dit-act"])
    world = (["tok", "enc", "pred", "out"], ["qwen-tok", "tok-pred", "enc-pred", "pred-out", "in-enc"])
    phases = [
        ([], [], WHITE, "The VLA: Qwen3-VL reads the images and the instruction."),
        (["in", "qwen"], ["in-qwen"], YELLOW, "Qwen3-VL reads the two cameras and the instruction."),
        (*acting, YELLOW, "Acting: 32 embodied tokens condition a flow-matching DiT that outputs the next 7 actions."),
        (acting[0] + ["tok"], acting[1] + ["qwen-tok"], YELLOW, "Qwen also writes 24 'action tokens' (hidden states at placeholder tokens in its prompt)."),
        (world[0], world[1], BLUE, "World model: the frozen V-JEPA2 encoder embeds the frames; the predictor guesses the next ones."),
        (None, None, WHITE, "Training: L1 between predicted and real embeddings (weight 0.1) trains the predictor and Qwen's tokens."),
        (None, None, WHITE, "At inference only the top path runs; the bottom one is run here by scripts/vla_jepa/predictor.py."),
    ]
    frames = []
    for lit_boxes, lit_arrows, colour, caption in phases:
        canvas = Image.new("RGB", (W, H), DARK)
        draw = ImageDraw.Draw(canvas)
        final = lit_boxes is None
        for key, (x, y, w, h, title, lines) in boxes.items():
            on = final or key in lit_boxes
            base = (GREEN if key in acting[0] else BLUE) if final and key not in ("in", "qwen") else (colour if on else DIM)
            if final and key in ("in", "qwen"):
                base = YELLOW
            rect = (x, y, x + w, y + h)
            if key in world[0] and (final or key in lit_boxes):
                dashed_rect(draw, rect, base)  # not run when acting
            elif key in world[0]:
                dashed_rect(draw, rect, DIM)
            else:
                draw.rounded_rectangle(rect, radius=6, outline=base, width=2)
            draw.text((x + 8, y + 6), title, font=big, fill=WHITE if on else GREY)
            for i, line in enumerate(lines):
                draw.text((x + 8, y + 26 + i * 14), line, font=small, fill=GREY if on else DIM)
        for key, (a, sa, b, sb) in arrows.items():
            on = final or key in lit_arrows
            col = (GREEN if key in acting[1] or key == "in-qwen" else BLUE) if final else (colour if on else DIM)
            if key == "in-qwen" and final:
                col = YELLOW
            arrow(draw, anchor(a, sa), anchor(b, sb), col if on else DIM)
        draw.text((14, 6), "VLA-JEPA", font=font(15), fill=WHITE)
        draw.text((14, 414), caption, font=small, fill=WHITE)
        if final:
            draw.text((402, 244), "solid: used when acting      dashed: world-model head", font=small, fill=GREY)
        frames.append(np.array(canvas))
    write_gif(out, frames + frames[-1:], fps=0.45, diagram=True)


def main_run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir or REPO / "outputs" / "vla_jepa_gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    def main(cfg: EvalPipelineConfig):
        architecture_gif(out_dir / "architecture.gif")
        sim = setup(cfg)
        rp = RuntimePredictor(sim.policy, stride=1)
        episodes = {}
        for task_id, env in sorted(sim.envs.items()):
            if task_id not in args.tasks and task_id != args.map_task:
                continue
            frames, task, success = run_episode(sim, env)
            print(f"task {task_id}: {len(frames)} steps, {'success' if success else 'failed'}", flush=True)
            if task_id == args.map_task:
                world_model_gif(rp, frames, task, out_dir / "world_model.gif", sim.policy.config.n_action_steps)
            if task_id in args.tasks:
                episodes[task_id] = (frames, task, success)
            else:
                del frames
        rollout_gif({t: episodes[t] for t in args.tasks}, rp.keys, out_dir / "rollouts.gif")

    launch(out_dir, main)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 3, 4], help="tasks of the rollout GIF")
    ap.add_argument("--map-task", type=int, default=0, help="task of the world-model GIF")
    ap.add_argument("--output-dir", type=Path, default=None)
    main_run(ap.parse_args())
