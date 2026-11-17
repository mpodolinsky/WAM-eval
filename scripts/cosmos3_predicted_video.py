"""Compare the video Cosmos3 predicts with what the simulator then does, in a closed-loop LIBERO-10 episode.

    uv run --no-sync python scripts/cosmos3_predicted_video.py --task-id 0 --chunks 12

The policy runs as in `eval.sh` (one episode, batch size 1). Every 16 steps it predicts a chunk of 16 actions together
with the next 16 frames of the two cameras; the prediction is decoded and paired with the real frames that follow.
Writes to `outputs/cosmos3_predicted_video/task<ID>/`:

    episode.gif            real (top) and predicted (bottom) frames of every chunk, one chunk after the other
                           (every second frame, at 10 fps)
    metrics.json           PSNR of the predicted frame k steps ahead against the real one, and of the copy baseline
    latents.gif            the predicted latents of the first detail chunk, and the pixel frames decoded from them
    chunk<C>_denoise.png   the sampler's estimate of the clean video at several denoising steps (--detail-chunks)
    chunk<C>_samples.png   real frames, the prediction, and predictions from other noise seeds (--detail-chunks)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero"))

from lerobot.configs import parser  # noqa: E402  (after the environment variables LIBERO and MuJoCo read at import)
from lerobot.configs.eval import EvalPipelineConfig  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.cosmos3.modeling_cosmos3 import FlowState  # noqa: E402
from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

CHUNK = 16  # actions per chunk = frames predicted
DENOISE_STEPS = [0, 2, 5, 10, 20, 29]  # sampler steps (of 30) whose clean-video estimate is decoded
FRAME_COLS = [0, 4, 8, 12, 16]  # frames shown in the figures
PSNR_STEPS = [1, 4, 8, 12, 16]


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """[1, 3, T, H, W] in [-1, 1] -> [T, 3, H, W] uint8."""
    return ((video[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 0, 2, 3).cpu().numpy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return 10 * np.log10(255.0**2 / max(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2), 1e-10))


def tag(frame: np.ndarray, text: str) -> Image.Image:
    """A [3, H, W] frame as an image with a caption."""
    image = Image.fromarray(frame.transpose(1, 2, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 7 * len(text) + 6, 13], fill=(0, 0, 0))
    draw.text((3, 1), text, fill=(255, 255, 255))
    return image


def sheet(rows: list[list[np.ndarray]], captions: list[list[str]], scale: float = 0.7) -> Image.Image:
    h, w = rows[0][0].shape[1:]
    out = Image.new("RGB", (w * len(rows[0]), h * len(rows)))
    for r, (row, caps) in enumerate(zip(rows, captions)):
        for c, (frame, cap) in enumerate(zip(row, caps)):
            out.paste(tag(frame, cap), (c * w, r * h))
    return out.resize((int(out.width * scale), int(out.height * scale)), Image.LANCZOS)


def write_gif(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """RGB frames [H, W, 3] to an optimised GIF through ffmpeg's palette filters."""
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-filter_complex", "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
        "-loop", "0", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)


def font(size: int):
    """A TrueType font when the system has one (the default bitmap font squeezes spaces), else PIL's default."""
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, color=(200, 200, 200)) -> None:
    draw.line([x0, y, x1 - 5, y], fill=color, width=2)
    draw.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill=color)


def latent_gif(latents: np.ndarray, pred: np.ndarray, path: Path) -> None:
    """The predicted latents [C, T, h, w] as colour maps (first 3 principal components of the C channels), a flowchart
    of the VAE decoder, and the pixel frames [17, 3, H, W] it decodes: latent frame j covers pixel frames 4j-3 .. 4j."""
    c, t, gh, gw = latents.shape
    flat = latents.reshape(c, -1).T
    flat = flat - flat.mean(0)
    rgb = flat @ np.linalg.svd(flat, full_matrices=False)[2][:3].T
    lo, hi = np.percentile(rgb, [1, 99], axis=0)
    maps = (((rgb - lo) / (hi - lo)).clip(0, 1).reshape(t, gh, gw, 3) * 255).astype(np.uint8)
    cell, pad = 6, 8
    maps = [Image.fromarray(m).resize((gw * cell, gh * cell), Image.NEAREST) for m in maps]
    width = t * gw * cell + (t + 1) * pad
    pix_w = width - 2 * pad
    pix_h = round(pred.shape[2] * pix_w / pred.shape[3])
    maps_y, flow_y, flow_h = 44, 44 + gh * cell + 26, 56
    top = flow_y + flow_h + 12
    big, small = font(13), font(11)
    white, grey, yellow = (255, 255, 255), (150, 150, 150), (255, 210, 0)
    height, wd = pred.shape[2], pred.shape[3]
    frames = []
    for j in range(t):
        for k in [0] if j == 0 else range(4 * j - 3, 4 * j + 1):
            canvas = Image.new("RGB", (width, top + pix_h + 26), (24, 24, 24))
            draw = ImageDraw.Draw(canvas)
            draw.text((pad, 4), f"predicted latents: {c} channels x {t} frames x {gh} x {gw} = {t * gh * gw} vectors of {c} numbers", font=big, fill=white)
            draw.text((pad, 23), "each cell stands for a 16x16 pixel patch, each frame for 4 video frames (the first is the given frame)", font=small, fill=grey)
            for i, m in enumerate(maps):
                x = pad + i * (gw * cell + pad)
                canvas.paste(m, (x, maps_y))
                draw.text((x, maps_y + gh * cell + 4), f"latent frame {i}", font=small, fill=white if i == j else (110, 110, 110))
                if i == j:
                    draw.rectangle([x - 2, maps_y - 2, x + gw * cell + 1, maps_y + gh * cell + 1], outline=yellow, width=2)
            # flowchart: selected latent frame -> VAE decoder -> pixel frames
            frames_text = "pixel frame 0" if j == 0 else f"pixel frames {4 * j - 3}-{4 * j}"
            boxes = [
                (pad, 150, [f"latent frame {j}", f"{c} x {gh} x {gw}"], yellow),
                (pad + 150 + 36, 262, ["Wan2.2 VAE decoder", "undo the standardisation", "16x space, 4x time (causal 3D conv)"], white),
                (width - pad - 150, 150, [frames_text, f"3 x {height} x {wd}"], white),
            ]
            for n, (x, w, lines, color) in enumerate(boxes):
                draw.rounded_rectangle([x, flow_y, x + w, flow_y + flow_h], radius=6, outline=color, width=2)
                for li, text in enumerate(lines):
                    draw.text((x + 8, flow_y + 7 + li * 16), text, font=big if li == 0 else small, fill=white if li == 0 else grey)
                if n < 2:
                    nx = boxes[n + 1][0]
                    arrow(draw, x + w + 4, nx - 4, flow_y + flow_h // 2)
            frame = Image.fromarray(pred[k].transpose(1, 2, 0)).resize((pix_w, pix_h), Image.LANCZOS)
            canvas.paste(frame, (pad, top))
            draw.text((pad, top + pix_h + 6), f"decoded pixel frame {k} ({k / 20:.2f} s after the current frame)", font=big, fill=white)
            frames.append(np.array(canvas))
    write_gif(path, frames + frames[-1:] * 3, fps=4)


def run(args: argparse.Namespace) -> None:
    detail = {int(c) for c in args.detail_chunks.split(",") if c}
    out_dir = args.output_dir or REPO / "outputs" / "cosmos3_predicted_video" / f"task{args.task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    @parser.wrap()
    def main(cfg: EvalPipelineConfig):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        set_seed(cfg.seed)
        device = torch.device("cuda")

        env = next(iter(next(iter(make_env(cfg.env, n_envs=1, use_async_envs=False).values())).values()))
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
        model, network = policy.model, policy.model.network
        plain_sample = type(model).sample.__get__(model)

        def decode(latents: torch.Tensor) -> np.ndarray:
            with torch.no_grad():
                return to_uint8(network.decode_video(latents.to(device)))

        trajectory: list = []

        def recording_sample(velocity_fn, x_t):
            """The sampler of the model, keeping its estimate of the clean video x0 = x_t - sigma * v at each step."""
            video, actions = model.make_scheduler(device), model.make_scheduler(device)
            trajectory.clear()
            for i, timestep in enumerate(video.timesteps):
                v = velocity_fn(x_t, timestep)
                trajectory.append(x_t.video_latents - float(video.sigmas[i]) * v.video_latents)
                x_t = FlowState(
                    video.step(v.video_latents, timestep, x_t.video_latents).prev_sample,
                    actions.step(v.actions, timestep, x_t.actions).prev_sample,
                )
            return x_t

        chunks: list[dict] = []
        t_now = {"t": 0}
        predict = policy.predict_action_chunk

        def predict_and_record(batch, **kwargs):
            index = len(chunks)
            keep = index in detail
            model.sample = recording_sample if keep else plain_sample
            actions = predict(batch, **kwargs)
            latents = policy.last_predicted_latents
            rec = {"t0": t_now["t"], "pred": decode(latents)}
            if keep:
                rec["latents"] = latents[0].float().cpu().numpy()  # [48, 5, 10, 20]
                rec["x0"] = np.stack([decode(trajectory[s]) for s in DENOISE_STEPS])
                model.sample = plain_sample
                extra = []
                for seed in range(args.samples):
                    torch.manual_seed(1000 + seed)
                    predict(batch, **kwargs)
                    extra.append(decode(policy.last_predicted_latents))
                rec["extra"] = np.stack(extra)
            chunks.append(rec)
            policy.last_predicted_latents = latents
            return actions

        policy.predict_action_chunk = predict_and_record

        policy.reset(), pre.reset(), post.reset()
        obs, _ = env.reset(seed=[cfg.seed], options={NEW_ROLLOUT_OPTION: True})
        task = list(env.call("task_description"))
        print("task:", task[0], flush=True)
        real, done = [], False
        for step in range(args.chunks * CHUNK + 1):
            o = preprocess_observation(obs)
            o["task"] = task
            o = pre(env_pre(o))
            canvas = o["pixel_values_videos_gen"][0, :, 0]  # the model's view of the current frame, [3, H, W]
            real.append(((canvas.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy())
            if step == args.chunks * CHUNK or done:
                break
            t_now["t"] = step - step % CHUNK
            with torch.inference_mode():
                action = transition_to_policy_action(policy_output_to_transition(policy.select_action(o)))
            action = env_post({"action": post(action)})["action"]
            obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = bool(np.all(terminated | truncated))
            if done:
                print(f"episode ended at step {step + 1}", flush=True)
        report(real, chunks, task[0], out_dir, args)

    sys.argv = [
        "predicted_video", "--policy.path=lerobot/Cosmos3-Edge-Policy-LIBERO_10", "--policy.dtype=bfloat16",
        "--policy.save_predicted_video=true", "--tf32=false", f"--seed={args.seed}", "--env.type=libero",
        "--env.task=libero_10", f"--env.task_ids=[{args.task_id}]", "--env.obs_type=pixels",
        "--env.observation_height=256", "--env.observation_width=256", "--env.episode_length=510",
        "--eval.n_episodes=1", "--eval.batch_size=1", "--eval.use_async_envs=false",
        f"--output_dir={out_dir / '_cfg'}",
    ]  # fmt: skip
    main()


def report(real: list[np.ndarray], chunks: list[dict], task: str, out_dir: Path, args: argparse.Namespace) -> None:
    """GIF, metrics and figures. The canvas has reflection padding below the content, which is cropped."""
    height = chunks[0]["pred"].shape[2]
    real = np.stack(real)[:, :, :height]
    chunks = [c for c in chunks if c["t0"] + CHUNK < len(real)]  # the last chunk needs its 16 real frames

    gif = []  # every second frame (the GIFs are large), played at half the frame rate: real time
    for i, c in enumerate(chunks):
        for k in range(2, CHUNK + 1, 2):
            top, bottom = tag(real[c["t0"] + k], f"real  t={c['t0'] + k}"), tag(c["pred"][k], f"predicted (chunk {i}, k={k})")
            frame = Image.new("RGB", (top.width, 2 * top.height))
            frame.paste(top, (0, 0)), frame.paste(bottom, (0, top.height))
            gif.append(np.array(frame))
    write_gif(out_dir / "episode.gif", gif, fps=10)

    metrics = {"task": task, "chunks": len(chunks), "psnr_db": {}}
    for k in PSNR_STEPS:
        pred = [psnr(c["pred"][k], real[c["t0"] + k]) for c in chunks]
        copy = [psnr(real[c["t0"]], real[c["t0"] + k]) for c in chunks]
        metrics["psnr_db"][k] = {"predicted": float(np.mean(pred)), "copy_first_frame": float(np.mean(copy))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["psnr_db"], indent=1))

    first = next((c for c in chunks if "latents" in c), None)
    if first is not None:
        latent_gif(first["latents"], first["pred"], out_dir / "latents.gif")

    for i, c in enumerate(chunks):
        if "x0" not in c:
            continue
        t0, cols = c["t0"], FRAME_COLS
        rows = [[c["x0"][s][k] for k in cols] for s in range(len(DENOISE_STEPS))] + [[c["pred"][k] for k in cols]]
        caps = [[f"x0 step {s} k={k}" for k in cols] for s in DENOISE_STEPS] + [[f"final k={k}" for k in cols]]
        sheet(rows, caps).save(out_dir / f"chunk{i}_denoise.png")
        rows = [[real[t0 + k] for k in cols], [c["pred"][k] for k in cols]] + [[e[k] for k in cols] for e in c["extra"]]
        caps = [[f"real k={k}" for k in cols], [f"predicted k={k}" for k in cols]] + [
            [f"seed {s + 1} k={k}" for k in cols] for s in range(len(c["extra"]))
        ]
        sheet(rows, caps).save(out_dir / f"chunk{i}_samples.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-id", type=int, default=0, help="LIBERO-10 task (0-9)")
    ap.add_argument("--chunks", type=int, default=12, help="chunks (of 16 steps) to run")
    ap.add_argument("--detail-chunks", default="0", help="chunks that also get the denoising and noise-seed figures")
    ap.add_argument("--samples", type=int, default=3, help="extra noise seeds for the detail chunks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None)
    run(ap.parse_args())
