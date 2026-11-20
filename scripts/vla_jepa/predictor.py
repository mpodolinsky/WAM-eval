"""Predict VLA-JEPA's V-JEPA2 latents at runtime from a small frame buffer, and score them against the real future.

    uv run --no-sync python scripts/vla_jepa/predictor.py --task-ids 0 1 2 --stride 1

VLA-JEPA trains a world-model head on top of the policy: from the V-JEPA2 embeddings of the first positions of an
8-frame clip (one position = a tubelet of 2 frames) and the "action tokens" Qwen3-VL produces for the clip's first
frame, an action-conditioned predictor guesses the embeddings of the next position. It is never run when the policy
acts. `RuntimePredictor` runs it anyway: it keeps the last `stride + 1` frames, treats them as tubelet 0, and rolls
the predictor forward for up to 3 positions (the prompt holds 24 action tokens, 8 per position).

The policy runs unchanged (same actions as `eval.sh`). After each episode every policy call is replayed through the
predictor and compared with V-JEPA2's embeddings of the real frames that followed. Writes
`outputs/vla_jepa_predictor/stride<S>/metrics.json`.

`--stride` is the number of env steps between two frames of a training clip. The dataset is nominally 10 fps while the
simulator steps at 20 Hz; 1 is the natural reading (a dataset step is an env step), 2 is the alternative.
"""

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / ".libero"))

from lerobot.configs import parser  # noqa: E402  (after the environment variables LIBERO and MuJoCo read at import)
from lerobot.configs.eval import EvalPipelineConfig  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.utils import NEW_ROLLOUT_OPTION  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

TOKENS_PER_POSITION = 256  # 16 x 16 patches of one tubelet of the 256 px V-JEPA2
CONTROL_TASK = "put the black bowl in the bottom drawer of the cabinet and close it"
CONTROL_TASK_ALT = "turn on the stove and put the moka pot on it"  # when the control would be the real task


class RuntimePredictor:
    """Rolls the world-model head of a `VLAJEPAPolicy` forward from a buffer of the last `stride + 1` frames.

    `push` every env step with the batch the policy sees; `predict` when the policy is called. Clips are built with the
    policy's own `_prepare_model_inputs`, so Qwen sees the buffer's oldest frame, as in training (the first frame of
    a clip).
    """

    def __init__(self, policy, stride: int = 1, horizon: int = 3):
        self.policy, self.model, self.stride, self.horizon = policy, policy.model, stride, horizon
        self.keys = list(policy.config.image_features)
        self.buffer = collections.deque(maxlen=stride + 1)
        self.tubelet = self.model.video_encoder.config.tubelet_size
        self.tokens_per_step = policy.config.num_action_tokens_per_timestep
        max_horizon = policy.config.num_video_frames // self.tubelet - 1  # action token groups in the prompt
        if not 1 <= horizon <= max_horizon:
            raise ValueError(f"horizon must be 1..{max_horizon}")

    def push(self, batch: dict) -> None:
        self.buffer.append({k: batch[k].detach() for k in self.keys})

    def ready(self) -> bool:
        return len(self.buffer) == self.stride + 1

    def context(self) -> list[dict]:
        """The two frames of tubelet 0: the oldest and the newest of the buffer, `stride` env steps apart."""
        return [self.buffer[0], self.buffer[-1]]

    # -- building blocks (also used for the reference clip) -------------------------------------------------------
    def clip_inputs(self, frames: list[dict], task: str) -> dict:
        batch = {k: torch.stack([f[k][0] for f in frames], dim=0)[None] for k in self.keys}  # [1, T, C, H, W]
        batch["task"] = [task]
        return self.policy._prepare_model_inputs(batch, training=True)  # the training layout: images, videos

    def embed(self, videos: torch.Tensor) -> torch.Tensor:
        """V-JEPA2 embeddings [1, positions * 256, views * 1024] of a clip [1, views, T, C, H, W]."""
        m = self.model
        b, v, t, c, h, w = videos.shape
        pixels = m.video_processor(
            videos=list(videos.reshape(b * v, t, c, h, w)), return_tensors="pt", device=m.video_encoder.device,
            do_rescale=False,
        )["pixel_values_videos"]  # fmt: skip
        emb = m.video_encoder.get_vision_features(pixel_values_videos=pixels)
        assert emb.shape[1] == (t // self.tubelet) * TOKENS_PER_POSITION, emb.shape
        return m._merge_views(emb, b, v)

    def action_tokens(self, inputs: dict, task: str | None = None) -> torch.Tensor:
        instructions = inputs["instructions"] if task is None else [task]
        return self.model._encode_qwen(inputs["images"], instructions, need_action_tokens=True)[1]

    def rollout(self, context: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Autoregressive predictions [horizon, 256, D] from the embeddings of position 0 and the action tokens."""
        predictor, state, out = self.model.video_predictor, context, []
        for k in range(self.horizon):
            pred = predictor(state.float(), tokens[:, : (k + 1) * self.tokens_per_step].float())
            out.append(pred[:, -TOKENS_PER_POSITION:])
            state = torch.cat([state, out[-1]], dim=1)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def predict(self, task: str, task_for_tokens: str | None = None) -> torch.Tensor | None:
        """Predicted embeddings of the next `horizon` positions [horizon, 256, D], None while the buffer fills."""
        if not self.ready():
            return None
        inputs = self.clip_inputs(self.context(), task)
        tokens = self.action_tokens(inputs, task_for_tokens)
        return self.rollout(self.embed(inputs["videos"]), tokens)


def check_weights(policy, repo_id: str) -> dict:
    """The predictor of the loaded policy must be the checkpoint's: nothing here would notice a random one."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    params = dict(policy.model.video_predictor.named_parameters())
    worst, matched, missing = 0.0, 0, []
    with safe_open(hf_hub_download(repo_id, "model.safetensors"), "pt") as f:
        prefix = "model.video_predictor."
        ckpt_keys = {k[len(prefix):] for k in f.keys() if k.startswith(prefix)}
        for name, p in params.items():
            if name not in ckpt_keys:
                missing.append(name)
                continue
            worst = max(worst, (f.get_tensor(prefix + name).float() - p.detach().float().cpu()).abs().max().item())
            matched += 1
    unused = sorted({k.split(".")[0] for k in ckpt_keys - set(params)})
    out = {"parameters": len(params), "matched": matched, "missing_in_checkpoint": missing,
           "checkpoint_keys_not_in_model": unused, "max_abs_diff": worst}  # fmt: skip
    assert not missing and worst < 1e-6, out
    return out


def score(pred: torch.Tensor, gt: torch.Tensor, ref: torch.Tensor, moving: float = 0.2) -> dict:
    """L1 and cosine of pred [K, N, D] against gt [K, N, D] per position, and on the `moving` fraction of tokens
    that differ most between the context `ref` [K, N, D] (the persistence prediction) and gt."""
    pred, gt, ref = pred.float(), gt.float(), ref.float()  # the encoder runs in bf16
    out = {}
    for k in range(gt.shape[0]):
        l1 = (pred[k] - gt[k]).abs().mean(-1)  # per token
        cos = F.cosine_similarity(pred[k], gt[k], dim=-1)
        change = (ref[k] - gt[k]).abs().mean(-1)
        top = change >= torch.quantile(change, 1 - moving)
        out[k + 1] = {"l1": l1.mean().item(), "cos": cos.mean().item(), "l1_moving": l1[top].mean().item(),
                      "cos_moving": cos[top].mean().item()}  # fmt: skip
    return out


def unit_embeddings(rp: RuntimePredictor, frames: list[dict], t: int, task: str) -> dict | None:
    """The embeddings behind one policy call at step `t`: the runtime prediction from the buffer, and the real ones.

    Returns None when the episode has too few frames around `t` for the real clip of the training layout
    `[t - s, t, t + s, ...]`, embedded in one pass (the ground truth)."""
    s, h = rp.stride, rp.horizon
    n_frames = 2 * (h + 1)  # positions 0..h, 2 frames each
    if t - s < 0 or t + (n_frames - 2) * s >= len(frames):
        return None
    rp.buffer.clear()
    for f in frames[t - s : t + 1]:
        rp.push(f)
    with torch.no_grad():
        inputs = rp.clip_inputs(rp.context(), task)
        tokens = rp.action_tokens(inputs)  # 24 tokens, from the clip's first frame (t - s)
        context = rp.embed(inputs["videos"])  # position 0 from the two buffered frames only
        clip = rp.clip_inputs(frames[t - s : t - s + n_frames * s : s], task)
        full = rp.embed(clip["videos"]).reshape(1, h + 1, TOKENS_PER_POSITION, -1)[0]
        return {"inputs": inputs, "tokens": tokens, "context": context, "runtime": rp.rollout(context, tokens),
                "clip": clip, "full": full}  # fmt: skip


def evaluate_episode(rp: RuntimePredictor, frames: list[dict], task: str, call_every: int) -> list[dict]:
    """Replay the policy calls of an episode through the buffer and score each prediction."""
    h, units = rp.horizon, []
    control = CONTROL_TASK if task != CONTROL_TASK else CONTROL_TASK_ALT
    previous = None  # the action tokens of the previous policy call: right kind of tokens, wrong moment
    for t in range(0, len(frames), call_every):  # t = the step of a policy call
        u = unit_embeddings(rp, frames, t, task)
        if u is None:
            continue
        inputs, tokens, context, runtime, clip, full = (u[k] for k in ("inputs", "tokens", "context", "runtime", "clip", "full"))
        with torch.no_grad():
            wrong = rp.rollout(context, rp.action_tokens(inputs, control))
            zero = rp.rollout(context, torch.zeros_like(tokens))
            other_time = rp.rollout(context, previous) if previous is not None else None
            previous = tokens
            gt = full[1:]
            tf = rp.model.video_predictor(
                full[:h].reshape(1, -1, full.shape[-1]).float(), tokens[:, : h * rp.tokens_per_step].float()
            )
            tf = tf.reshape(h, TOKENS_PER_POSITION, -1)
            reference = rp.model._world_model_loss(clip["videos"], tokens).item() if h == 3 else float("nan")
            persistence = context.reshape(1, TOKENS_PER_POSITION, -1).expand(h, -1, -1)
            persistence_tf = full[:h]
        units.append({
            "t": t, "reference_loss": reference,
            "teacher_forced_l1": F.l1_loss(tf, gt).item(),
            "runtime": score(runtime, gt, persistence), "persistence": score(persistence, gt, persistence),
            "wrong_task_tokens": score(wrong, gt, persistence), "zero_tokens": score(zero, gt, persistence),
            "teacher_forced": score(tf, gt, persistence_tf), "persistence_tf": score(persistence_tf, gt, persistence_tf),
        })  # fmt: skip
        if other_time is not None:
            units[-1]["other_time_tokens"] = score(other_time, gt, persistence)
    return units


def aggregate(units: list[dict], horizon: int) -> dict:
    out = {"units": len(units)}
    names = ("runtime", "persistence", "wrong_task_tokens", "zero_tokens", "other_time_tokens", "teacher_forced",
             "persistence_tf")  # fmt: skip
    for name in names:
        have = [u[name] for u in units if name in u]  # other_time_tokens is missing for the first call
        out[name] = {k: {m: float(np.mean([u[k][m] for u in have])) for m in have[0][1]}
                     for k in range(1, horizon + 1)}  # fmt: skip
    for name, base in (("runtime", "persistence"), ("teacher_forced", "persistence_tf")):
        out[f"{name}_l1_improvement_over_persistence"] = {
            k: 1 - out[name][k]["l1"] / out[base][k]["l1"] for k in range(1, horizon + 1)
        }
    return out


def launch(out_dir: Path, fn) -> None:
    """Call `fn(cfg)` with the LeRobot config of the VLA-JEPA LIBERO-10 evaluation (one episode, batch size 1)."""
    sys.argv = [
        "vla_jepa", "--policy.path=lerobot/VLA-JEPA-LIBERO", "--seed=0", "--env.type=libero", "--env.task=libero_10",
        "--eval.n_episodes=1", "--eval.batch_size=1", "--eval.use_async_envs=false", f"--output_dir={out_dir / '_cfg'}",
    ]  # fmt: skip
    parser.wrap()(fn)()


def setup(cfg: EvalPipelineConfig) -> SimpleNamespace:
    """The environments, policy and processors of the evaluation, built as `lerobot-eval` builds them."""
    set_seed(cfg.seed)
    envs = next(iter(make_env(cfg.env, n_envs=1, use_async_envs=False).values()))
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    weights = check_weights(policy, "lerobot/VLA-JEPA-LIBERO")
    print("predictor weights:", weights, flush=True)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )  # fmt: skip
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    return SimpleNamespace(cfg=cfg, envs=envs, policy=policy, pre=pre, post=post, env_pre=env_pre, env_post=env_post,
                           weights=weights)  # fmt: skip


def run_episode(sim: SimpleNamespace, env) -> tuple[list[dict], str, bool]:
    """One closed-loop episode with the unchanged policy: the camera frames the policy saw at every step, the task
    and whether it succeeded."""
    keys = list(sim.policy.config.image_features)
    sim.policy.reset(), sim.pre.reset(), sim.post.reset()
    obs, _ = env.reset(seed=[sim.cfg.seed], options={NEW_ROLLOUT_OPTION: True})
    task = list(env.call("task_description"))
    frames, done, success = [], False, False
    while not done and len(frames) < env.call("_max_episode_steps")[0]:
        o = preprocess_observation(obs)
        o["task"] = task
        o = sim.pre(sim.env_pre(o))
        frames.append({k: o[k].detach().half().cpu() for k in keys})
        with torch.inference_mode():
            action = transition_to_policy_action(policy_output_to_transition(sim.policy.select_action(o)))
        action = sim.env_post({"action": sim.post(action)})["action"]
        obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
        done, success = bool(np.all(terminated | truncated)), bool(np.any(terminated))
    return [{k: v.float().cuda() for k, v in f.items()} for f in frames], task[0], success


def run(args: argparse.Namespace) -> None:
    out_dir = args.output_dir or REPO / "outputs" / "vla_jepa_predictor" / f"stride{args.stride}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def main(cfg: EvalPipelineConfig):
        sim = setup(cfg)
        rp = RuntimePredictor(sim.policy, stride=args.stride, horizon=args.horizon)
        all_units, per_task = [], {}
        for task_id, env in sorted(sim.envs.items()):
            if args.task_ids and task_id not in args.task_ids:
                continue
            frames, task, success = run_episode(sim, env)
            t0 = time.time()
            units = evaluate_episode(rp, frames, task, call_every=sim.policy.config.n_action_steps)
            for u in units:
                u["task_id"] = task_id
            all_units += units
            per_task[task_id] = {"task": task, "steps": len(frames), "success": success, "units": len(units)}
            print(f"task {task_id}: {len(frames)} steps, {len(units)} predictions scored in {time.time() - t0:.0f}s", flush=True)
            del frames
            torch.cuda.empty_cache()
        result = {"stride": args.stride, "horizon": args.horizon, "weights": sim.weights, "tasks": per_task,
                  **aggregate(all_units, args.horizon)}  # fmt: skip
        reference_gap = float(np.nanmax([abs(u["reference_loss"] - u["teacher_forced_l1"]) for u in all_units]))
        result["max_gap_to_library_world_model_loss"] = reference_gap
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("units", "runtime", "persistence", "teacher_forced",
                                                  "max_gap_to_library_world_model_loss")}, indent=1))  # fmt: skip

    launch(out_dir, main)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-ids", type=int, nargs="*", default=None, help="LIBERO-10 tasks (default: all 10)")
    ap.add_argument("--stride", type=int, default=1, help="env steps between the frames of a clip")
    ap.add_argument("--horizon", type=int, default=3, help="positions to predict (2 frames each), at most 3")
    ap.add_argument("--output-dir", type=Path, default=None)
    run(ap.parse_args())
