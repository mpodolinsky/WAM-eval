"""Evaluate a LeRobot policy on LIBERO-X with LIBERO-X's own protocol (its `eval_template.py`).

    uv run --no-sync python scripts/libero_x/eval.py --levels LEVEL1 --n-tasks 5 --n-rollouts 5

This does not use `lerobot-eval`: LIBERO-X is a fork of LIBERO with its own `libero` package (new objects, textures
and goal predicates), and its data uses another image convention than LeRobot's LIBERO environment (see
docs/implementation-notes.md). The protocol is the template's: a fresh environment per episode, reset to the stored
init states (the first `--n-rollouts` of the 10 of each task), the policy's action applied one per env step, at most
`--max-steps` (1200) steps, and success is the environment's `done`.

The policy is any LeRobot checkpoint with this input and output layout: two cameras, an 8-D state
`[eef pos, axis-angle, gripper fingers]` and 7-D actions `[dpos, axis-angle, gripper]` (+-1 after post-processing).
Images are fed at the simulator's 256 px, flipped vertically only (LIBERO-X's native convention); the policy resizes.

Writes to `--output-dir`: `results.jsonl` (one line per episode; an interrupted run resumes), `summary.json` and,
with `--save-videos`, one mp4 per episode.
"""

import argparse
import dataclasses
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIBERO_X = REPO / "submodules" / "LIBERO-X"
TASKS = Path(__file__).resolve().parent / "tasks"
DEFAULT_POLICY = "podolinsky/vla-jepa-libero-x-level123-step10000"
LEVELS = ("LEVEL1", "LEVEL2", "LEVEL3")  # the levels with a task list: the 218 tasks the checkpoint was trained on


def configure_libero_x() -> None:
    """Make `import libero` mean LIBERO-X's fork, in this process only.

    `hf-libero`, which LeRobot uses for LIBERO-10, installs a package of the same name. Whichever is imported first
    stays in `sys.modules`, so this must run before anything else can import `libero`."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(LIBERO_X))
    config = REPO / ".libero-x"  # LIBERO reads its paths from a config file and prompts on stdin if it is missing
    config.mkdir(exist_ok=True)
    root = LIBERO_X / "libero" / "libero_x"
    (config / "config.yaml").write_text(
        "\n".join([f"benchmark_root: {root}", f"bddl_files: {root / 'bddl'}", f"init_states: {root / 'init'}",
                   f"datasets: {config / 'datasets'}", f"assets: {root}", ""])
    )  # fmt: skip
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    # LIBERO-X's vector-env module imports the old `gym` only for type annotations; gymnasium is a drop-in for that.
    import gymnasium

    sys.modules.setdefault("gym", gymnasium)
    import libero.libero
    import libero.libero.envs  # noqa: F401

    if not Path(libero.libero.__file__).resolve().is_relative_to(LIBERO_X):
        raise RuntimeError(f"`libero` was imported from {libero.libero.__file__}, not from LIBERO-X")


configure_libero_x()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.utils.parse_bddl import parse_bddl_file  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat2axisangle(quat) -> np.ndarray:
    """robosuite quaternion (x, y, z, w) to axis-angle, as in LIBERO's data and `eval_template.py`."""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else q[:3] * 2.0 * math.acos(q[3]) / den


def upright(image: np.ndarray) -> np.ndarray:
    """The simulator renders upside down; LIBERO-X's data are flipped vertically only (not rotated by 180 degrees)."""
    return np.ascontiguousarray(image[::-1, :])


def raw_batch(obs: dict, prompt: str) -> dict:
    """An observation of the environment as the policy's preprocessor expects it: float images in [0, 1], CHW."""

    def image(key: str) -> torch.Tensor:
        return torch.from_numpy(upright(obs[key])).permute(2, 0, 1).float().div(255.0)[None]

    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    return {
        "observation.images.image": image("agentview_image"),
        "observation.images.image2": image("robot0_eye_in_hand_image"),
        "observation.state": torch.from_numpy(state.astype(np.float32))[None],
        "task": [prompt],
    }


def compatible_checkpoint(path: str) -> str:
    """A local copy of the checkpoint whose config only has fields the pinned LeRobot knows.

    LeRobot's config decoding is strict, and a checkpoint written by a newer LeRobot may carry fields that the pinned
    commit does not have (the LIBERO-X VLA-JEPA fine-tune was trained with 0.7.0 and has `world_model_rope_legacy`).
    Unknown fields are listed and dropped; the weights are symlinked. Check that the dropped fields cannot matter for
    the policy's outputs (here: a flag of the world-model head, which is not run when acting)."""
    from huggingface_hub import snapshot_download

    src = Path(path) if Path(path).is_dir() else Path(snapshot_download(path))
    raw = json.loads((src / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(PreTrainedConfig.get_choice_class(raw["type"]))} | {"type"}  # "type" picks the class
    unknown = sorted(set(raw) - known)
    if not unknown:
        return str(src)
    dst = REPO / ".cache" / "checkpoints" / src.parent.parent.name.replace("models--", "") / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        link = dst / f.name
        if f.name != "config.json" and not link.exists():
            link.symlink_to(f.resolve())
    (dst / "config.json").write_text(json.dumps({k: v for k, v in raw.items() if k in known}, indent=2))
    print(f"config fields unknown to the pinned LeRobot, dropped: { {k: raw[k] for k in unknown} }", flush=True)
    return str(dst)


class PolicyRunner:
    """A LeRobot policy with its processors: `act` returns one 7-D simulator action per call (the policy keeps a queue
    of the chunk it predicted)."""

    def __init__(self, path: str, device: str = "cuda"):
        path = compatible_checkpoint(path)
        config = PreTrainedConfig.from_pretrained(path)
        self.policy = get_policy_class(config.type).from_pretrained(path).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
        )  # fmt: skip

    def reset(self) -> None:
        self.policy.reset(), self.pre.reset(), self.post.reset()

    def act(self, obs: dict, prompt: str) -> list[float]:
        batch = self.pre(raw_batch(obs, prompt))
        with torch.inference_mode():
            action = transition_to_policy_action(policy_output_to_transition(self.policy.select_action(batch)))
        return np.clip(self.post(action)[0].float().cpu().numpy(), -1.0, 1.0).tolist()


def load_task_names(level: str, args: argparse.Namespace) -> list[tuple[int, str]]:
    """The tasks to run as `(position in the level's list, BDDL name)`. The position, not the order of the selection,
    seeds an episode, so a task gets the same result whichever tasks are run with it."""
    path = args.tasks_file or TASKS / f"{level}.txt"
    names = path.read_text().split()
    if args.task_indices:
        return [(i, names[i]) for i in args.task_indices]
    if args.n_tasks:
        return sorted(random.Random(args.seed).sample(list(enumerate(names)), min(args.n_tasks, len(names))))
    return list(enumerate(names))


def run_episode(runner: PolicyRunner, bddl: Path, init_state, ep: int, prompt: str, args, seed: int) -> dict:
    """One episode, as `eval_template.py` runs it in init mode: reset, regenerate the stored state, then step."""
    torch.manual_seed(seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        horizon=args.max_steps + args.num_steps_wait + 1,  # beyond the loop below, so that `done` is only success
    )  # fmt: skip
    frames, done, steps = [], False, 0
    try:
        env.seed(ep)
        env.reset()
        obs = env.regenerate_obs_from_state(init_state.numpy() if hasattr(init_state, "numpy") else init_state)
        runner.reset()
        for steps in range(1, args.max_steps + 1):
            if args.save_videos:
                frames.append(upright(obs["agentview_image"]))
            obs, _, done, _ = env.step(runner.act(obs, prompt))
            if done:
                break
        if done and not env.check_success():
            raise RuntimeError("the environment reports done but not success")
    finally:
        env.close()
    return {"success": bool(done), "steps": steps, "frames": frames}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def summarise(rows: list[dict]) -> dict:
    out = {}
    for level in sorted({r["level"] for r in rows}):
        lr = [r for r in rows if r["level"] == level]
        k = sum(r["success"] for r in lr)
        tasks = {}
        for r in lr:
            t = tasks.setdefault(r["task_file"], [0, 0])
            t[0] += r["success"]
            t[1] += 1
        out[level] = {"episodes": len(lr), "successes": k, "success_rate_pct": round(100 * k / len(lr), 1),
                      "ci95_pct": wilson(k, len(lr)), "tasks": len(tasks),
                      "tasks_with_a_success": sum(t[0] > 0 for t in tasks.values()),
                      "mean_steps_when_successful": round(float(np.mean([r["steps"] for r in lr if r["success"]])), 1) if k else None,
                      "per_task": {name: f"{s}/{n}" for name, (s, n) in sorted(tasks.items())}}  # fmt: skip
    return out


def main(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    done_keys = {(r["level"], r["task_file"], r["episode_index"]) for r in rows}
    print(f"policy {args.policy_path}; {len(rows)} episodes already in {results}", flush=True)
    runner = PolicyRunner(args.policy_path, args.device)

    for level in args.levels:
        base = LIBERO_X / "libero" / "libero_x"
        names = load_task_names(level, args)
        print(f"{level}: {len(names)} tasks x {args.n_rollouts} episodes", flush=True)
        for pos, name in names:
            bddl = base / "bddl" / level / name
            init_states = torch.load(base / "init" / level / f"{bddl.stem}.init", weights_only=False)
            prompt = parse_bddl_file(bddl)["language"]
            for ep in range(min(args.n_rollouts, len(init_states))):
                if (level, name, ep) in done_keys:
                    continue
                t0 = time.time()
                try:
                    r = run_episode(runner, bddl, init_states[ep], ep, prompt, args, seed=args.seed * 10007 + pos * 101 + ep)
                    error = None
                except Exception as e:  # one broken episode must not end a long run; it is recorded as a failure
                    r, error = {"success": False, "steps": 0, "frames": []}, f"{type(e).__name__}: {e}"
                video = ""
                if r["frames"]:
                    import imageio.v2 as imageio

                    (out / "videos" / level).mkdir(parents=True, exist_ok=True)
                    video = str(out / "videos" / level / f"{bddl.stem}_ep{ep}_{'success' if r['success'] else 'failure'}.mp4")
                    imageio.mimwrite(video, r["frames"], fps=20)
                row = {"level": level, "task_file": name, "task_position": pos, "task_desc": prompt, "episode_index": ep,
                       "success": r["success"], "steps": r["steps"], "seconds": round(time.time() - t0, 1),
                       "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
                       "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20), "video": video, "error": error}  # fmt: skip
                rows.append(row)
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{level} task {pos:2d} ep{ep}] {'success' if r['success'] else 'failure'} in {r['steps']:4d} steps, "
                      f"{row['seconds']:5.1f}s, rss {row['peak_rss_mb']} MB, gpu {row['peak_gpu_mb']} MB"
                      + (f"  ERROR {error}" if error else ""), flush=True)  # fmt: skip
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps({"policy": args.policy_path, "max_steps": args.max_steps, **summary}, indent=1))
    for level, s in summary.items():
        print(f"{level}: {s['successes']}/{s['episodes']} = {s['success_rate_pct']}% (95% CI {s['ci95_pct'][0]}-{s['ci95_pct'][1]}), "
              f"{s['tasks_with_a_success']}/{s['tasks']} tasks with a success")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", default=DEFAULT_POLICY, help="LeRobot checkpoint (Hub id or folder)")
    ap.add_argument("--levels", nargs="+", default=["LEVEL1"], choices=LEVELS)
    ap.add_argument("--n-tasks", type=int, default=None, help="sample N tasks of each level (seeded); default all")
    ap.add_argument("--task-indices", type=int, nargs="*", default=None, help="positions in the level's task list")
    ap.add_argument("--tasks-file", type=Path, default=None, help="BDDL names, one per line (default: the level's list)")
    ap.add_argument("--n-rollouts", type=int, default=5, help="episodes per task: init states 0 .. N-1 (at most 10)")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--num-steps-wait", type=int, default=10, help="only used for the environment horizon, as in the template")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "libero_x")
    main(ap.parse_args())
