# Implementation notes

How the environment is put together and why. None of this is needed to run an evaluation (`./setup.sh`, then
`./eval.sh`); read it when something breaks or when you want to change a pin.

## Layout

| | |
|---|---|
| `setup.sh` | creates the environment: submodules, MuJoCo 2.3.7 built from source, `uv sync`, a check of the build |
| `eval.sh` | `lerobot-eval` with the defaults of each policy's model card or guide; extra flags override them |
| `submodules/lerobot` | LeRobot at the commit of the Cosmos3 model card |
| `submodules/LIBERO` | `hf-libero` 0.1.4, as in the Cosmos3 model card |
| `submodules/LIBERO-X` | Meituan's [LIBERO-X](https://github.com/meituan/LIBERO-X) at the commit the earlier feature-collection pipeline used (about 800 MB with its assets) |
| `scripts/cosmos3/` | `predicted_video.py` (compares the video Cosmos3 predicts with the real frames) and `architecture_diagram.py`; its README shows the architecture |
| `scripts/libero_x/` | `eval.py`, the LIBERO-X evaluation, and `tasks/LEVEL{1,2,3}.txt`, the 218 tasks the fine-tune was trained on |
| `scripts/vla_jepa/` | `predictor.py` (runs VLA-JEPA's world-model head from a frame buffer and scores it) and `gifs.py` (the GIFs of the README's VLA-JEPA section, imports `predictor.py`) |

`eval.sh` also writes a repo-local LIBERO config (`.libero/`, ignored by git), because LIBERO prompts on stdin at its
first import otherwise, and sets `MUJOCO_GL=egl` for headless rendering. LIBERO's simulation assets are downloaded
from the Hub on first use.

## Pinned versions

The environment follows the Cosmos3 model card, which gives the most complete list of versions. The other policies
run in the same environment.

| | here | Cosmos3 model card |
|---|---|---|
| LeRobot | commit `d0a20dc` (submodule) | `d0a20dc` |
| hf-libero | 0.1.4, `8561c60` (submodule) | 0.1.4 |
| MuJoCo | 2.3.7, built from source | 2.3.7 |
| robosuite | 1.4.0 | 1.4.0 |
| PyTorch / transformers / NumPy | 2.11.0+cu128 / 5.5.4 / 2.2.6 | 2.11.0+cu128 / 5.5.4 / 2.2.6 |

LeRobot is a submodule instead of the PyPI release because PyPI `lerobot==0.7.0` pins transformers 5.22 and
refactors the LIBERO environment processors, so it is not the code the card measured. `d0a20dc` is a commit of a pull
request branch (huggingface/lerobot#5134, squash-merged later), reachable through GitHub's `refs/pull/5134/head`;
`git submodule update` fetches it by SHA. If GitHub ever stopped serving that ref, the fallback is the merge commit
`0265529` (the policy and environment code differ slightly from `d0a20dc`).

## MuJoCo

The Cosmos3 model card asks for MuJoCo 2.3.7: with MuJoCo 3.x the outcome of contact-rich LIBERO tasks changes, so
the numbers are not comparable. `hf-libero` asks for `mujoco>=3`, which `[tool.uv] override-dependencies` overrides.

2.3.7 has no wheel for Python 3.12, so `setup.sh` builds the source distribution against the official C library,
which it downloads to `.mujoco/`.

**The NumPy 2 problem.** The Python bindings of MuJoCo 2.3.7 are built with pybind11 2.11, which does not work with
NumPy 2 (LeRobot requires it): every `uint8` array of the bindings gets stride 0. robosuite hides the collision
meshes of the scene with `vopt.geomgroup[0] = 0`, and with stride 0 the six entries of the array alias one byte, so
the assignment is lost. The cameras then render the collision geoms in their flat colours (yellow table, green and
blue robot) instead of the textured visuals. The same happens with the official 2.3.7 wheel on Python 3.11 under
NumPy 2, so the build is not at fault. The Cosmos3 policy failed on those images: 0 of 8 completed episodes in a
one-episode-per-task check, against 10 of 10 after the fix.

`setup.sh` therefore builds the bindings with pybind11 2.13 (`MUJOCO_CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_PYBIND11=...`),
which leaves the physics (the C library) untouched. Downgrading NumPy is not an option: LeRobot needs `numpy>=2`, and
SciPy 1.18 and pyarrow need it too.

Two safeguards, because the failure is silent (the simulation still runs and the policy just fails):

- `uv` caches built wheels without regard to build flags, so an earlier build with the default pybind11 may be
  reused. `setup.sh` checks `MjvOption().geomgroup.strides == (1,)` and, if it fails, runs `uv cache clean mujoco`
  and rebuilds.
- `eval.sh` runs the same check and refuses to start on broken bindings. It also uses `uv run --no-sync`, so a stale
  lock never triggers a MuJoCo rebuild without the build variables.

The Cosmos3 model card's own environment (NumPy 2.2.6 with MuJoCo 2.3.7) probably had this bug as well. That is an
inference from reproducing it, not something the card states.

## Policies

`eval.sh` keeps the defaults of each policy in one `case` block. To add a policy, add its extra to the `lerobot[...]`
requirement in `pyproject.toml`, run `./setup.sh`, and add a block with the flags of its model card.

| policy | checkpoint | extra |
|---|---|---|
| `cosmos3` | `lerobot/Cosmos3-Edge-Policy-LIBERO_10` | `cosmos3` |
| `vla-jepa` | `lerobot/VLA-JEPA-LIBERO` | `vla_jepa` (adds `qwen-vl-utils`) |

## LIBERO-X

`./eval.sh vla-jepa-libero-x` runs `scripts/libero_x/eval.py`, not `lerobot-eval`. Three things force that, each
checked against the data rather than assumed.

**A second `libero`.** LIBERO-X is a fork of LIBERO with its own `libero` package (new objects, textures, goal
predicates), and `hf-libero`, which LeRobot uses for LIBERO-10, installs one of the same name. Whichever is imported
first stays in `sys.modules`. `eval.py` therefore puts the submodule first on `sys.path`, imports it before anything
else can, and fails if `libero` did not come from `submodules/LIBERO-X`. Nothing is installed, so LIBERO-10 evaluations
are unaffected. Its `.libero-x/config.yaml` is written at start-up (LIBERO prompts on stdin otherwise). LIBERO-X's
vector-env module imports the old `gym` only in type annotations, so `gymnasium` is aliased as `gym` in that process
instead of pinning a 2022 package.

**Another image convention.** LeRobot's LIBERO environment rotates images by 180 degrees. LIBERO-X's data are flipped
vertically only (its `eval_template.py` has `flip_images=False`). I checked this on a training frame: against renders of
the same task, the best pixel error over its 10 init states was 1355 for a vertical flip, 2993 for a 180 degree
rotation and 11946 for the raw image (agent view; the wrist view does not tell the two apart). Feeding the model
rotated images would have been silent.

**One action per env step.** The dataset is labelled 10 fps and the simulator steps at 20 Hz. Replaying a demo's first
60 actions open loop reproduces the recorded end-effector displacement to 3 mm with one env step per action and 84 mm
with two (and the response to the first 5 actions is identical, -0.0215 m, in the data and in the simulator), so a
dataset frame is an env step and the 10 fps is nominal.

**Protocol.** `eval_template.py`'s: a fresh environment per episode, reset to the stored init states (10 per task; the
first `--n-rollouts` are used) with no dummy steps, the horizon set past the step limit so that `done` can only mean
success (and `check_success()` must agree), at most 1200 steps, the policy's whole 7-step chunk executed, a policy seed
per episode, `env.close()` after each episode. The init states are the tasks' scene layouts, not the demonstrations' own
start states (the demos start from a different robot pose).

**Checking the pipeline before trusting a success rate.** The checkpoint had never run in the simulator, so a low number
could be the model or the pipeline. Before any rollout: all 218 task strings of the lists equal the dataset's task
strings exactly; the environment's 8-D state has the dataset's layout, axis-angle branch and gripper signs; and 24 real
training chunks pushed through the evaluation's own preprocess, act and postprocess path differ from the dataset's
actions by 0.034 on average (the gripper exactly), against 0.264 for actions taken from another moment.

**A newer LeRobot's checkpoint.** The fine-tune was written with LeRobot 0.7.0, this repo pins an older commit (the
Cosmos3 card's), and the checkpoint's `config.json` has one field the pinned commit does not know,
`world_model_rope_legacy` (False, the default). `eval.py` lists and drops unknown config fields into a symlinked copy
under `.cache/`. This is safe only because the two LeRobot versions' VLA-JEPA code differs in nothing that decides the
actions: the Qwen interface, the action head and both processor files are identical, and the difference (a RoPE flag
and 41 lines of `world_model.py`) is in the world-model head, which is not run when acting. Bumping the submodule to
0.7.0 would remove the workaround but changes the environment the Cosmos3 numbers were measured in.

