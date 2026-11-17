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
| `scripts/cosmos3_predicted_video.py` | compares the video Cosmos3 predicts with the real frames (see the README) |

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
