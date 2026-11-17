# cosmos3-libero

Evaluate [`lerobot/Cosmos3-Edge-Policy-LIBERO_10`](https://huggingface.co/lerobot/Cosmos3-Edge-Policy-LIBERO_10)
(NVIDIA's [Cosmos 3](https://arxiv.org/abs/2606.02800) world action model, fine-tuned on LIBERO-10) in the LIBERO-10
simulator, with the evaluation protocol of its model card.

This is a thin wrapper. [LeRobot](https://github.com/huggingface/lerobot) and
[LIBERO](https://github.com/huggingface/LIBERO) are git submodules pinned to the versions the model card was measured
with, `setup.sh` builds the environment, and `eval.sh` is the card's `lerobot-eval` command.

## Setup

Linux x86_64, one NVIDIA GPU (about 10 GB of memory at batch size 1), and `git`, `uv`, `cmake`, `g++`, `curl`.

```bash
git clone --recurse-submodules https://github.com/mpodolinsky/cosmos3-libero && cd cosmos3-libero
./setup.sh
```

`setup.sh` creates the Python 3.12 environment with `uv`. MuJoCo 2.3.7, the version of the card's results, has no wheel
for Python 3.12, so it is built from source (about a minute); see [MuJoCo](#mujoco). The policy weights (about 9 GB)
are downloaded from the Hub on the first run.

## Run

```bash
./eval.sh                                   # the model card's protocol: 10 tasks x 50 episodes, 4 environments in parallel
./eval.sh --eval.n_episodes=5               # 5 episodes per task
./eval.sh --env.task_ids='[0,1]' --eval.n_episodes=2 --eval.batch_size=1 --eval.use_async_envs=false \
          --output_dir=outputs/smoke        # a quick check: tasks 0 and 1
./eval.sh --policy.save_predicted_video=true   # also decode the future video the policy predicts
```

Any [`lerobot-eval`](https://huggingface.co/docs/lerobot/libero) flag can be appended; it overrides the default
(the last value wins). Results go to `outputs/eval/libero_10/` (`eval_info.json` with the success rate overall and per
task, and one video per episode).

The defaults of `eval.sh` are those of the model card: bfloat16, TF32 off, seed 0, 256x256 cameras, 510 steps per
episode, 30 UniPC sampling steps, the 16 actions of a chunk executed in full, the fixed LIBERO initial states.

## Pinned versions

| | here | model card |
|---|---|---|
| LeRobot | commit `d0a20dc` (submodule) | `d0a20dc` |
| hf-libero | 0.1.4, `8561c60` (submodule) | 0.1.4 |
| MuJoCo | 2.3.7, built from source | 2.3.7 |
| robosuite | 1.4.0 | 1.4.0 |
| PyTorch / transformers / NumPy | 2.11.0+cu128 / 5.5.4 / 2.2.6 | 2.11.0+cu128 / 5.5.4 / 2.2.6 |

## MuJoCo

The model card asks for MuJoCo 2.3.7: with MuJoCo 3.x the outcome of contact-rich LIBERO tasks changes, so the
numbers are not comparable.

The Python bindings of MuJoCo 2.3.7 are built with pybind11 2.11, which does not work with NumPy 2 (LeRobot requires
it): every `uint8` array of the bindings gets stride 0. robosuite then cannot hide the collision geoms with
`vopt.geomgroup`, and the cameras render them in flat colours (yellow table, green robot) instead of the textured
scene. The policy fails on these images. `setup.sh` therefore builds the bindings with pybind11 2.13, which leaves the
physics (the C library) untouched, and `eval.sh` refuses to run if the bindings are broken. If `uv` has a cached
earlier build of MuJoCo, `setup.sh` clears the cached MuJoCo builds and rebuilds.

## Results

The model card reports 95.2% (476/500) on LIBERO-10 (50 episodes per task) on an A800.

The full 500-episode evaluation has not been run here. A check with one episode per task (`--eval.n_episodes=1
--eval.batch_size=1`, RTX 4090) succeeded on 10 of 10 tasks, 19 s per episode on average (a failed episode takes about
43 s). Peak GPU memory was 10.2 GB at batch size 1. Rendering with the broken MuJoCo bindings described above gave 0 of
8 completed episodes on the same check.

## Licence

The code in this repository is Apache-2.0 licensed. The Cosmos3 weights are released under the
[OpenMDW-1.1 licence](https://openmdw.ai/license/1-1/) and are not part of this repository. LeRobot (Apache-2.0) and
LIBERO (MIT) keep their own licences.

## References

- NVIDIA Cosmos Team, *Cosmos 3: Omnimodal World Models for physical AI*, 2026. [arXiv:2606.02800](https://arxiv.org/abs/2606.02800)
- [LeRobot](https://github.com/huggingface/lerobot); [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
