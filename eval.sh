#!/usr/bin/env bash
# Evaluate a policy on LIBERO-10 with the protocol of its model card or guide.
#   ./eval.sh <policy> [flags...]        policy: cosmos3 | vla-jepa | vla-jepa-libero-x | cosmos3-libero-x
# The first two run lerobot-eval; any of its flags can be appended and overrides the default (the last value wins), e.g.
#   ./eval.sh cosmos3 --eval.n_episodes=2 --env.task_ids='[0,1]' --eval.batch_size=1 --output_dir=outputs/smoke
set -euo pipefail
cd "$(dirname "$0")"

# Headless rendering, and a repo-local LIBERO config so that its first import never prompts.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PWD/.libero}"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    root="$PWD/submodules/LIBERO/libero/libero"
    cat >"$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $root
bddl_files: $root/bddl_files
init_states: $root/init_files
datasets: $PWD/submodules/LIBERO/libero/datasets
assets: $root/assets
EOF
fi

# Fail fast on the broken MuJoCo bindings that render collision meshes instead of the scene (see setup.sh).
uv run --no-sync python -c \
    'import mujoco; assert mujoco.MjvOption().geomgroup.strides == (1,)' 2>/dev/null ||
    { echo "eval.sh: the environment is missing or its MuJoCo bindings are broken, run ./setup.sh" >&2; exit 1; }

policy="${1:-}"
[ "$#" -gt 0 ] && shift
case "$policy" in
    cosmos3)  # lerobot/Cosmos3-Edge-Policy-LIBERO_10, model card: 50 episodes per task
        defaults=(
            --policy.path=lerobot/Cosmos3-Edge-Policy-LIBERO_10 --policy.dtype=bfloat16 --tf32=false --seed=0
            --env.obs_type=pixels --env.observation_height=256 --env.observation_width=256 --env.episode_length=510
            --eval.n_episodes=50 --eval.batch_size=4 --eval.use_async_envs=true
        ) ;;
    vla-jepa)  # lerobot/VLA-JEPA-LIBERO, LeRobot guide: 10 episodes per task
        defaults=(--policy.path=lerobot/VLA-JEPA-LIBERO --eval.n_episodes=10 --eval.batch_size=5) ;;
    vla-jepa-libero-x)  # the LIBERO-X fine-tune of VLA-JEPA, with LIBERO-X's own protocol (not lerobot-eval)
        exec uv run --no-sync python scripts/libero_x/eval.py --output-dir "outputs/eval/$policy" "$@" ;;
    cosmos3-libero-x)  # the LIBERO-X fine-tune of Cosmos3-Edge (EMA weights, bfloat16, TF32 off as in the Cosmos3 card)
        exec uv run --no-sync python scripts/libero_x/eval.py --output-dir "outputs/eval/$policy" \
            --policy-path podolinsky/cosmos3-edge-libero-x-level123-step8750 --policy-subfolder ema --dtype bfloat16 --no-tf32 "$@" ;;
    *)
        echo "usage: ./eval.sh <cosmos3|vla-jepa|vla-jepa-libero-x|cosmos3-libero-x> [flags...]" >&2
        exit 2 ;;
esac

exec uv run --no-sync lerobot-eval --env.type=libero --env.task=libero_10 \
    --output_dir="outputs/eval/$policy" "${defaults[@]}" "$@"
