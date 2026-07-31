#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/ws-rtx/Documents/Projects/lidar-centerpoint"
WAVE="${1:-}"

cd "$ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: Activate lidar_centerpoint_g first."
    exit 1
fi

PYTHON="$CONDA_PREFIX/bin/python"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/research/src${PYTHONPATH:+:$PYTHONPATH}"

# Prevent each training process from creating excessive CPU threads.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

case "$WAVE" in
    1)
        JOBS=(
            "0|pillar02_dcn|pillar02_dcn_screen10"
            "1|voxel01|voxel01_screen10"
            "2|voxel01_dcn|voxel01_dcn_screen10"
        )
        ;;

    2)
        JOBS=(
            "0|voxel0075|voxel0075_screen10"
            "1|voxel0075_dcn|voxel0075_dcn_screen10"
        )
        ;;

    *)
        echo "Usage: $0 1|2"
        exit 1
        ;;
esac

pids=()
names=()

for job in "${JOBS[@]}"; do
    IFS="|" read -r gpu model experiment <<< "$job"

    config="research/configs/centerpoint/${model}.py"
    work_dir="research/experiments/${experiment}"
    console_log="research/experiments/${experiment}.console.log"

    if [[ ! -f "$config" ]]; then
        echo "ERROR: Missing config: $config"
        exit 1
    fi

    if find "$work_dir" -maxdepth 1 -name '*.pth' -print -quit \
        2>/dev/null | grep -q .
    then
        echo "ERROR: Existing checkpoints found in $work_dir"
        echo "Remove the directory or resume it explicitly."
        exit 1
    fi

    mkdir -p "$work_dir"

    echo "Starting $model on physical GPU $gpu"

    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" research/tools/train.py \
        "$config" \
        --work-dir "$work_dir" \
        --max-epochs 10 \
        > "$console_log" 2>&1 &

    pid=$!

    echo "$pid" > "research/experiments/${experiment}.pid"

    pids+=("$pid")
    names+=("$experiment")

    echo "  PID: $pid"
    echo "  log: $console_log"
done

echo
echo "Waiting for wave $WAVE..."

status=0

for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    name="${names[$index]}"

    if wait "$pid"; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        status=1
    fi
done

exit "$status"