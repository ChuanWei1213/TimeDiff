#!/usr/bin/env bash
# Run from the app terminal to see live tqdm progress; also retain a full log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tag="${1:-crash_20260929}"
if [[ ! "$tag" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Run name must contain only letters, numbers, underscores or hyphens.' >&2
  exit 2
fi
python_bin="${SPX_PYTHON:-python3}"
run_dir="outputs/spx/$tag"
for required in datasets/spx/prices.csv outputs/spx/initial/model.pt outputs/spx/initial/train_conditions.csv; do
  if [[ ! -f "$required" ]]; then
    echo "Missing input: $required" >&2
    exit 1
  fi
done
# Atomic creation prevents accidental double launches or overwriting prior runs.
mkdir "$run_dir"
exec > >(tee -a "$run_dir/run.log") 2>&1
trap 'echo "Experiment failed at line $LINENO; inspect $run_dir/run.log" >&2' ERR
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/$run_dir/matplotlib}"
export IPYTHONDIR="${IPYTHONDIR:-$PWD/$run_dir/ipython}"
export SPX_CRASH_RUN="$PWD/$run_dir"

conditions=(--condition -0.05 0.15 --condition 0.05 0.15
            --condition -0.05 0.35 --condition 0.05 0.35
            --condition -0.10 0.40 --condition -0.20 0.60 --condition -0.30 0.80)

echo "SPX experiment: $SPX_CRASH_RUN"
echo 'Stage 1/4: retrain without mixup (AR 500, diffusion 2000, seed 2024)'
"$python_bin" -u spx_experiment.py train \
  --train-end 2021-12-31 --val-end 2023-12-31 \
  --ar-steps 500 --steps 2000 --eval-every 100 --val-batches 4 \
  --batch-size 32 --seed 2024 --device cpu --out "$run_dir/train"

echo 'Stage 2/4: sample seven conditions from the retrained model (200 paths each)'
"$python_bin" -u spx_experiment.py sample \
  --checkpoint "$run_dir/train/model.pt" --history-end 2023-12-29 \
  "${conditions[@]}" --n-paths 200 --batch-size 32 --seed 2024 --device cpu \
  --out "$run_dir/no_mixup_samples"

echo 'Stage 3/4: sample the same conditions from the existing mixup baseline'
"$python_bin" -u spx_experiment.py sample \
  --checkpoint outputs/spx/initial/model.pt --history-end 2023-12-29 \
  "${conditions[@]}" --n-paths 200 --batch-size 32 --seed 2024 --device cpu \
  --out "$run_dir/mixup_samples"

echo 'Stage 4/4: execute the crash comparison notebook and validate metrics'
"$python_bin" notebooks/execute_in_process.py notebooks/spx_crash_comparison.ipynb
echo "Complete. Results: $SPX_CRASH_RUN/comparison"
