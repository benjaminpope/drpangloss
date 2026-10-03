#!/bin/bash
# PAVO robustness tests (local minima; uv-coverage vs fitted PA) on
# OzSTAR/NT, one array task per slice of work. See multistart.py,
# profile_like.py and injection.py (all take --task-index/--n-tasks), then
# summarise.py locally.
#
# Task layout, per star (4 stars in the order of STARS below; 46 tasks per
# star, 184 in all, indices 0-183; star s owns indices 46*s ... 46*s+45):
#
#   tasks  work (fits per task, all float64 LM fits)               
#   ----   -----------------------------------------------------------
#      2   multistart, 64 starts                      (32 fits)
#     12   profile pa,inc   (36x19 grid x 3 starts)  (~171 fits)
#      8   profile omega,inc (21x19 grid x 3 starts) (~150 fits)
#      4   profile diam,omega (9-19 x 21 grid x 3 starts) (~100-170 fits)
#     12   injection, real: 100 injections x 8 starts (~67 fits)
#      8   injection, null: 100 injections x 8 starts (~100 fits)
#
# Timing (laptop, one fit with 4 free parameters; scale ~1.5x for a
# cluster core): upsTau 10 s, lamBoo 15 s, upsUMa 22 s, epsCep 50 s (its
# fits mostly run to the 400-step limit). A task is 30 min (upsTau) to
# ~3 h (epsCep, the slowest) of work; --time=08:00:00 leaves headroom.
#
# Submit from the repository root on an NT login node (tooarrana), after
# fetching the data once (compute nodes may be offline):
#     python examples/elr_pavo/fit_pavo.py --download-only
#     mkdir -p logs
#     sbatch examples/elr_pavo/robustness/robustness.sbatch      # all 184
#     sbatch --array=0-45 examples/elr_pavo/robustness/robustness.sbatch  # upsTau only
#     sbatch --array=92-137 ...                                  # epsCep only
# Then fetch examples/elr_pavo/robustness/output/ and run, locally:
#     python examples/elr_pavo/robustness/summarise.py
# Check the layout without running anything:
#     ROBUST_DRY=1 SLURM_ARRAY_TASK_ID=100 bash examples/elr_pavo/robustness/robustness.sbatch
#
#SBATCH --job-name=elr-robust
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-183%48
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

STARS=(upsTau upsUMa epsCep lamBoo)
# kind, argument, number of tasks
TABLE=(
  "multistart - 2"
  "profile pa,inc 12"
  "profile omega,inc 8"
  "profile diam,omega 4"
  "injection real 12"
  "injection null 8"
)

# Decode the array index into (star, kind, arg, slice).
idx=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID}
per_star=0
for row in "${TABLE[@]}"; do per_star=$((per_star + ${row##* })); done
STAR=${STARS[$((idx / per_star))]}
local_idx=$((idx % per_star))
for row in "${TABLE[@]}"; do
  read -r KIND ARG NT <<< "$row"
  if (( local_idx < NT )); then TASK=$local_idx; break; fi
  local_idx=$((local_idx - NT))
done

DIR=examples/elr_pavo/robustness
case $KIND in
  multistart) CMD=(python $DIR/multistart.py --n-starts 64) ;;
  profile)    CMD=(python $DIR/profile_like.py --pair "$ARG") ;;
  injection)  CMD=(python $DIR/injection.py --n-inject 100)
              [[ $ARG == null ]] && CMD+=(--null) ;;
esac
CMD+=(--star "$STAR" --task-index "$TASK" --n-tasks "$NT")
echo "task=${idx} star=${STAR} kind=${KIND} ${ARG} slice=${TASK}/${NT} host=$(hostname)"
echo "${CMD[*]}"
[[ -n $ROBUST_DRY ]] && exit 0

module -q load conda
conda activate virgil

# Use this checkout's source, whatever virgil the env has installed.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu  # CPU job: keep JAX off any GPU
export PYTHONUNBUFFERED=1  # log progress as it happens

srun "${CMD[@]}"
