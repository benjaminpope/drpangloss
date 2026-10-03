#!/bin/bash
# Re-fit Shashank Dholakia's four CHARA/PAVO rapid rotators with
# GravityDarkenedStar on OzSTAR/NT: one star per array task, his NUTS
# settings (2000 warmup + 2500 samples, 2 chains in parallel on CPU).
#
# Submit from the repository root on an NT login node (tooarrana):
#     python examples/elr_pavo/fit_pavo.py --download-only   # compute nodes
#                                                            # may be offline
#     mkdir -p logs && sbatch examples/elr_pavo/fit_pavo.sbatch
# then, when all four tasks have finished:
#     python examples/elr_pavo/compare.py
#
#SBATCH --job-name=elr-pavo
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module -q load conda
conda activate drpangloss

STARS=(upsTau upsUMa epsCep lamBoo)
STAR=${STARS[$SLURM_ARRAY_TASK_ID]}
echo "task=${SLURM_ARRAY_TASK_ID} star=${STAR} host=$(hostname)"

# Use this checkout's source, whatever drpangloss the env has installed.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# CPU job: keep JAX off any GPU. Chains are vectorized (vmap) rather than
# parallel (pmap): with the env's jax/equinox, pmap breaks optimistix's
# root finder ("Closure-converted function called with different dynamic
# arguments").
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1  # log progress as it happens

srun python examples/elr_pavo/fit_pavo.py --stars "$STAR" --no-progress-bar \
  --chain-method vectorized
