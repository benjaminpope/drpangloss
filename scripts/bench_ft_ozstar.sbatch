#!/bin/bash
# GPU benchmark of the image Fourier transforms (scripts/bench_ft.py --gpu),
# for Swinburne's NT/OzSTAR cluster. It reports DFT and NUFFT timings, checks
# that the DFT is not computed in TF32, and measures FINUFFT's GPU accuracy.
#
# One-time setup, on a tooarrana login node, in $MYFRED/code/drpangloss
# (a clone of the repo on the imaging branch). pip wheels of jax-finufft are
# CPU-only, so the environment comes from conda-forge, and jax, jaxlib and
# jax-finufft must all come from there to match each other:
#
#   module -q load conda
#   # The login nodes have no GPU. Tell conda which CUDA the GPU driver
#   # supports: check `nvidia-smi` in a `gpushell` (CUDA Version, top
#   # right). The cuda129 builds need a driver for CUDA >= 12.9.
#   export CONDA_OVERRIDE_CUDA=12.9
#   conda create -n drpangloss-nufft -c conda-forge python=3.12 \
#       "jaxlib=*=cuda*" jax "jax-finufft>=1.3.1=cuda*"
#   conda activate drpangloss-nufft
#   pip install uv
#   UV_CACHE_DIR=$MYFRED/.cache/uv uv pip install \
#       --python $CONDA_PREFIX/bin/python -e ".[dev]"
#   # uv must not replace the conda jax: check it is still the CUDA build.
#   python -c "import jax, jax_finufft; print(jax.__version__, jax_finufft.__file__)"
#
# Then submit from the repository root:
#
#   mkdir -p logs && sbatch scripts/bench_ft_ozstar.sbatch
#
# (Slurm does not create the logs/ directory itself.)
#
# and send back logs/bench-ft-<jobid>.out and bench_ft_gpu.json.

#SBATCH --job-name=bench-ft
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

module -q load conda
conda activate drpangloss-nufft
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# The cluster environment sets JAX_DEFAULT_MATMUL_PRECISION=highest, which
# would hide the problem this job tests for. Use JAX's own default (TF32 on
# A100/H100), so the check shows that drpangloss asks for HIGHEST itself.
export JAX_DEFAULT_MATMUL_PRECISION=default

echo "job=${SLURM_JOB_ID} host=$(hostname) env=${CONDA_DEFAULT_ENV} start=$(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import jax; print(jax.__version__, jax.devices())"

srun python scripts/bench_ft.py --gpu --npix 64 128 256 512 \
    --out bench_ft_gpu.json

echo "end=$(date)"
