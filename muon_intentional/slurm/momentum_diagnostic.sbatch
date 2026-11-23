#!/bin/bash
#SBATCH --job-name=b1-momentum-diag
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
set -eo pipefail
source /etc/profile.d/llm-conda.sh
conda activate llmenv
cd "$(git rev-parse --show-toplevel)/muon_intentional"
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python -u scripts/momentum_diagnostic.py --config configs/b1_momentum_diagnostic.json "$@"
