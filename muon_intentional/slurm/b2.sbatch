#!/bin/bash
#SBATCH --job-name=mi-b2
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
source /etc/profile.d/llm-conda.sh
conda activate llmenv
cd /home/15Cu2Plus/muon_intentional
echo "host=$(hostname) job=$SLURM_JOB_ID"
srun python -I scripts/run_experiments.py --stage b2 --mode full
