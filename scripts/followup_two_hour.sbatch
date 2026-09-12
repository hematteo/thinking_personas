#!/bin/bash
#SBATCH --job-name=persona-role2
#SBATCH --partition=ampere
#SBATCH --qos=cls_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a40:8
#SBATCH --nodelist=mauao
#SBATCH --time=02:00:00
#SBATCH --output=logs/followup-%j.out
#SBATCH --error=logs/followup-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PERSONA_SCRATCH=/local/scratch/mh2274/persona-55665
export HF_HOME="$PERSONA_SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="$PERSONA_SCRATCH/tmp"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_USE_V1=0
export MPLCONFIGDIR="$PERSONA_SCRATCH/mpl-followup"
mkdir -p "$MPLCONFIGDIR"
exec timeout --signal=TERM --kill-after=15 7170 "$PERSONA_SCRATCH/venv/bin/python" scripts/run_followup_two_hour.py
