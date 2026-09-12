#!/bin/bash
#SBATCH --job-name=persona-reserves
#SBATCH --partition=ampere
#SBATCH --qos=cls_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:a40:8
#SBATCH --time=01:20:00
#SBATCH --output=logs/reserves-%j.out
#SBATCH --error=logs/reserves-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PERSONA_START="$(date -d '2026-09-11T21:19:33+01:00' +%s)"
export PERSONA_SCRATCH=/local/scratch/mh2274/persona-55665
export PERSONA_PYTHON="$PERSONA_SCRATCH/venv/bin/python"
export HF_HOME="$PERSONA_SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$PERSONA_SCRATCH/tmp"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 VLLM_USE_V1=0
export PERSONA_CONFIG=configs/a40_two_hour_reserves.yaml
export PERSONA_EXTEND_FROM=runs/a40-two-hour
export PERSONA_LOGS=logs/reserves
mkdir -p "$PERSONA_LOGS"
exec "$PERSONA_PYTHON" scripts/run_camlsys_stages.py
