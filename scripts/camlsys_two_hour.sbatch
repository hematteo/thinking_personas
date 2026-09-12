#!/bin/bash
#SBATCH --job-name=persona-e0-e1
#SBATCH --partition=ampere
#SBATCH --qos=cls_master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:a40:8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PERSONA_PROJECT="$PWD"
export PERSONA_START="$(date +%s)"
export PERSONA_SCRATCH="${PERSONA_CACHE_ROOT:-/local/scratch/mh2274/persona-${SLURM_JOB_ID}}"
export HF_HOME="$PERSONA_SCRATCH/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$PERSONA_SCRATCH/uv"
export TMPDIR="$PERSONA_SCRATCH/tmp"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=8
export VLLM_USE_V1=0
mkdir -p "$TMPDIR" "$HF_HOME" logs
if [[ ! -f data/prepared/prompts.jsonl && -f data/camlsys-prepared/advice_review.json ]]; then
  mkdir -p data/prepared
  cp -a data/camlsys-prepared/. data/prepared/
fi
date -Is
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
nvidia-smi topo -m
SHARED_PYTHON=/nfs-share/mh2274/readout-development/.venv/bin/python
export PERSONA_PYTHON="$PERSONA_SCRATCH/venv/bin/python"

# Keep the shared runtime unchanged. All large files stay on node-local SSD.
timeout --kill-after=30s 900 bash -c '
  /nfs-share/mh2274/bin/uv venv --python /usr/bin/python3 "$PERSONA_SCRATCH/venv"
  /nfs-share/mh2274/bin/uv pip install --python "$PERSONA_PYTHON" \
    vllm==0.9.2 transformers==4.53.3 numpy==2.2.6 pandas==2.3.3 \
    matplotlib==3.10.8 datasets==3.6.0 accelerate==1.10.1 pytest==8.4.2 PyYAML==6.0.2
' >logs/setup-runtime.log 2>&1 &
env_pid=$!
timeout --kill-after=30s 900 "$SHARED_PYTHON" -c '
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-32B", revision="9216db5781bf21249d130ec9da846c4624c16137", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"], max_workers=8)
' >logs/setup-model.log 2>&1 &
model_pid=$!
timeout --kill-after=30s 900 bash -c '
  /nfs-share/mh2274/readout-development/.venv/bin/python -m persona_dynamics fetch-assets
  if [[ ! -f data/prepared/prompts.jsonl ]]; then
    /nfs-share/mh2274/readout-development/.venv/bin/python scripts/prepare_camlsys.py
  fi
' >logs/setup-data.log 2>&1 &
data_pid=$!
setup_failed=0
for task_pid in "$env_pid" "$model_pid" "$data_pid"; do
  wait "$task_pid" || setup_failed=1
done
if (( setup_failed )); then
  echo 'SETUP_FAILED: see logs/setup-{runtime,model,data}.log'
  exit 1
fi
"$PERSONA_PYTHON" -c 'import importlib.metadata as m, json; print(json.dumps({d.metadata["Name"]: d.version for d in m.distributions()}, indent=2))' >logs/runtime-versions.json
"$PERSONA_PYTHON" -m pytest -q >logs/verification.log 2>&1
exec "$PERSONA_PYTHON" scripts/run_camlsys_stages.py
