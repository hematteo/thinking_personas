# Running Thinking Personas

Run these commands from the repository root. Python 3.11 or later is required. The package is `persona_dynamics` and the command is `persona`.

## Install and check the software

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,analysis]'
persona run --config configs/smoke.yaml
python scripts/verify_run.py runs/smoke
python -m pytest -q
```

The smoke run uses synthetic data and needs no model weights or GPU. It checks the software, not the research claims. Its report is saved to `runs/smoke/results.md`.

The files `requirements-smoke.lock` and `requirements-analysis.lock` record the packages used in the original local environment. Optional model tests require the model dependencies.

## Run Qwen3-32B

Use a Linux CUDA environment with enough GPU memory for the model. Install compatible PyTorch and vLLM packages for your hardware, then install the project dependencies:

```bash
python -m pip install -e '.[models,gpu,test]'
persona fetch-assets
python -m persona_dynamics.data prepare \
  --output data/prepared --advice /path/to/advice.jsonl \
  --curated-advice --fetch-public --seed 2026
persona run --config configs/qwen3_32b.yaml
python scripts/verify_run.py runs/qwen3-32b
```

Replace the advice path with your input file. See [data preparation](DATA.md). Model weights and research datasets are separate downloads and are not included in Git.

The full configuration is in `configs/qwen3_32b.yaml`; available settings and defaults are in `src/persona_dynamics/config.py`. This configuration supports a broader experiment than the completed study summarized in the README. Check its domains, roles, seeds, and output directory before starting.

Use a new output directory when changing settings or inputs:

```bash
persona run --config configs/qwen3_32b.yaml --set output_dir=runs/new-study
```

Repeating the same command with unchanged settings resumes cached work. A run stops if validation fails or too few valid responses remain.

## Read the outputs

| File | Contents |
|---|---|
| `results.md` | Summary report |
| `summary.json` | Results in machine-readable form |
| `config.json`, `manifest.json` | Settings, data sources, and software versions |
| `transcripts/` | Generated text and exact token records |
| `tokens.csv.gz` | Measurements for each recorded token |
| `figures/` | Plots |

Rebuild the standard report from a completed run with:

```bash
persona analyze runs/new-study
```

The role/style study uses a separate analyzer:

```bash
python -m persona_dynamics.followup --config configs/followup_two_hour.json --stage analyze
```

That command requires the saved follow-up inputs and measurements. The full study archive and generated figures are kept outside Git. The [methods](METHODS.md) explain what the measurements mean.
