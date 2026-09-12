# Running and reproducing Thinking Personas

See the [project README](../README.md) for the research question, findings, and limitations. This guide covers the implementation and reproduction commands. The Python package remains `persona_dynamics` and the command remains `persona`.

Saved inputs, runs, and generated figures are kept outside Git. Links into `data/`, `runs/`, and `artifacts/` refer to the local research archive. The main synthetic smoke run works without that archive; the recorded studies and their follow-up smoke configuration require it. Historical cluster paths must be adjusted for another machine.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke.lock
python -m pip install --no-deps -e .
persona run --config configs/smoke.yaml
python scripts/verify_run.py runs/smoke
python -m pytest -q
```

Alternatively install `python -m pip install -e '.[test]'` for compatible current dependencies. `requirements-smoke.lock` pins the exact packages used for the recorded offline verification. Torch/model tests are skipped if optional model packages are absent. Tests of a tiny random Qwen3 model require `.[models]`; the optional real-tokenizer integration uses `PERSONA_QWEN_TOKENIZER_DIR` pointing to a local Qwen3-32B tokenizer.

The completed smoke report is `runs/smoke/results.md`. It contains all six core figures, the two persona-space figures, and the norm diagnostic, each as PNG and vector PDF. `summary.json` and CSV tables are the machine-readable results. See [the verification record](VERIFICATION.md) for what was actually executed.

## Run Qwen3-32B

Use a Linux CUDA environment for vLLM. Generation and HF extraction run in separate processes, so their model copies do not occupy GPU memory together. The default is bf16, layer 32, 2,048 generated tokens, temperature 0.6, top-p 0.95, top-k 20, and seeds 0/1/2. The upstream model and vector revisions are pinned. Install a CUDA-compatible vLLM/PyTorch combination on the target machine and retain its package freeze alongside the recorded manifest.

```bash
python -m pip install -e '.[models,gpu,test]'
persona fetch-assets
python -m persona_dynamics.data prepare \
  --output data/prepared --advice /path/to/advice.jsonl \
  --curated-advice --fetch-public --seed 2026
persona run --config configs/qwen3_32b.yaml
python scripts/verify_run.py runs/qwen3-32b
```

The advice path is a real input, not a bundled replacement for WildChat/LMSYS. See [data preparation](DATA.md) for accepted exports, fixed authored AI-philosophy probes, dataset-revision flags, frozen role strata, and reserve candidates. The run stops if E0 fails or fewer than 40 paired prompts survive in any domain. Length-limited and malformed completions are rejected. An E4 run also requires at least six non-default roles plus default to complete the same planned extraction questions across all seeds; incomplete roles are recorded.

`backend: hf` provides an ordinary Transformers generation fallback. For Qwen3-8B, compute a **model-specific** axis and role/default vectors using the pinned upstream five-step pipeline; import them with the data module's `roles` command. Set the model/revision, paths, zero-indexed layer, and output directory in a new config. Never reuse the 32B vectors for 8B. Layer 32 is validated upstream for 32B; another model's layer is a new preregistered choice. No sub-8B model is a reporting configuration.

## Iterate and reproduce

Every public configuration field is documented by its name/default in `src/persona_dynamics/config.py`. Unknown keys fail. Overrides use YAML values:

```bash
persona run --config configs/qwen3_32b.yaml \
  --set 'seeds=[10,11,12]' --set output_dir=runs/replication

python scripts/sweep.py --config configs/smoke.yaml \
  --grid configs/ablations.json --output runs/ablations

persona analyze runs/smoke
```

The example grid runs all-answer-token, short-think, and E4-omitted sensitivities. Change model/layer/temperature/filter settings only in named, separately reported runs. Layers can be swept this way for the appendix; no layer-sweep p-values are generated. Baselines and E2 controls run automatically; `experiments: [E1,E2,E3]` omits E4 while retaining the essential role/null/random controls.

Prefer running multiple seeds in one config: they share calibration, cohort selection, and PCA frame. `persona aggregate RUN_A RUN_B --output runs/pooled` supports disjoint-seed runs with identical inputs, settings, code, and measured directions. It rejects incompatible rulers and duplicate observations rather than pooling different geometries. A changed PCA frame normally makes independently fitted E4 runs incompatible.

Re-running the identical command resumes transcript and extraction caches. Config, source files, input contents, and numerical dependency versions form the run identity; changing them requires a new output directory. Each transcript has stable generation seeds, exact prefix/completion token IDs, finish status, and source identity. Resume preserves manual review files. No token hidden-state arrays are persisted; E4 and calibration retain only small segment mean vectors.

Individual stages are available for debugging:

```bash
persona worker --config runs/smoke/config.json --stage generate-gate
persona worker --config runs/smoke/config.json --stage gate
persona worker --config runs/smoke/config.json --stage generate-experiments
persona worker --config runs/smoke/config.json --stage extract
persona worker --config runs/smoke/config.json --stage analyze
```

E0 is enforced in both generation and extraction entry points. Do not bypass a failed gate to obtain headline figures.

## Interpret the outputs

| Artifact | Use |
|---|---|
| `manifest.json`, `config.json` | Frozen settings, sources, code hashes, runtime versions |
| `transcripts/*.json` | Exact token records, generation seeds, parsing/finish status |
| `e0.json`, `e0_tokens.csv.gz` | Axis gate and auditable validation measurements |
| `directions*.npz` | Unit vectors, fixed center, control/calibration provenance |
| `tokens.csv.gz` | One row per measured token in the retained cohort |
| `attrition.csv`, `cohort.json`, `role_retention.csv` | Every candidate's eligibility, reasons, and retained sets |
| `e1_effects.csv`, `specificity.csv` | Paired cosine gaps, standardized effects and control contrasts |
| `e2_effects.csv` | Paired tag/content factorial and delimiter-control contrasts |
| `role_shifts.csv`, `role_regression.csv` | Default-matched role shifts and descriptive stiffness regression |
| `boundary_first5.csv` | Boundary, first five, main answer, and all-answer diagnostics |
| `persona_space.json`, `persona_frame.npz` | Common PCA frame, alignment, cloud spread/translation statistics |
| `review/`, `human_notes.md` | Blank human calibration and full-CoT review materials |
| `results.md`, `summary.json`, `figures/` | Regenerable report, limitations, tables, PNG/PDF figures |

Read [the analysis protocol and assumptions](PROTOCOL.md) before interpreting results. CIs resample prompts or roles, never tokens. Multiple seeds are averaged within each prompt before pooled uncertainty; per-seed estimates remain available. Zero-centered cosine is an operational coordinate, not evidence of persona neutrality. A small random-direction gap is insufficient specificity evidence. The report does not automatically assign H1/H2/H3 or turn synthetic results into claims.

[Judge and human review instructions](JUDGE.md) explain the fixed rubric, required 30 real human labels, agreement metrics, optional API execution, label imports, and refusal flags. Until labels are supplied, the scatter explicitly reports the missing judge; no human agreement is invented. Selected real outputs have been inspected and the study has been written up. Systematic human review and judge calibration remain incomplete.

## Code map

`data.py` prepares immutable prompt/role sets; `transcripts.py` handles token spans and transplants; `backends.py` separates generation from synthetic fixtures; `geometry.py` implements scalar hooks and PCA; `pipeline.py` coordinates gates, retention and resumable stages; `analysis.py` contains paired estimators; `plots.py` renders figures; `judge.py` manages sentence labels and review. Tests focus on causal/token correctness, statistical units, leakage, gates, and artifact integrity.

The design follows the [Assistant Axis code and released artifacts](https://github.com/safety-research/assistant-axis/tree/a98961956072224eaf244eb289d6c01700b63795), with the main layer convention verified against its pipeline and model config. Qwen's [model card](https://huggingface.co/Qwen/Qwen3-32B) defines the thinking toggle and generation settings. The upstream repository is downloaded as a pinned reference rather than installed as an additional abstraction layer.

## Completed CamlSys pilot and audit

The actual Qwen3-32B E0/E1/E2 run is complete. Read [the results and code audit](RESULTS_AND_AUDIT.md) and [the completed write-up](WRITEUP_TWO_HOUR.md). The Assistant-axis gaps do not exceed structured controls; the study does not support a persona-specific interpretation. The corrected report is in `artifacts/code-audit/reanalysis-final`; the original `runs/a40-two-hour-reserves` artifacts and original source snapshot remain frozen. Post-audit fixes require a new output directory for future runs.

## Raw token projections

For subsequent work, use the [linear-projection measurement contract](MEASUREMENT.md), aligned with the Assistant Axis paper's drift readout. Raw `dot_*` values were already retained per measured token. Export them without averaging or applying inclusion flags:

```bash
python scripts/export_raw_projections.py runs/a40-two-hour-reserves --output artifacts/new-raw-export
```

A verified export is available in `artifacts/raw-projections`. The original cosine analysis remains historical; raw-projection post-processing is separate and requires no new GPU run for the saved spans/directions.

## Complementary role experiment

The [frozen follow-up protocol](FOLLOWUP_RUN_PROTOCOL.md) tests whether role prompts change thinking and answers equally. It uses nine published roles, the default assistant, three style controls, 12 matched questions and seeds 17/23: 312 main responses. Raw projections are primary; centered cosine and skipping the first five answer tokens are sensitivities. This is a focused E3 study, with its own configuration and analyzer.

**Completed:** [results, interpretation and independent audit](FOLLOWUP_RESULTS.md). All 312 responses were measured in a 10-minute-28-second eight-A40 allocation. Several roles affect answers more than thinking, but style-only controls reproduce or exceed that asymmetry; it is insufficient evidence for a distinct internal persona. [Full sample outputs](../artifacts/role-followup-supplement/balanced_transcripts.md) and [the frozen numerical report](../runs/role-susceptibility-20260912/results.md) are available.

The [CPU-only role-geometry report](../artifacts/role-geometry/README.md) adds shared PCA maps, full-dimensional effect sizes, similarity clustering, domain comparisons and seed checks. The thinking role-centroid spread is about half the answer spread, with correlated role-distance patterns; the result persists after removing the Assistant-axis component. Role differences are smaller in thinking but remain structured. This is an exploratory result for the nine tested roles; it does not establish a distinct thinking persona. Install `.[analysis]` or `requirements-analysis.lock` to reproduce them with `scripts/postprocess_role_geometry.py`.

The [literature figure adaptations](../artifacts/literature-figures/README.md) add scree curves, an overlay in the 275-role released Qwen PCA space, PC-alignment plots, matched token heatmaps, and activation-colored text. Open the [five-page PDF](../artifacts/literature-figures/literature_figures.pdf) or the [self-contained token viewer](../artifacts/literature-figures/token_viewer.html), which includes all 312 responses. Reference vectors are pinned and hashed; the external PCA is fitted without using our experimental responses. The report documents differences from the paper's fitting set and the lack of browser visual verification.

```bash
# Offline pipeline check: choose a fresh output_dir in this config for a new run.
python -m persona_dynamics.followup --config configs/followup_smoke.json --stage init
python -m persona_dynamics.followup --config configs/followup_smoke.json --stage generate-pilot
python -m persona_dynamics.followup --config configs/followup_smoke.json --stage validate
for seed in 17 23; do
  python -m persona_dynamics.followup --config configs/followup_smoke.json --stage generate --seed "$seed"
  python -m persona_dynamics.followup --config configs/followup_smoke.json --stage extract --seed "$seed"
done
python -m persona_dynamics.followup --config configs/followup_smoke.json --stage analyze
python scripts/verify_followup.py runs/followup-smoke
python scripts/audit_followup_results.py runs/followup-smoke --output artifacts/new-followup-audit
```

For CamlSys, `scripts/followup_two_hour.sbatch` records the actual eight-A40 launcher, including the existing checkpoint/environment cache on `mauao`. Adjust that cache path if provisioning another environment. `scripts/run_followup_two_hour.py` enforces stage deadlines and requires fresh pilot generation plus three numerical reference replays before the main study. The original study is preserved. Use `python -m persona_dynamics.followup --stage analyze --config CONFIG` in the frozen runtime to reproduce follow-up analysis; the older `persona analyze` command handles a different study schema.

## Paper figures

The [captioned figure report](../artifacts/paper-figures/README.md) and [combined PDF](../artifacts/paper-figures/paper_figures.pdf) contain three main figures and six appendix diagnostics. Individual PNGs/PDFs and source CSV tables are included. The main figures show E1 paired prompt gaps, E2 identical-content context effects, and token trajectories, each with the control directions visible. All inference uses prompts as the sampling unit, with fixed bootstrap seeds.

```bash
python scripts/build_paper_figures.py runs/a40-two-hour-reserves --output artifacts/new-paper-figures
python scripts/verify_paper_figures.py runs/a40-two-hour-reserves artifacts/new-paper-figures
```

Use a fresh output directory. The builder performs CPU post-processing only and preserves the original measurements and cosine analysis. The raw-projection analysis is a post hoc methodological amendment, documented in the report.
