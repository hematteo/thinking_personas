# Preparing data

The synthetic smoke run creates its own inputs. Model experiments need separate prompt files, model weights, and the published Assistant Axis vectors.

## Download reference assets

From the repository root, with the model dependencies installed:

```bash
persona fetch-assets
```

This downloads the pinned role instructions, questions, and Qwen3-32B vectors. It does not download model weights. The code uses the [Assistant Axis release](https://github.com/safety-research/assistant-axis/tree/a98961956072224eaf244eb289d6c01700b63795).

## Prepare prompts

Provide an advice file with one JSON object per line:

```json
{"text":"I feel overwhelmed by work and want help making time for friends. What should I consider?","source":"researcher-curated-advice-v1"}
```

Then run:

```bash
python -m persona_dynamics.data prepare \
  --output data/prepared --advice /path/to/advice.jsonl \
  --curated-advice --fetch-public --seed 2026
```

Replace the advice path with your file. The command downloads GSM8K and MBPP benchmark inputs and records their sources. Reference answers are kept as metadata and are not included in generation prompts.

To use local benchmarks instead, replace `--fetch-public` with `--benchmarks /path/to/benchmarks.jsonl`. Each row needs `prompt_id`, `domain` (`math` or `code`), `text`, and `source`.

To import a local WildChat or LMSYS export, omit `--curated-advice`. The importer accepts `conversation` or `conversations` records and selects the first user turn. It applies a basic advice filter; inspect the retained prompts before running the model. It does not download conversation exports automatically.

## Check the prepared inputs

Prepared files record the exact prompt text, source, selection order, and role instructions. Calibration, validation, and evaluation inputs are kept separate. Duplicate normalized prompt text is rejected.

The full configuration supports advice, math, code, and authored AI-philosophy probes. The completed default-instruction study reported in the README used advice and math only. Check the domains and sample sizes in your configuration before generation.

Screening and response-length filters affect which prompts enter the final analysis. Report exclusions as well as retained counts. The saved advice set was screened for relevant content and should not be described as an unrestricted random sample of WildChat.

Prepared data and saved model outputs are kept outside Git. Reproducing a recorded study requires its original input files. See [running the code](RUNNING.md) for the next steps.
