# Data and frozen selection

Run from the repository root after activating the environment and installing model/data dependencies:

```bash
source .venv/bin/activate
python -m pip install -e '.[models,test]'
python -m persona_dynamics fetch-assets --output data/assets --prepared data/prepared --seed 2026
python -m persona_dynamics.data prepare \
  --output data/prepared --advice /path/to/curated_advice.jsonl \
  --curated-advice --fetch-public --seed 2026
```

`fetch-assets` retrieves the pinned [upstream role instructions and extraction questions](https://github.com/safety-research/assistant-axis/tree/a98961956072224eaf244eb289d6c01700b63795/data) and the released [Qwen3-32B vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors/tree/main/qwen-3-32b). It downloads roughly 180 MB of vectors, not model weights. All source prompts are preserved verbatim, including five system-prompt variants. The experiment uses variant zero for each role. Using all five would multiply the intended 60 × 5 experiment by another factor of five.

There is no upstream role-projection JSON file. The importer calculates sampling scores from the **released** per-role vectors dotted with the unit released Assistant Axis at fixed layer 32. These scores never use this experiment's generations. Roles are sorted by score, divided into three equal-size rank strata, and 20 are sampled without replacement per stratum with a fixed seed. `roles.jsonl` retains all imported roles for E0; `selected_roles.jsonl` records the planned E3 subset. Five shared extraction questions are sampled once from the upstream question pool. The main run records its own frozen role selection using the configured analysis seed; keep the preparation and analysis seeds equal.

`prepare` downloads [GSM8K test](https://huggingface.co/datasets/openai/gsm8k) and [MBPP full/test](https://huggingface.co/datasets/google-research-datasets/mbpp). It records dataset fingerprints and accepts `--gsm8k-revision SHA` and `--mbpp-revision SHA` for immutable dataset revisions. The prompt manifest itself is saved and content-hashed in each run. Benchmark reference answers/tests remain metadata and are never included in generation prompts. MBPP test code is not executed by this repository.

To work without downloads, replace `--fetch-public` with `--benchmarks /path/to/benchmarks.jsonl`. Each benchmark row needs `prompt_id`, `domain` (`math` or `code`), `text`, and `source`. Additional reference metadata is preserved.

A curated advice file contains one object per line:

```json
{"text":"I feel overwhelmed by work and want help making time for friends. What should I consider?","source":"researcher-curated-advice-v1"}
```

Alternatively, omit `--curated-advice` to import a local WildChat/LMSYS export with `conversation` or `conversations` records. The filter uses the first user turn, an English language field when present, 40–4000 characters, and a visible personal-advice keyword rule in `data.py`. Unknown language is treated as English for this convenience filter; inspect retained rows by hand. Duplicate normalized text is removed within the advice importer. No conversation export is downloaded automatically. Review local exports for private information before intentionally using an external judge.

Preparation reserves **eight calibration and five E0 prompts per math/code/advice domain**, with no overlap in normalized text anywhere in the manifest. Calibration freezes the mean center and null direction. E0 checks the ruler before evaluation. The remaining rows receive fixed per-domain `candidate_rank` values. Generation and length filtering use this frozen order: target 50 prompts paired across all configured seeds for natural think/no-think; take sequential reserves up to the configured cap; fail below 40 survivors. The minimum-length filter changes the population being studied. Report counts, reasons, and shortages from `attrition.csv` and `cohort.json`.

The AI-philosophy domain contains exactly 50 fixed repository-authored prompts, all assigned to evaluation. This is a material deviation from the proposal's suggested LLM-generated set: it is a designed probe set, not a random sample of AI conversations. It has **no built-in reserves** and cannot guarantee 40 surviving long CoTs. If it falls below the threshold, supply an expanded, documented corpus and start a new run before examining projections; do not silently lower the filter or reuse evaluation prompts for calibration. Calibration and E0 consequently represent the other three domains only.

The `prompts.jsonl` schema is:

| Field | Meaning |
|---|---|
| `prompt_id` | Unique stable task identifier |
| `domain` | `math`, `code`, `advice`, or `ai_philosophy` |
| `text` | Exact user task text |
| `source` | Dataset/export/author provenance |
| `split` | `calibration`, `e0`, or `eval` |
| `candidate_rank` | Fixed order within domain and split |
| `text_sha256` | NFKC-normalized, case-folded, whitespace-collapsed text hash |

`role_questions.jsonl` uses `domain=role, split=role` and is kept separate. Exact text and ID duplicates are rejected, including across partitions; this does not detect paraphrase leakage. The provided `offline_fixture_prompts` and `offline_fixture_roles` functions are explicitly synthetic smoke-test data. They cannot support scientific claims or replace a missing corpus.
