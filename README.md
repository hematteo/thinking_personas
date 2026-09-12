# Thinking Personas

**Does a reasoning model use the same persona while thinking and while answering?**

This project studies that question in **Qwen3-32B**, using the published Assistant Axis and role directions. Here, “thinking” means the text inside the model's `<think>` block; “answer” means the final response after it.

**We found no clear evidence of a distinct thinking persona.** Thinking and answer activations differ, but the differences depend on the task and also appear along control directions. Role instructions usually change thinking activations less than answer activations. Style instructions produce the same pattern.

## Why this question matters

The Assistant Axis is a direction in a model's activation space, computed from the difference between its default assistant responses and its roleplay responses. Higher projections point toward the default assistant end of that direction. It provides a way to study how a model responds to persona instructions.

If thinking responds differently from answers, tools used to monitor or steer persona in answers may need separate validation for thinking. This project measures that difference. It does not test whether applying those tools to thinking makes the model safer.

The [original proposal](thinking_persona_proposal.md) considers three possibilities: thinking keeps the default assistant persona, is persona-neutral, or uses a distinct persona. The experiments do not settle those identities. In particular, a projection near zero does not establish persona neutrality.

## Main findings

### 1. There is no uniform thinking–answer gap on the Assistant Axis

We compared thinking and answer activations under default instructions, using **40 retained WildChat advice prompts and 40 retained GSM8K math prompts**. For each prompt, we averaged token projections within thinking and within the answer, then took thinking minus answer.

| Prompt set | Mean thinking − answer projection | 95% confidence interval |
|---|---:|---:|
| WildChat advice | +0.03 | −1.53 to +1.59 |
| GSM8K math | −1.51 | −2.63 to −0.37 |

The advice estimate is close to zero; the math estimate is negative. Gaps along the Skeptic and Judge control directions are larger than the Assistant-axis gaps. Together, these results give no evidence of a consistent difference specific to the Assistant Axis across the two domains.

These numbers are **raw projections**, not centered cosine similarities. Raw projection is the dot product of a token activation with a fixed unit direction. The original analysis used centered cosine; the raw-projection analysis was added after the first experiment. Both measurements are retained. See [measurement details](docs/MEASUREMENT.md) and the [results and audit](docs/RESULTS_AND_AUDIT.md).

### 2. Role and style instructions affect answers more than thinking

A follow-up used nine roles, three style instructions, and a default assistant instruction on the same 12 questions, with two generation seeds: **312 responses**.

Six of the nine roles showed a larger shift away from the default along the Assistant Axis in answers than in thinking, after adjustment for the planned comparisons. Instructions to write in a particular style while keeping the AI identity produced the same pattern, sometimes more strongly.

Thinking is therefore less responsive to many of these instructions along this axis. The result is not specific to adopting a persona. See the [follow-up results](docs/FOLLOWUP_RESULTS.md) and [combined interpretation](docs/AGGREGATED_FINDINGS.md).

### 3. Role differences remain in thinking, but are smaller

PCA plots and measurements across all 5,120 activation dimensions show that the nine roles retain a partly shared arrangement in thinking and answers. Roles that are far apart in answers tend to be far apart in thinking too, but the thinking role averages are more tightly grouped.

The spread of the role averages is **8.37 in thinking and 16.51 in answers**: thinking has about half the spread. Removing the Assistant-axis component leaves a similar ratio, **0.53**. This difference extends beyond the Assistant Axis and is not just a feature of a two-dimensional PCA plot.

This is consistent with thinking acting as a more stable scratchpad or computation channel. A stable “attractor” is one possible explanation, but these measurements do not establish that mechanism or explain how training produced it. Thinking still changes under role instructions.

## How the experiments work

We generate responses with vLLM, then replay the exact token sequences through the model to measure activations. All reported experiments use Qwen3-32B in bf16 and the output of decoder block 32, counted from zero, after its MLP.

Generation settings are temperature 0.6, top-p 0.95, top-k 20, and a 2,048-token output limit. The first study uses seed 0; the role/style follow-up uses seeds 17 and 23.

The first study pairs a thinking-enabled response with a thinking-disabled response for each prompt. The headline table compares the thinking block with the final answer of the thinking-enabled response. Thinking-disabled responses provide an additional baseline. The advice prompts were screened for personal or practical advice before generation; they are not an unrestricted random sample of WildChat. Responses that hit the output limit or failed parsing were excluded, and retained thinking blocks had at least 200 tokens. The first five answer tokens were excluded from the headline segment means.

We measure the Assistant Axis, two controls derived from published Skeptic and Judge role directions, a direction made from randomly split calibration responses, and a random direction. Before the main study, a validation step checks that default answers score above roleplay answers on the Assistant Axis. A separate replay experiment places identical text under different prefixes to measure how context affects the scores.

Confidence intervals resample prompts or questions, not individual tokens. The [analysis protocol](docs/PROTOCOL.md), [data guide](docs/DATA.md), and [follow-up protocol](docs/FOLLOWUP_RUN_PROTOCOL.md) give the full design.

## Limitations and next steps

These results cover **one model, one layer, and small prompt sets**. The advice prompts were screened, and the follow-up used a fixed selection of roles and only 12 questions. The findings may not carry over to other models, layers, tasks, or personas.

The evidence is mainly descriptive. It does not identify a causal persona mechanism, establish persona-neutral thinking, or show that thinking has a distinct identity. The PCA analysis is exploratory. Systematic human review and behavioral validation remain incomplete.

Useful next steps are to test more models and layers, separate identity instructions from style with matched prompts, and intervene on activations to test whether the measured directions cause changes in behavior.

## Run the code

The repository is `thinking_personas`; the Python package is `persona_dynamics` and its command is `persona`.

Start with the synthetic smoke run. It checks that the software works; it provides no evidence about Qwen or personas.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,analysis]'
persona run --config configs/smoke.yaml
python scripts/verify_run.py runs/smoke
python -m pytest -q
```

Python 3.11 or later is required. The [running guide](docs/RUNNING.md) covers pinned dependencies, GPU setup, configuration changes, resuming runs, and reproducing figures. Real model experiments require separate model weights, vectors, prepared data, and GPU hardware. Optional model tests require additional dependencies.

## What is included

| Directory | Contents |
|---|---|
| `src/persona_dynamics/` | Generation, activation measurements, analysis, and reporting |
| `configs/` | Synthetic checks and experiment settings |
| `scripts/` | Data preparation, cluster launchers, figure builders, and verification |
| `tests/` | Tests of token handling, measurements, statistics, and pipeline behavior |
| `docs/` | Methods, results, limitations, and reproduction instructions |

**Saved datasets, model assets, run outputs, and generated figures are kept outside Git.** Paths under `data/`, `runs/`, and `artifacts/` in the detailed reports refer to the local research archive. Reproducing the recorded studies and their figures requires those files. The main smoke run above generates its own synthetic inputs; `followup_smoke.json` requires the separate follow-up input bundle.

The local figure READMEs explain the first experiment (`artifacts/paper-figures/README.md`), role geometry (`artifacts/role-geometry/README.md`), and example outputs (`artifacts/report-token-examples/README.md`). They are not bundled in a fresh clone.

The code builds on the [Assistant Axis repository and released vectors](https://github.com/safety-research/assistant-axis/tree/a98961956072224eaf244eb289d6c01700b63795). See the [verification record](docs/VERIFICATION.md) for the checks performed and their limits.
