# Thinking Personas

**Does a reasoning model use the same persona while thinking and answering?**

We study **Qwen3-32B**, comparing activations during its `<think>` block with those during its final answer. We use the published [Assistant Axis](https://github.com/safety-research/assistant-axis/tree/a98961956072224eaf244eb289d6c01700b63795), a direction in activation space that separates default assistant responses from roleplay responses. Higher projections point toward the default assistant end of that direction.

**We find no clear evidence of a distinct thinking persona.** Role instructions often shift thinking activations less than answer activations, but style instructions produce the same pattern.

## Findings

**Thinking–answer differences depend on the task.** Under default instructions, we compared 40 retained WildChat advice prompts and 40 GSM8K math prompts:

| Prompt set | Mean thinking − answer projection | 95% confidence interval |
|---|---:|---:|
| Advice | +0.03 | −1.53 to +1.59 |
| Math | −1.51 | −2.63 to −0.37 |

These are raw Assistant-axis projections averaged within each segment of the same response. Larger gaps appear along the Skeptic and Judge control directions, so the difference is not specific to the Assistant Axis.

**Role instructions affect answers more than thinking in several cases.** A follow-up tested nine roles, three style instructions, and a default instruction on 12 questions with two seeds: 312 responses. Six of nine roles showed a larger shift away from default in answers after correction for multiple comparisons. Style instructions that preserve AI identity produced similar or larger differences.

**Role differences remain in thinking, but are smaller.** The spread of the nine role-average activation vectors is about half as large in thinking as in answers. This remains true after removing the Assistant-axis component. PCA plots show a partly shared arrangement of roles, not a separate thinking identity.

## Method and limits

We generate responses, then replay their exact tokens to measure activations at decoder block 32 (zero-indexed), after its MLP. Headline scores are dot products with unit directions; confidence intervals resample prompts or questions, not tokens.

These results cover **one model, one layer, and small, selected prompt sets**. The advice prompts were screened, and response-length filters affect the retained sample. The evidence is mainly descriptive: it does not establish a causal persona mechanism or persona-neutral thinking.

See [methods](docs/METHODS.md) for filtering, controls, the original cosine analysis, and the exploratory geometry analysis.

## Quick start

Requires Python 3.11 or later. This synthetic smoke run checks the software without model weights or a GPU; it does not test the research claims.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,analysis]'
persona run --config configs/smoke.yaml
python scripts/verify_run.py runs/smoke
python -m pytest -q
```

See [running the code](docs/RUNNING.md) and [preparing data](docs/DATA.md) for model experiments.

The repository includes code, configurations, tests, and documentation. **Datasets, model assets, saved runs, and generated figures are kept outside Git.** Reproducing the recorded studies requires their separate input and output files.
