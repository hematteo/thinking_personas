# Measurement choice and raw-data contract — 12 September 2026

Following the user's request after the initial results, the preferred geometric readout for subsequent work is the **unaveraged linear projection** of each measured token onto the fixed unit Assistant direction:

`dot_assistant[t] = h[t] @ assistant_unit_vector`

Apply the same readout to each structured control. There is no activation centering, activation-norm division, smoothing, token averaging or baseline subtraction at this stage. Normalizing the axis itself fixes the ruler's units; it does not normalize each activation. The signed score is an activation coordinate, not a probability or calibrated degree of persona.

This is a dated methodological amendment, not the original proposal's preregistered analysis. The proposal requested centered cosine, which the initial pipeline implemented. The previous conclusions are conclusions about that metric. Do not present them as established for raw projections until that separate analysis is performed. Preserve both results, including disagreements, instead of choosing a metric by its outcome.

## What the literature actually measures

| Work | Relevant readout | Implication for this project |
|---|---|---|
| Lu et al., *The Assistant Axis* (2026), §4.1, §4.3 and Appendix E.3 | Linear projection of mean response activations for persona drift; cosine elsewhere for comparing role/trait vectors and axes | Closest reference for our Assistant-axis measurement. Retain token projections and perform response/segment aggregation afterward. |
| Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024), §5.1/Figure 5 | Cosine similarity between last-token residual activations and the refusal direction | Cosine has precedent for a different question and token selection; it is not the Assistant Axis drift readout. |
| Zou et al., *Representation Engineering* (2023), §3.1.1 and Appendix C.1 | Dot-product readout along a learned reading vector; the detailed pipeline also specifies normalization using training-derived parameters | Linear readouts are common, but preprocessing varies. Do not describe every dot-product method as an unprocessed activation score. |

Primary sources: [Assistant Axis](https://arxiv.org/html/2601.10387v1#S4), [refusal direction](https://arxiv.org/html/2406.11717v1#S5), [Representation Engineering](https://arxiv.org/html/2310.01405v4#S3). The [Persona Selection Model](https://alignment.anthropic.com/2026/psm/) is a conceptual framework and evidence review, not a separate calibrated scalar measurement of Assistant-ness.

## Collection and analysis are separate

The original extraction already saved one scalar dot product per direction per measured token. Segment averaging happened afterward in `analysis.py`; it did not replace the token data. The new export simply makes that separation explicit.

A later mean of token projections equals projecting the mean activation, for the same token set and weighting:

`mean_t(h[t] @ a) = mean_t(h[t]) @ a`

Thus storing token projections preserves the information needed for the reference paper's linear response summary. Means are still useful for comparisons between prompts of different lengths. Tokens from the same prompt must not become independent statistical replicates. Token plots, medians, quantiles, windows, first-five inclusion and paired prompt summaries can all be chosen explicitly in post-processing. A fixed scalar baseline also cancels from paired linear differences; it need not enter collection.

The fixed axis, layer, exact content/context design and control directions stay the same. Reusing the reference readout does not make our E1/E2 design an exact replication of that paper's multi-turn conversation experiment. Centered cosine remains available as a secondary sensitivity measure in the original token table; it should not be called the raw projection.

## Export and inspect the existing run

```bash
python scripts/export_raw_projections.py runs/a40-two-hour-reserves \
  --output artifacts/raw-projections
```

The completed export is `artifacts/raw-projections/raw_token_projections.csv.gz`, with **354,533 rows across 480 measured transcripts**. It retains all five `dot_*` columns, raw activation norm, model/layer/seed, prompt and record identifiers, token IDs and absolute/segment positions. Existing first-five and primary-inclusion flags remain metadata; the export applies none of those filters. Boundary rows and all previously measured first-five answer tokens are retained.

`artifacts/raw-projections/manifest.json` records source/export hashes, the formula, coverage and exact CSV numeric round-trip validation. Use a fresh output directory to repeat the export. No GPU inference or new statistical aggregation is performed.

```python
import pandas as pd
raw = pd.read_csv(
    'artifacts/raw-projections/raw_token_projections.csv.gz',
    float_precision='round_trip',
)
# Inspect individual measurements, without aggregating or applying inclusion flags.
print(raw[['prompt_id', 'condition', 'token_idx', 'token_id',
           'segment', 'dot_assistant', 'dot_ctrl_1', 'dot_ctrl_2',
           'dot_null', 'dot_random']].head())
```

“Raw” here means uncentered, unnormalized scalar projections. The original run did not retain full 5,120-dimensional hidden vectors, unselected candidate activations, all prompt tokens, or excluded formatting-token activations. Exact generation token IDs are available separately. New directions, new layers, cosine of an averaged hidden vector, or newly measured spans would require another extraction pass. Changing only aggregation of saved projections does not.

Do not rerun the historical `persona analyze` command expecting raw-projection inference: it remains the frozen cosine analysis. Raw-data export is intentionally separate.

## Completed figure reanalysis

The [paper figure report](../artifacts/paper-figures/README.md) now contains the separate raw-projection analysis, paired prompt estimates, control comparisons, and cosine sensitivity checks. It uses the same frozen per-token measurements as the export. Aggregation is explicit post-processing; the raw token records remain unchanged. This metric change is post hoc, and the original cosine results are preserved.

Reproduce the figures and independently verify the paired comparisons with a fresh output directory:

```bash
python scripts/build_paper_figures.py runs/a40-two-hour-reserves --output artifacts/new-paper-figures
python scripts/verify_paper_figures.py runs/a40-two-hour-reserves artifacts/new-paper-figures
```
