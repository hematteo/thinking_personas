# Do thinking and answer tokens occupy the same persona geometry?

**Completed Qwen3-32B pilot: E0/E1/E2.** [Full results and code audit](RESULTS_AND_AUDIT.md). [Corrected report and figures](../artifacts/code-audit/reanalysis-final/results.md).

## Research question

Do Qwen3-32B thinking tokens and answer tokens occupy different positions along the published Assistant Axis, and are those differences larger than changes on structured control directions? Does identical content move when placed under thinking versus answer prefixes?

## Hypotheses and observations

| Hypothesis | E1 prediction | E2 prediction | E3/E4 predictions, deferred | Observed |
|---|---|---|---|---|
| H1: shared Assistant state | Similar thinking and answer projections | Content-linked differences | Similar role shifts; overlapping role clouds | Advice gap inconclusive; small math gap. Content/context interaction prevents a simple content-only account. H1 not established. |
| H2: persona-neutral computation | Low/near-origin thinking, higher answers | Context-linked differences | Minimal role-induced thinking shift; collapsed cloud | Prefix matters, but think-prefix Assistant projection increases for identical content. Specificity fails; neutrality not established. |
| H3: distinct stable character | Stable non-default thinking level | Context-linked differences | Compressed role shifts; translated cloud | Domain trajectories differ. Prefix effects are not Assistant-specific. Character stability and role compression untested. |

These hypotheses are not exhaustive. A zero-centered coordinate is not proof of neutrality. The proposal's prerequisite for persona interpretation—Assistant gaps larger than structured controls—is not met.

## Methods

Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, bf16, published axis at zero-indexed post-residual decoder block 32. One generation seed, temperature 0.6, top-p 0.95, top-k 20, maximum 2,048 output tokens. vLLM generation and HF teacher-forced scalar extraction were separate stages on CamlSys A40s.

The frozen candidate pool supplied 50 advice and 100 math prompts. The original 50 math candidates yielded only 36 paired survivors, triggering a documented extension to the next 50 fixed reserve candidates with all thresholds unchanged. Completion, at least 200 thinking tokens, and a valid paired no-think answer determined eligibility. The first 40 eligible pairs in each domain were retained.

Sixteen disjoint default no-think calibration answers define a token-weighted center. Controls are released Skeptic/Judge contrasts orthogonalized against the Assistant axis, a random-half calibration contrast, and a fixed random unit vector. E0 used three separate advice questions with default and ten roles. Primary natural-answer means exclude the first five content tokens. Matched E2 cells measure all identical source-content tokens in think/plain/scratch/atomic tool-response contexts.

Statistics aggregate tokens within each prompt, then use paired prompt bootstrap intervals (1,000 resamples). Forty prompts, not tokens, are the sampling units in each domain. Intervals are pointwise and conditional on one seed. E3/E4, PC controls, step-by-step no-think generation, behavioral judging and human annotation were deferred.

## Results

E0 passed: default advantage 0.10727, 95% CI [0.10226, 0.11180], role-rank Spearman 0.782. Advice had 45/50 eligible pairs and math 79/100; 40 per domain were selected. Five advice and 21 math thinking generations hit the output limit. Fifteen of the math cases originally carried an incomplete-think-block label; the audit corrected the label to truncation without changing exclusion or results.

| Assistant-axis contrast | Mean | 95% CI |
|---|---:|---:|
| Advice thinking − answer | −0.00209 | [−0.00920, +0.00493] |
| Math thinking − answer | −0.00740 | [−0.01390, −0.00141] |
| Advice, same CoT content: think − plain prefix | +0.01443 | [+0.01224, +0.01697] |
| Advice, same answer content: think − plain prefix | +0.00717 | [+0.00599, +0.00852] |

The structured controls show larger differences. Math thinking-minus-answer gaps are −0.0752 on Skeptic, −0.1106 on Judge and +0.0543 on the null. Paired absolute-gap comparisons favor both role controls over the Assistant axis in both domains. The main specificity criterion therefore fails.

E2's average context effect is +0.01080 on the Assistant axis, while Skeptic, Judge and null effects are −0.02443, −0.02605 and +0.02155. The content main effect is inconclusive (−0.00219, CI [−0.00966, +0.00528]), with a positive content/context interaction (+0.00727, CI [+0.00571, +0.00933]). Scratch and atomic tool-response blocks do not reproduce the positive Assistant think-prefix shift. This shows dependence on the particular prefix, but does not isolate persona from formatting, position or delimiter effects.

Late-minus-early thinking increases in advice (+0.02175) and decreases in math (−0.01027). Including the first five answer tokens does not change the E1 interpretation. Independent 10,000-resample intervals preserve the headline conclusions.

The completed data comprise 160 natural and 320 transplanted measurements, 354,533 token rows, and six figure pairs. Source token alignment, calibration, E0, 43 statistical contrasts, and fresh GPU replay were independently checked. Reporting and metadata bugs were fixed; numeric results are unchanged. No human review or calibrated behavioral labels are claimed.

## Figures and interpretation

- [Trajectory](../artifacts/code-audit/reanalysis-final/figures/01_trajectory.png): direction and domain matter; no universal thinking-to-answer transition identifies a persona change.
- [Direction specificity](../artifacts/code-audit/reanalysis-final/figures/10_direction_specificity.png): structured controls shift more strongly than the Assistant axis; compare paired gaps, not overlapping level intervals.
- [Prompt gap distributions](../artifacts/code-audit/reanalysis-final/figures/05_gap_histogram.png): individual effects vary and overlap zero; the plotted histogram alone does not establish multimodality.
- [Exact-content transplants](../artifacts/code-audit/reanalysis-final/figures/02_transplants.png): prefixes move identical text across several directions, with content-dependent magnitude.

## Conclusion, limitations and falsifiers

This pilot yields a negative result for the specificity requirement of a persona interpretation. Its strongest positive finding is a reproducible prefix effect on identical content, shared by structured directions. It does not distinguish H1/H2/H3 or show that CoT monitoring reads a different agent.

The next decisive evidence would combine E3 role perturbations with calibrated behavioral labels and E4 geometry, retain structured controls, and replicate across seeds. A large Assistant-specific role/context effect with corresponding behavioral change would challenge the current generic-context explanation. The current inference remains limited by one layer/model/seed, convenience-sampled advice, possible shared users, completion/length filtering, absent behavioral validation and non-isolated prefix interventions.
