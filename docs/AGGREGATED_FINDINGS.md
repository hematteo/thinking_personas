# Aggregated findings and research decision

The supported conclusion is that **thinking and answer activations have different susceptibility to some role instructions, but that asymmetry is also produced by writing-style instructions that preserve AI identity**. It is insufficient evidence to identify thinking as a separate persona or as persona-neutral computation.

## New aggregation of the complementary study

These are **post hoc descriptive aggregates** of the completed run `451c511a6c7af9b0`. They average fixed roles and the two seeds equally within each question, then average the 12 questions equally. The earlier E0/E1/E2 run is not pooled into these numbers: its questions, conditions and estimands differ.

All 312 requested responses were completed and measured. Negative shifts indicate lower raw projection along the frozen unit Assistant Axis relative to the default arm. The interaction is the thinking shift minus the answer shift.

| Fixed set | Thinking shift | Answer shift | Interaction | Pointwise 95% CI for interaction |
|---|---:|---:|---:|---|
| All nine selected roles | −3.24 | −7.17 | +3.93 | [+1.93, +6.19] |
| Composer, mystic, poet roles | −4.88 | −12.95 | +8.07 | [+4.22, +12.17] |
| Their three style controls | −4.92 | −18.09 | +13.17 | [+8.48, +18.03] |

Intervals use 10,000 question-bootstrap resamples. They are exploratory, unadjusted, conditional on these fixed roles and seeds, and based on only 12 questions. They do not make the purposive role set representative or turn 312 responses into 312 independent question observations.

The directly paired aggregate of the three role-minus-style comparisons is:

| Contrast | Mean | Pointwise 95% CI |
|---|---:|---|
| Role − style in thinking | +0.04 | [−0.74, +0.75] |
| Role − style in answers | +5.15 | [+3.35, +7.17] |
| Role − style interaction | −5.11 | [−6.88, −3.53] |

The thinking difference is unresolved, not established as equivalent. The larger style-control interaction is chiefly associated with a larger answer shift. Identity instructions, style strength, wording and prompt length are not independently controlled here, so this comparison does not isolate style as the sole cause.

For the original prespecified individual comparisons, six roles have positive interaction intervals excluding zero after the 12-comparison family adjustment: alien, composer, mystic, optimist, poet and stoic. Doctor, editor and translator do not. The role-versus-style differences survive that correction for composer and mystic, but not poet. See [all planned comparisons](../artifacts/role-followup-supplement/primary_interactions.csv).

Across the nine fixed roles, the descriptive interaction is **+8.55 on identity**, **+2.77 on social**, and **+0.47 on reasoning** questions. Each domain contains only four questions. The prompt mix therefore materially affects the grand mean. A weak reasoning effect could reflect limited role elicitation as well as stable task processing; it is not proof that reasoning is persona-neutral. The inspected reasoning example was solved correctly across all 13 seed-17 conditions, so it did not test difficult capability boundaries.

## What the experiments jointly establish

| Experiment | Aggregate observation | Supported interpretation |
|---|---|---|
| E0, validation | Published direction distinguishes default and role answers under the frozen validation protocol | The imported measurement is functioning on those examples |
| E1, natural thinking/answers | Raw thinking-minus-answer: advice +0.03, CI [−1.53, +1.59]; math −1.51, CI [−2.63, −0.37]; 40 prompts per domain | No uniform gap across these domains; structured controls prevent an Assistant-specific interpretation |
| E2, identical-content replay | Assistant think-prefix effect: +3.13 for thinking text and +1.51 for answer text, with intervals excluding zero | Prefix/context changes the measurement even with fixed content; it does not isolate a persona switch |
| E3, role/style follow-up | Role-induced shifts are often larger in answers; style controls reproduce or exceed the asymmetry | Unequal susceptibility is not a sufficient diagnostic of internal persona identity |

The original E1/E2 [raw-projection tables and captions](../artifacts/paper-figures/README.md) document their controls and uncertainty. Their raw analysis was a methodological amendment after the original cosine analysis; raw projection was fixed before the new follow-up.

## Inference about the proposal

- The simple prediction that every role moves thinking and answers equally is inconsistent with several measured comparisons. That does not refute the broader idea that the model is generating both as an assistant: different linguistic functions can respond differently.
- The strict prediction that role instructions never change thinking is also unsupported: thinking projections move, and inspected traces discuss role instructions. We have not calibrated a geometric zero corresponding to persona-neutral computation.
- Smaller thinking shifts can resemble the proposal's “distinct, stable character” prediction, but style-only instructions reproduce the signature. No distinct character has been identified or causally validated.
- These findings do not show that the Assistant Axis is meaningless. The [original paper](https://arxiv.org/html/2601.10387v1) already associates steering with stylistic as well as identity changes. It also reports disabling Qwen's thinking mode. Our result concerns the limitations of extending that measurement to infer identity from differences between thinking and answer tokens.

No result establishes unfaithful thinking, a hidden agent, consciousness, or increased danger. The output example that invents theft is a useful behavioral failure to investigate, but it is not evidence that axis shifts cause that failure.

## Is another experiment needed?

For a scoped report about **limitations of the proposed geometric diagnostic**, the current experiments support a defensible result. The highest-priority remaining work is systematic behavioral review of existing outputs and careful writing. More random prompts, more roles, or more sampling seeds alone would not resolve the main ambiguity.

For a stronger claim about **whether identity changes affect thinking differently from answers independently of style**, one focused follow-up is worthwhile:

1. Cross identity instruction (AI assistant / named role) with style instruction (plain / role-associated style), using the same wording template and final-answer constraint in all four cells.
2. Use composer, poet and mystic as explicitly selected follow-up cases, new held-out questions, and two seeds. With 24 questions and a shared AI/plain baseline, there are 10 unique arms and **480 responses**. This is a workload design, not a power calculation or a guaranteed runtime.
3. Predeclare identity-by-segment effects within fixed style, style-by-segment effects within fixed identity, and their interaction. Preserve all outcomes and report failures of role/style elicitation rather than selecting successful examples after seeing geometry.
4. Add a fixed behavioral rubric for role identification, discussion of the role from an outside perspective, stylistic expression, unsupported claims and task correctness. Have a human review a balanced sample with geometry and condition labels hidden; any model judge should be calibrated against those annotations.

If identity-specific effects remain after style is held fixed and align with behavioral labels on held-out questions, the persona interpretation gains support. If style drives the segment asymmetry and identity contributes little with adequately narrow intervals, the methodological conclusion strengthens. This still would not by itself identify a separate internal agent. A causal activation intervention would be a later, stronger test.

Harder math alone is lower priority for this question: it increases task difficulty without separating identity from style. Include reasoning tasks with checkable answers, but prioritize hypotheses that the controls can distinguish. PCA of existing segment vectors can be supplementary; it does not settle this confound. No new GPU run was launched for this aggregation or recommendation.

## Reproduce

```bash
python scripts/aggregate_followup.py runs/role-susceptibility-20260912 \
  --output artifacts/new-role-followup-aggregation
```

[Aggregate estimates](../artifacts/role-followup-aggregation/fixed_role_aggregates.csv), [domain means](../artifacts/role-followup-aggregation/fixed_role_domain_means.csv), [question-level observations](../artifacts/role-followup-aggregation/question_aggregates.csv), and [provenance](../artifacts/role-followup-aggregation/provenance.json) are saved separately from the frozen experiment.
