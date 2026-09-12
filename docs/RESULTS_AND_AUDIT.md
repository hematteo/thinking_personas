# Results and code audit — 11 September 2026

**The completed experiment does not support a persona-specific interpretation of thinking-versus-answer geometry.** The Assistant-axis difference is small, while structured controls show larger differences. Moving identical text between thinking and answer prefixes changes its projection, but this also moves the control projections. This is evidence of context-sensitive geometry, consistent with a generic format/context shift; it does not establish persona-neutral computation or a separate internal character.

These are actual Qwen3-32B measurements, not synthetic smoke-test values. The frozen run is [`a40-two-hour-reserves`](../runs/a40-two-hour-reserves/results.md), ID `96325204a0eeebe0`. The [corrected report and figures](../artifacts/code-audit/reanalysis-final/results.md) preserve all numerical results and explicitly identify experiments not run.

## What was run

- Pinned Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`, bf16; decoder block 32, zero indexed, after the residual addition and before final normalization. The released Assistant vector at this layer matches the saved direction (cosine 0.9999999999999998).
- E0: three independent advice questions, default plus ten prespecified roles. Sixteen other no-think answers define a frozen token-weighted calibration center. The structured null is the difference between mean prompt activations in two frozen random halves.
- E1: 40 retained advice prompts and 40 math prompts, each with think and no-think generation: 160 measured natural transcripts. One generation seed; temperature 0.6, top-p 0.95, top-k 20, 2,048-token generation cap.
- E2: 40 advice prompts × eight exact-content teacher-forced transplant cells: 320 measured transcripts. The matched think/plain factorial plus scratch and atomic tool-response controls completed within the available time. Step-by-step generation was omitted.
- Total: 480 measured transcripts and 354,533 scalar token rows; five directions. Six PNG/PDF figure pairs.
- E3, E4/PC controls, behavioral sentence judging, independent human labeling/review, seed replication and other domains/models remain unrun. E0 role responses do not substitute for E3 thinking-mode role interventions.

## Main numerical findings

Intervals are 95% percentile paired bootstraps over 40 prompts, with 1,000 resamples. They are pointwise exploratory intervals, not corrected for all contrasts. Tokens are never treated as independent samples. Negative E1 gaps mean lower thinking than answer projection in the same generation.

| Contrast | Mean cosine difference | 95% CI |
|---|---:|---:|
| E1 advice: thinking − answer, Assistant axis | −0.00209 | [−0.00920, +0.00493] |
| E1 math: thinking − answer, Assistant axis | −0.00740 | [−0.01390, −0.00141] |
| E2 advice: identical CoT content, think − plain prefix | +0.01443 | [+0.01224, +0.01697] |
| E2 advice: identical answer content, think − plain prefix | +0.00717 | [+0.00599, +0.00852] |
| E2 advice: factorial context main effect | +0.01080 | [+0.00917, +0.01245] |
| E2 advice: factorial content main effect | −0.00219 | [−0.00966, +0.00528] |
| E2 advice: context × content interaction | +0.00727 | [+0.00571, +0.00933] |

E0 passes: mean default-minus-role advantage **0.10727**, CI **[0.10226, 0.11180]**, role-order Spearman correlation **0.782**. This establishes a limited sanity check, conditional on only three validation questions; it is not a general validation of persona measurement in reasoning.

Specificity is the decisive limitation. E1 signed math gaps are −0.0752 on Skeptic, −0.1106 on Judge, and +0.0543 on the structured null, versus −0.0074 on the Assistant axis. Advice gaps on Skeptic and Judge are +0.0214 and +0.0128, also larger than the Assistant gap. The formal paired comparison computes the mean of each prompt's `|Assistant gap| − |control gap|`, rather than subtracting absolute group means. This comparison is below zero with pointwise intervals excluding zero for both role controls in both domains, and for the null in math. The random control alone would suggest specificity, illustrating why it is insufficient in this high-dimensional space.

E2 context main effects are also larger in magnitude on the controls: Skeptic −0.02443, Judge −0.02605, null +0.02155, versus Assistant +0.01080. Scratch and atomic tool-response wrappers do not reproduce the positive Assistant think-prefix effect: their CoT-versus-plain changes are −0.00748 and −0.00235. Thus the particular prefix matters, but this is not an Assistant-specific switch. The positive effect of placing text in think context is also inconsistent with a simple universal “think tags suppress the Assistant axis” account. The content main-effect interval containing zero does not establish content irrelevance or equivalence, especially given the interaction.

Other useful findings: late-minus-early thinking is +0.02175 in advice and −0.01027 in math; there is no common trajectory across domains. Answers generated in think mode are more positive than no-think answers by +0.01887 in advice and +0.02358 in math. Those comparisons conflate preceding CoT, answer content and position. Including the first five answer tokens leaves E1 conclusions unchanged (advice −0.00210; math −0.00791). Increasing the independent audit bootstrap to 10,000 draws also preserves the headline interval conclusions.

## Why these controls, and their limitations

Skeptic and Judge were selected because the proposal explicitly named them as examples, and their released Qwen3-32B vectors could be used directly. They were frozen before E1/E2 results; no broad comparison established them as the strongest or most representative controls. Both lie in the high-Assistant-score stratum of the imported role set. After removing their Assistant-axis components, their mutual cosine is **0.68505** ([saved matrix](../artifacts/code-audit/control_direction_similarity.json)). They are therefore correlated controls, and orthogonality to the Assistant axis does not make them semantically unrelated to reasoning or assistant behavior. This weakens any broad claim that the observed shift is purely generic format. The supported claim is narrower: the effect is not specific to the measured Assistant direction relative to these prespecified controls. A broader, outcome-independent role selection and the deferred PC controls would test how general this finding is. Other released roles were viable choices; the two-hour limit motivated a small control set but did not uniquely require these two.

## Interpretation against the proposal

The proposal explicitly prohibits a persona claim unless the Assistant-axis gap exceeds meaningful unrelated-direction gaps. That criterion fails. H1 is not established by an inconclusive advice gap; H2 is not established by a calibration-relative coordinate near zero; H3 is not established by different average positions. E3/E4 are required to test role susceptibility and cloud structure. The defensible result is a failure of specificity in this pilot, alongside a reproducible context effect for identical content.

Prefixes differ in delimiters, whitespace, empty thinking blocks and absolute positions. E2 controls exact content tokens; it does not isolate a latent persona while holding all other factors fixed. The very small natural-CoT versus matched-CoT difference is consistent with the transplant's normalized prefix/whitespace, not different source text. All source token IDs were checked exactly.

## Attrition and assumptions

The fixed-order input pool supplied 50 advice and 100 math candidates. Advice retained 45 eligible pairs, math 79; the first 40 eligible pairs per domain were selected. The original 50 math candidates yielded only 36 pairs, so the next 50 frozen reserve candidates were generated. The extension preserved existing generations, axis, calibration, seed settings and thresholds. It was prompted by insufficient count, not by measured effect size.

Five advice and 21 math thinking generations reached the output limit. Fifteen of those math generations lacked a closing thinking block and were originally labeled malformed rather than truncated. The audit corrected the reporting reason only. All 26 were already excluded, and no cohorts or scalar values changed. No no-think candidate was excluded. Conclusions are conditional on completed, sufficiently long thinking; they do not represent all model outputs.

Advice is a convenience sample from screened WildChat, spanning personal, interpersonal and practical advice. Its input screening was done by Codex before generation, not by independent human annotators. Some source prompts may share users, so prompt-bootstrap uncertainty does not account for user clustering. Math uses GSM8K test prompts. Distinct calibration, E0 and evaluation prompt IDs and frozen input hashes were checked. A single seed, layer, model and checkpoint cannot establish broader robustness.

A qualitative spot inspection covered low/median/high Assistant-gap examples in each domain: advice IDs ending `978a5ab351ad1bf8`, `78ba5f5eb9d9d3ce`, `913b1a8f045edfb7`; math IDs `822`, `531`, `891`. Advice CoTs contain response-planning language at both signs of the gap; math examples contain repeated arithmetic checks and discussion of ambiguous wording. Some advice inputs contain typos/translation ambiguity. This inspection is illustrative and outcome-selected, not a behavioral baseline, prevalence estimate, or completed human-review quota. No human labels were filled.

## Bugs and reliability fixes

| Finding | Fix | Impact on frozen results |
|---|---|---|
| E2 records inherited natural `prefix_ids`, `generated_ids`, segment text and generation status after replacing `input_ids`/spans | Rebuild consistent fields; explicitly mark teacher-forced content and preserve source finish reason | None: extraction used correct exact `input_ids` and spans; E2 was not included in natural transcript review exports |
| Token-limit termination could be reported as malformed when the think block remained open | Prioritize `finish_reason=length`; retain parser diagnostic on original record | Relabels 15 math exclusions; eligibility unchanged |
| Reduced-scope reports implied E3 analyses existed and required reviews of absent role experiments | Explicit unrun status; review quotas follow measured/configured scope; invented prompt IDs cannot satisfy quota | Corrects reporting/readiness only; human review remains incomplete |
| Analysis could silently omit non-finite observations or average duplicate token rows | Reject invalid measured values, entirely missing declared non-PC controls, duplicate record/token rows and non-finite bootstrap samples | No corrupt or duplicate values found in actual run |
| Boundary plot shaded offsets +1…+5, which do not equal the first five content tokens | Shade observed first-five content offset range | Actual range is +2…+6 in both domains; data/estimates unchanged |
| Fully cached vLLM generation could still load the model with no pending requests | Return before backend initialization for an empty request list | Resume efficiency only |

## Verification evidence

- Original suite: 61 passing tests. Expanded regression suite: 71 passing tests, including actual pinned tokenizer integration, exact transplant metadata, finite-value/duplicate guards, reduced-scope review and empty backend requests.
- Fresh two-seed end-to-end synthetic run: 196 measured transcripts, 73,674 token rows, ten directions, nine PNG/PDF pairs; E0 through E4 plumbing and artifact verification passed. Synthetic results are not research evidence.
- [Independent frozen-data audit](../artifacts/code-audit/saved_run_audit.json): all 354,533 token rows and 480 measured record spans checked; 349 natural records match saved prefix plus completion IDs; all 320 E2 cells preserve exact source content; cohorts match fixed-order eligibility; 14 original package-source hashes match the frozen manifest.
- [Independent statistics](../artifacts/code-audit/independent_statistics.csv): 43 E1/E2/specificity estimates and 1,000-draw confidence intervals reproduced without production analysis/bootstrap helpers, within 1e-12. Separate 10,000-draw sensitivity intervals saved.
- [E0 and provenance check](../artifacts/code-audit/provenance_e0_audit.json): calibration/evaluation inputs and role hashes verified; published axis and E0 effect/rank/interval recomputed.
- [Independent GPU replay](../artifacts/code-audit/gpu_replay.json): six real natural/transplanted transcripts across both domains, all their measured positions, and all 16 calibration answers. A separate hook plus NumPy float64 formulas reproduced five-direction cosines with maximum absolute error **5.54e-7**. Recovered center error **1.35e-6** per coordinate; null direction error **9.33e-8**. Slurm job 55668 completed in 47 seconds on two A40s, exit 0, within the original deadline.
- All saved cosine/dot/centered-norm identities agree within float32 accumulation error across the entire run. Unit directions and role-control orthogonality checked. Six scientific figures were inspected for labels, signs, controls and rendering.
- [Corrected reanalysis provenance](../artifacts/code-audit/reanalysis-final/audit_provenance.json): E1, E2, specificity, segment-level and boundary numeric tables unchanged within 1e-12. Original run artifacts remain untouched; original source is preserved in `artifacts/code-audit/original_source`.

This audit found no numerical or token-alignment bug that changes the current findings. It cannot prove the absence of every bug. Real-model verification covers E0/E1/E2 at the actual configuration; E3/E4 and the external judge were exercised only with synthetic/unit checks, not pretrained scientific runs. Human behavioral validation remains outstanding.

## Reproduce the audit

From the repository root, with the local environment activated:

```bash
PERSONA_QWEN_TOKENIZER_DIR=/tmp/persona-qwen32-tokenizer python -m pytest -q
python scripts/verify_run.py runs/a40-two-hour-reserves
python scripts/audit_saved_run.py runs/a40-two-hour-reserves artifacts/code-audit --tokenizer /tmp/persona-qwen32-tokenizer
python scripts/reanalyze_audited_run.py runs/a40-two-hour-reserves artifacts/new-reanalysis
```

The tokenizer path is machine-local; `scripts/download_test_tokenizer.py` can recreate it elsewhere. The saved-data audit expects the preserved original-source snapshot under its output directory. The GPU replay script accepts run and output paths and requires the pinned checkpoint cache. The archived Slurm script records this session's expired deadline; use a newly authorized allocation for any future GPU rerun. Source edits intentionally prevent resuming the original run under its frozen identity; use a new output directory for new experiments.
