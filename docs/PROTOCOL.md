# Analysis protocol and material assumptions

The proposal remains the source of the hypotheses. This document records the choices needed to make them testable and the limitations those choices introduce. Freeze them before reading evaluation projections.

**Measurement amendment, 12 September 2026:** this document describes the original centered-cosine protocol. Following the user's subsequent request, [MEASUREMENT.md](MEASUREMENT.md) defines raw per-token linear projection as the preferred readout for future analysis, with aggregation deferred to post-processing. This change is post hoc for the existing run; its original cosine results remain unchanged.

## Hypotheses and evidence

| Test | H1: shared Assistant persona | H2: neutral computation | H3: distinct stable character |
|---|---|---|---|
| E1 | Similar thinking/answer axis levels | Lower thinking, boundary change | Stable non-default thinking level |
| E2 | Primarily content-linked | Think-context-linked | Think-context-linked |
| E3 | Similar role-induced shifts | Little thinking shift | Compressed nonzero thinking shift |
| E4 | Overlapping clouds | Collapsed thinking role cloud | Similar shape with translation |

These are operational predictions, not exhaustive generative models. They do not identify an agent, consciousness, or literal internal identity. Failure to detect a difference does not establish equivalence. Near-zero cosine is defined relative to the chosen center and cannot by itself establish neutrality. The program deliberately reports measurements without automatically assigning a hypothesis.

Persona specificity requires a larger Assistant Axis effect than meaningful structured controls. `specificity.csv` bootstraps the within-prompt quantity `abs(assistant gap) - abs(control gap)`; this estimates typical per-prompt gap magnitude, not the absolute population-mean gap. Read it alongside signed domain gaps and distributions. Role controls, the calibration null, and PCs 2–5 matter; high-dimensional random vectors are a weak sanity check. PC1 may align with the Assistant Axis and is not an unrelated control. All intervals are pointwise; there is no simultaneous family-wise error claim.

If E2 mainly follows content, report evidence against the proposal's tag-keyed state interpretation. Similar effects for scratch and atomic tool blocks weaken think-specific explanations. Natural thinking and answer differ in content, absolute position, and context; no single contrast removes every confound.

## Measurement and calibration

- The primary ruler is the released Qwen3-32B Assistant Axis, sign preserved, at decoder block **32, zero indexed, post residual addition and before final model normalization**. This comes from the [pinned upstream model config](https://github.com/safety-research/assistant-axis/blob/a98961956072224eaf244eb289d6c01700b63795/assistant_axis/models.py) and [pipeline convention](https://github.com/safety-research/assistant-axis/blob/a98961956072224eaf244eb289d6c01700b63795/pipeline/README.md).
- Scalar cosine is `(h - center) · unit_direction / ||h - center||`. Raw dot is `h · unit_direction`; residual norm is `||h||`. Centered norm is retained too. Cosine of a zero centered vector is undefined, represented by NaN rather than silently assigned zero.
- Center is the token-weighted mean of **24 independent default no-think calibration answers**, excluding the first five answer tokens. It is frozen once and shared across seeds, directions and conditions. Calibration defaults to a separately fixed seed 42. New default generations are used because the released default role vector is not necessarily the required evaluation population mean.
- Skeptic and Judge controls are released role means minus the released default mean, then orthogonalized against the Assistant Axis and normalized. The null is the normalized difference between two fixed random halves of calibration prompt mean vectors. It is not orthogonalized: overlap with the Assistant Axis is itself informative. Random directions use a fixed seed. All metadata and vectors are saved.
- E0 measures default and about ten prespecified roles on held-out questions, using only no-think answer tokens. Every generation seed must have a positive lower 95% paired prompt-bootstrap bound for default minus average role, and Spearman rank correlation at least 0.3 with the released roles' projected ordering. These numerical thresholds operationalize the proposal's qualitative gate; they are not tuned from E1–E4. Missing configured role names are allowed only if at least six remain. A failure stops the pipeline.
- No runtime activation vector is copied to disk per token. Hooks compute all scalar directions in chunks. The teacher-forced **base decoder** is called without the LM head, cache, or hidden-state returns, avoiding a full sequence × vocabulary logits tensor. Measured h at token position i is the post-block representation **after observing token i**, not the representation predicting it.

## Exact spans and E2

Parsing begins only in the assistant content region; think-like strings in system/user prompts cannot define a boundary. Prefix and completion IDs are stored separately and concatenated directly. Opening delimiters, EOS, and separable leading/trailing whitespace are outside content means. The closing `</think>` token is retained in its own boundary row. An indivisible BPE token containing both text and whitespace remains intact. Missing, repeated, reversed, or unterminated think blocks are invalid, and token-limit completions are excluded even if a closing tag appeared.

Qwen3-32B's actual tokenizer marks `<think>`, `</think>`, `<tool_response>` and `</tool_response>` as dedicated **added tokens with `special=False`**. The proposal's wording “special tokens” is therefore imprecise. We preserve its `*_in_special` condition names but describe them as template-supported atomic-token controls. Tokenizer metadata is recorded; absent or non-atomic tool tags are reported as unsupported rather than disguised as valid controls.

E2 copies exactly the model's original thinking or answer **token IDs**, never a decode/re-encode round-trip. Each source is measured in four contexts: thinking, empty-think plain answer, ordinary scratch wrapper, and atomic tool-response wrapper. Two matched cells (`cot_as_cot`, `answer_as_answer`) supplement the proposal to permit a 2×2 tag/content factorial without using the natural long-CoT-prefixed answer as the plain-answer cell. Only the causal prefix and source span are needed. Following suffix tokens cannot affect the measured span; an actual tiny-Qwen test verifies this invariance.

All source tokens are included in every matched transplant cell, avoiding a first-five exclusion that would otherwise change the compared content across cells. Natural baselines retain the main answer-token rule. Prefix lengths still differ slightly and the empty-think answer prefix is unusual; absolute position is logged, and early/late thinking and boundary diagnostics are available. These are context manipulations, not clean interventions on an isolated persona variable.

## Cohorts and statistical units

Generation requests use frozen per-domain candidate ranks, one sample per prompt/condition/seed, and deterministic per-request seeds. Up to 100 candidates per domain are generated; the first 50 that have complete, valid thinking/no-think pairs **across all seeds** form the primary cohort. Fewer than 40 is a hard failure. The AI-philosophy probe set has only 50 candidates; shortages are possible. All candidate eligibility/reasons are saved, but projections are computed only for retained transcripts. The token table is not an attrition denominator.

The 200-thinking-token minimum and complete-seed requirement condition the studied population on generating sustained, completed thinking. Domain attrition may be substantial. The number of configured seeds can change this cohort; compare matched retained prompt sets when comparing seed studies. Step-by-step failures do not remove an otherwise eligible prompt from E1 or its exact-content transplants; the step-by-step paired contrast has its own available sample count.

Main natural answer means always exclude the first five content tokens, predeclared rather than decided from observed outliers. This may exclude very short math answers. Companion tables include the closing boundary, first five, and all-answer means. The all-answer-token ablation removes both this exclusion and its minimum-answer-length consequence. It should be labeled as a sensitivity analysis.

E1/E2 first average tokens within transcript segment, form differences within the same prompt and seed, then average seeds within prompt. Bootstrap resamples prompts. Paired standardized effect is `dz = mean(difference)/sample_sd(difference)` with bootstrap intervals; undefined/unstable tiny-sample values are not filled with zeros. Domain estimates remain separate. Per-seed tables assess stability. Pooled CIs are conditional on the observed seeds and model checkpoint.

E3 compares each role to default on identical extraction questions and seeds. Questions are averaged within each role before role-level summaries and bootstrap CIs. Surviving paired question counts can differ across roles and are reported in `role_retention.csv`; E3 role effects remain paired, but heterogeneous question composition limits cross-role comparisons. The descriptive OLS stiffness fit has a free intercept and role-bootstrap intervals. Both coordinates are noisy, so it is not an unbiased latent susceptibility estimate. Missing judges are explicitly reported; any answer or thinking refusal is flagged when labels are attached. The main geometry is retained with flags, not silently filtered after inspecting effects.

## E4 and interpretation limits

E4 only includes roles with the **same complete planned question set across all seeds**, including the default marker. It requires at least six non-default roles to define five answer PCs. Seeds are averaged within question, questions within role, and roles equally weighted. Default does not fit the PCA. Both clouds use the answer-role mean and answer PCA frame; a separate thinking PCA produces absolute PC alignment. PCA signs are deterministic. Degenerate thinking clouds are valid H2-like outcomes and have unavailable thinking PC axes rather than fabricated ones.

Full hidden-space cloud metrics report thinking/answer RMS role spread, centroid translation, and RMS residual role displacement after removing that translation. A translated identical cloud has spread ratio one and zero residual displacement; a collapsed thinking cloud has near-zero spread. These are descriptive summaries on sampled roles, not confirmatory population tests. PCA alignment is unstable when eigenvalues are nearly tied, so a non-diagonal alignment heatmap alone is not evidence of different semantic rulers.

There is an unavoidable dependency absent from the proposal's one-pass aspiration: PCs cannot be projected before they exist without storing token activations. E3 therefore gets a **mean-only precursor pass**, then a scalar pass once the common PCA is fitted. All other retained evaluation transcripts get one scalar pass. Only role/segment mean vectors are persisted. E4 PCs use the same role sample and are descriptive controls, while the independently released role controls and held-out null are available regardless of E4.

## Reproducibility and scope

Run identity covers resolved config, all input content, source files, and numerical package versions. Model/tokenizer and upstream assets are pinned; prepared dataset files have hashes and source fingerprints. Generation finish reasons and exact tokens are saved. Per-request seeds do not guarantee bitwise reproducibility across GPU kernels, vLLM versions, batching or hardware; the stored transcripts are the authoritative replay inputs. Reusing cached requests after interruptions is supported; changing the environment or protocol requires a new run directory.

The smoke backend is a hand-designed latent activation simulator with byte-level tokenization. Its purpose is plumbing, statistical-unit and plotting verification. Tiny random Qwen tests separately verify the real hook and tokenizer contract. Neither supplies substantive model results. Full pretrained generation and GPU performance remain to be established on the target hardware.

The implementation supplies blank human review forms and a fixed judge client; it does not invent the required manual reading, calibration labels, or agreement. The 50 AI-philosophy items are authored probes instead of an LLM-generated sample. Code benchmarks supply task prompts; generated code is not executed and task accuracy is not the primary dependent variable. Multi-turn drift, steering/capping, emergent-misalignment organisms, second-family validation, and faithfulness correlations remain the proposal's deferred work.
