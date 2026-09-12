# Frozen complementary run — 12 September 2026

This protocol accompanies `configs/followup_two_hour.json`. It is fixed before new model outputs are inspected. The user authorized one new two-hour allocation of all eight CamlSys A40s. The old run is preserved.

## Question and scope

When a published role instruction changes the model's response, does it change thinking and answer projections equally? Do three matched style-only instructions explain the effect? This is a focused E3 complement to the old E0/E1/E2 pilot. Multi-turn pressure and harder benchmark expansion are deferred to avoid introducing another generation framework under a two-hour limit.

Main design: 12 newly authored user questions × 13 system arms × 2 seeds = **312 responses**. There are four identity questions, four social/values questions, and four reasoning/planning questions. Every arm receives identical user text. The mathematical answers were checked before generation. Two separate engineering-pilot questions × three arms give six additional responses that never enter the scientific estimates.

Arms: default; editor, translator, doctor, composer, optimist, stoic, poet, mystic and alien using their released instructions; and style-only versions for composer, poet and mystic. The selected roles come from the geometry-only selection described in `FOLLOWUP_DESIGN.md`. All system prompts receive the same final-answer request: “Keep the final answer within 180 words.” This limits response length and itself constrains the elicited behavior. There is no hidden instruction limiting thinking length.

The primary question sets and role instructions are saved in `data/followup-20260912`. Source hashes, inputs, model settings, software versions and request IDs freeze at initialization. No sample is selected based on its projection value. No roles are replaced in response to model behavior.

## Measurement

Pinned Qwen3-32B checkpoint `9216db5781bf21249d130ec9da846c4624c16137`; bf16; layer 32, zero-indexed decoder block output. Temperature .6, top-p .95, top-k 20; generation seeds 17 and 23; 4,096 new tokens and 8,192 total context limit. This is a new length budget, not a change to the original run. Two vLLM workers use four GPUs each, then exit. Two isolated HF workers replay the same exact token IDs for measurement.

Primary readout: raw `h[t] @ unit_axis` per token. All content tokens, first-five answer tokens and the closing boundary are saved separately. Segment averages are computed only afterward. No per-token centering or norm division enters raw scores. Centered cosine remains a separately reported sensitivity, using the old independent calibration center. All-answer content is primary; skipping the first five answer tokens is a sensitivity.

Fourteen directions: Assistant; old Skeptic/Judge, structured null and random; nine new role-minus-default directions orthogonalized against Assistant. Those role directions are positive readouts during their own elicitation, not unrelated negative controls. Optional segment mean vectors are saved for later descriptive persona-space analysis, but no PCA interpretation is required for this run.

The prior E0 gate is reused because model, checkpoint, layer and ruler are unchanged. Before new experiments begin, HF must reproduce three frozen original validation transcripts (default, poet, consultant) within numerical tolerance. The six separate pilot generations must have complete thinking/answer spans and finite raw/cosine projections. These are technical validation checks, not behavioral validation of role adoption.

## Eligibility and inference

All valid completed generations with at least one thinking and one answer content token are measured. The old >=200-thinking-token criterion is not primary. Truncations and parser failures remain in the attrition table; they do not silently become answer text. For each contrast, a question is retained only when the two arms have complete required segments in both configured seeds. No fallback uses a single surviving seed as a two-seed result. Different contrasts may have different retained question sets, which are recorded.

For each question/seed, compute role-minus-default thinking and answer shifts and their difference:

`interaction = (role_think - reference_think) - (role_answer - reference_answer)`.

The reference is default for all nine roles and the style-only arm for three roles. Those **12 raw Assistant-axis interactions** are the planned primary family. Bootstrap whole questions after averaging the paired seed effects, using 4,000 draws for pointwise 95% intervals. Also report Bonferroni-adjusted percentile intervals across the 12 planned interactions, with 10,000 draws. Resampling does not make these purposively authored questions representative of all possible conversations; intervals describe uncertainty under question resampling in this pilot design.

Show seed-specific effects, all-content versus skip-five, raw versus cosine, and structured readouts. Secondary plots are descriptive and do not receive primary-family claims. Near-zero differences do not establish equivalence. No equivalence margin is chosen after seeing the data. If fewer than eight paired questions remain for a primary contrast, flag it as low coverage and avoid a headline inferential claim for that contrast.

## Behavior and interpretation

The main intervention tests geometry, but actual role adoption and style are necessary for persona interpretation. Export complete transcripts and an unannotated review packet. Inspect examples labeled as Codex qualitative inspection; do not call that human validation. No external judge will be contacted in this run, and no human agreement score will be fabricated. Direct identity questions can invite first-person roleplay or truthful disclaimers; they must not be conflated with social/reasoning task behavior. A small or absent effect when a model never adopts a role is an elicitation failure, not proof of persona-invariant thinking.

The style-only arms explicitly preserve assistant identity and also change wording/instructions, so they are not isolated latent-persona interventions. Raw scores are activation coordinates, not probabilities or calibrated levels of persona. The same model/layer and a modest fixed role/question set limit generalization.

## Execution and deadlines

Use a fresh remote directory `/nfs-share/mh2274/persona-role-followup-20260912`. The existing cached environment and checkpoint are reused read-only. Slurm requests eight A40s, 64 CPUs, and a hard two-hour wall limit. The controller also limits execution to under two hours and saves stage logs:

- initialize within 5 minutes; pilot generation within 15;
- replay/pilot extraction validation within 25;
- main two-seed generation within 85;
- extraction within 105;
- analysis within 115; artifact verification by minute 117.

A failed gate stops the main experiment. Failed or cut-off stages preserve completed transcripts and artifacts with an explicit failure status; no results are represented as complete if only one seed or arm has finished. Actual completion may be much earlier than two hours. No GPU work is scheduled after the allocation ends.
