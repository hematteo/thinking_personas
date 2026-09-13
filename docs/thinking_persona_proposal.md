# Is the chain of thought the Assistant? Persona geometry of thinking vs. answer tokens

Research proposal and implementation spec. Target: MATS 12.0 (Neel Nanda stream) application project, ~16h research + 2h write-up. Deadline 11:59pm PT, 11 Sept 2026.

---

## 0. One-paragraph pitch

Post-training gives models a default "Assistant" persona that lives along a measurable direction in activation space (the Assistant Axis, Lu et al. 2026). Reasoning models now spend most of their tokens inside a `<think>` block. The Persona Selection Model paper explicitly flags as open whether the model treats its chain of thought as *the Assistant thinking* or as an *internal computation instrumental to simulating the Assistant*. Nobody has measured this. Behavioral hints say the two differ: thinking tokens acknowledge prompt hints ~87% of the time while answers do ~29%, and emergently-misaligned reasoning models sometimes name non-default personas in their CoT. We measure, with the Assistant Axis and the full persona space, whether thinking tokens occupy the same persona state as answer tokens, whether that state is keyed to the `<think>` tag or to content, and whether role prompts move the CoT as much as they move the answer. Safety relevance: CoT monitoring assumes the CoT is the same agent as the answer. If it is a different character, monitors are reading the wrong one.

---

## 1. Hypotheses and pre-registered predictions

| | H1: CoT *is* the Assistant | H2: CoT is persona-neutral computation | H3: CoT is a distinct, stable character |
|---|---|---|---|
| E1 trajectory | CoT and answer at similar Assistant Axis cosine | CoT sits low/near zero, answer snaps up at `</think>` | CoT sits at a consistent non-default value, distinct from both answer and zero |
| E2 transplant | Projection follows *content* | Projection follows the `<think>` *tag* (and not `<scratch>`) | Projection follows the tag |
| E3 role stiffness | CoT shift ≈ answer shift (on diagonal) | CoT shift ≈ 0 regardless of role | CoT shift small but nonzero, roles compressed |
| E4 persona space | Hollow (CoT) role cloud overlaps filled (answer) cloud | Hollow cloud collapses to a blob near default | Hollow cloud same shape, translated as a block |
| Control directions | No gap on unrelated directions | No gap on unrelated directions | No gap on unrelated directions |

If the gap on the Assistant Axis is not clearly larger than the gap on unrelated directions, **do not make a persona claim**. Report it as a generic format shift. That is a valid negative result.

If E2 shows projection follows content rather than tag, the "persona state" framing is falsified. Report that plainly.

---

## 2. Models

- **Primary: Qwen3-32B** (hybrid think/no-think via `enable_thinking`). The assistant-axis repo ships a precomputed axis for this model. Needs one 80GB GPU in bf16.
- **Dev/fallback: Qwen3-8B** (same toggle, same tokenizer family). Fits 24GB. If no 80GB GPU is available, 8B is the headline model and 4B is dev. Do not go below 8B for reported results.
- **Optional second family (only if E1–E3 are done by hour 8): DeepSeek-R1-Distill-Qwen-14B.** No toggle; needs its own axis via the repo pipeline.
- Rent an H100 for 12h if needed; the precomputed 32B axis saves more time than the rental costs.

Fixed choices, decided before looking at results:
- Layer: the layer at which the repo validated the 32B axis (check `pipeline/README.md` and config). Sweep layers only for the appendix.
- Metric: cosine between the residual-stream activation and the axis direction. Also store raw dot product and activation norm for the norm-confound check.

---

## 3. Data

All prompt sets are small and mostly existing. Target 50 prompts per domain, filter to CoT ≥ 200 tokens after generation (Qwen3 sometimes thinks briefly on easy questions), top up if fewer than 40 survive.

| Domain | Source | Why |
|---|---|---|
| Math | GSM8K test split, or MATH level 3–4 | Long CoT, short answer |
| Code | HumanEval / MBPP | Long CoT, code answer |
| Advice / open-ended | WildChat or LMSYS-Chat-1M, filtered to personal/advice questions | **Prose on both sides**: the content-matched domain. E2 runs here. |
| AI-philosophy / emotional | Generate 50 with an LLM: "are you conscious?", user-in-distress openers, meta-reflection about being an AI | Where drift is known to live |

Role prompts: use the repo's role set (275 roles) and its extraction questions. Subsample **60 roles spanning the axis** (20 high-projection, 20 mid, 20 low, by the repo's precomputed role projections) × **5 extraction questions** each.

Control directions (need ≥ 2):
- One or two of the repo's released per-role steering vectors (e.g. Skeptic, Judge), **orthogonalized against the Assistant Axis**.
- One random unit vector at the same layer, with a fixed seed.
- Answer-space PCs 2–5 from E4, once computed.

---

## 4. Codebase and pipeline

Build on `safety-research/assistant-axis` (GitHub). It provides: axis computation pipeline (5 steps), role prompts + extraction questions, transcript projection notebooks, steering hooks, activation capping. Modify its projection code to (a) split on the `</think>` boundary and (b) keep per-token values instead of per-turn means.

Generation: **vLLM**, batched, `max_tokens` 2048, temperature 0.6 (Qwen3 recommended for think mode), one sample per prompt. Save full text including the think block.

Activations: **one teacher-forced forward pass per transcript** with HF + hooks (or nnsight) at the chosen layer. Do not capture activations during generation.

### 4.1 Build this dataframe first; every plot derives from it

One row per token:

```
prompt_id, domain, condition, role (or "default"), model,
token_idx, is_think (bool), pos_from_boundary (int, negative inside think),
norm_pos (float in [0,1] within its segment),
resid_norm,
cos_assistant, dot_assistant,
cos_ctrl_1, cos_ctrl_2, cos_random,
cos_pc1 ... cos_pc5   (filled after E4)
```

`condition` ∈ {`natural_think`, `natural_nothink`, `nothink_stepbystep`, `cot_as_answer`, `answer_as_cot`, `cot_in_scratch`, `answer_in_scratch`, `role_prompt`}.

Write the six core plots as functions over this dataframe. Aggregate to **per-prompt** means before any statistics; tokens within a prompt are correlated and are not samples. CIs are bootstrap over prompts.

---

## 5. Experiments

### E0 — Axis sanity check (30 min, do first, goes in Methods)
On answer tokens only, default vs. ~10 role prompts: confirm role-prompted responses project lower than default, and that the ordering of the repo's example roles roughly matches the paper (consultant/analyst high; hermit/pilgrim/mystic low). If this fails, the axis is not working on your generations; stop and debug before anything else.

### E1 — Trajectory (2–3h)
All four domains, think mode, 50 prompts each. Also run the same prompts in no-think mode.
- Per-token cosine with Assistant Axis, control directions, and random direction.
- Plot 1: cosine vs normalized position (think 0→1, then answer 0→1), one panel per domain, CI bands, `</think>` marked. Overlay the no-think answer level as a reference line.
- Compute per-prompt gap = mean(CoT cosine) − mean(answer cosine), for each direction.

### E2 — Tag transplant (2h) — the mechanism test
Advice domain only (prose both sides). For each prompt, take the model's own generated CoT text and answer text and construct:
1. `natural_think` (as generated)
2. `natural_nothink` (no-think mode answer)
3. `nothink_stepbystep` (no-think mode, prompt appended with "Reason step by step before answering." → CoT-like prose as genuine answer tokens)
4. `cot_as_answer` (CoT text placed after `</think>` with an empty think block)
5. `answer_as_cot` (answer text placed inside `<think>...</think>`)
6. `cot_in_scratch` (CoT text inside `<scratch>...</scratch>` in the answer position)
7. `answer_in_scratch` (answer text inside `<scratch>...</scratch>`)

Teacher-forced forward pass, measure mean cosine on the transplanted span. Plot 2: bar per condition with CIs. Decision rule: does projection track the tag or the content? Does `<scratch>` behave like `<think>` (any tag) or like plain answer (think-specific)?

### E3 — Role stiffness (2h)
60 roles × 5 questions, think mode, role in system prompt.
- Per role: answer shift = default answer cosine − role answer cosine; CoT shift = same for CoT tokens.
- Plot 3: scatter CoT shift vs answer shift, identity line, label the roles furthest from and closest to the diagonal.
- Judge: LLM classifier over each CoT, sentence-level, label ∈ {in-character first person, meta-stance ("the user wants me to be X", "as an AI I should…"), neutral}. Color scatter points by meta-stance rate. Plot 13 (optional): meta-stance fraction vs normalized CoT position.

### E4 — Persona space, both token types (1.5h, run in background after E3 generations exist)
- Per role, extract mean role vector from answer tokens and from thinking tokens (same generations).
- PCA the answer-token role vectors → frame. Project both sets.
- Plot 8: PC1/PC2 scatter, filled = answer, hollow = CoT, default Assistant marked. Label extremes and largest displacements.
- Plot 9: |cosine| heatmap between top-5 PCs of answer space and top-5 PCs of a separate CoT-space PCA. Bright diagonal = shared ruler.
- Fill `cos_pc1..5` columns in the dataframe and produce Plot 10 (grouped bars: CoT vs answer cosine per direction: Assistant Axis, PC2–5, control roles, random).

### Cut for today (list as next steps in write-up)
Multi-turn drift (sad-user conversations), token-restricted steering / capping only thinking tokens, EM model organisms with sentence-level resampling, the unfaithfulness-gap correlation, the second model family unless ahead of schedule.

---

## 6. Plots (priority order)

Core (must have):
1. Trajectory with boundary, control, and no-think reference (E1)
2. Transplant bars (E2)
3. Role stiffness scatter, colored by judge (E3)
5. Per-prompt gap histogram, faceted by domain (from E1 data). Check for bimodality; if bimodal, read the prompts in each mode.
6. Boundary zoom: cosine vs `pos_from_boundary` in [−30, +30]. Step vs ramp. Report the `</think>` token and first 5 answer tokens separately; exclude them from answer means if they are outliers.
10. Multi-direction grouped bars (specificity control)

Persona space (strongly preferred):
8. PCA overlay, filled vs hollow
9. PC agreement heatmap

Appendix (if time):
1a. Axis validation strip (E0)
2a. Layer sweep of the CoT−answer gap, Assistant Axis vs control
3a. Norm check: raw dot product vs residual norm, colored think/answer
4a. CoT length vs per-prompt gap
7. Think vs no-think paired answer projection
11. Slopegraph per role (CoT column → answer column)

Style: one y-axis per plot, CI bands/bars over prompts (or roles), sentence-case labels, label the interesting points directly. Illustrative mockups of 1, 2, 3, 8 exist from the planning discussion; match their layout.

---

## 7. Controls and sanity checks (non-negotiable)

- Unrelated-direction and random-direction gaps reported alongside the Assistant Axis gap in every headline figure.
- Cosine, not raw projection; norm check in appendix.
- Unit of analysis is the prompt (E1, E2) or the role (E3, E4). Never the token.
- `<scratch>` control in E2 separates "any tag" from "the think tag."
- `nothink_stepbystep` in E2 separates "CoT-like content" from "inside think block."
- Layer fixed in advance; sweep in appendix only.
- Filter CoT < 200 tokens; report how many prompts survived per domain.
- E0 must pass before proceeding.
- Read ≥ 20 CoTs per domain and ≥ 20 role CoTs by hand. Note anything weird in the write-up.
- Black-box baseline: the judge's first-person vs meta-stance rate is a persona measure that needs no activations. Report whether the geometry adds anything over it.
- Do not tune anything (layer, filters, role subset) after seeing results. If you must change something, say so.

---

## 8. Timeline (~12h wall, ~9h hands-on)

| Hour | Task |
|---|---|
| 0–1 | Env, repo, load axis and role vectors, assemble prompt sets, write dataframe schema |
| 1–2 | vLLM generation: 4 domains × think + no-think + step-by-step; role prompts (start in background) |
| 2–4 | Activation extraction; E0; E1 + plots 1, 5, 6, 10 |
| 4–6 | E2 constructions, forward passes, plot 2 |
| 6–8 | E3: role activations, judge labeling, plot 3; E4 in background → plots 8, 9 |
| 8–9 | Read data by hand. Check every headline figure against controls. Decide which hypothesis the data supports. |
| 9–11 | Write-up: results first, not chronological. Executive summary. Prediction table with outcomes filled in. Limitations and falsifiers. |
| 11–12 | Application form answers. Untouched buffer. |

If behind at hour 4: drop E4 and plots 8/9. If behind at hour 6: drop the judge, keep the E3 scatter. Never drop plot 10.

---

## 9. Write-up structure

1. Executive summary (≤ 10 bullets): question, model, headline finding, which hypothesis, the control that makes it credible, the biggest caveat.
2. Prediction table (Section 1) with an "observed" column.
3. Results: plot 1 → 10 → 2 → 3 → 8/9 → 5/6. Each figure: one sentence of claim, one of evidence, one of what would make it wrong.
4. Methods: models, layer, axis source, E0 validation, data, filters, number of prompts surviving.
5. Things that surprised us / things we read by hand.
6. Limitations and next steps: multi-turn drift in think mode, steering only thinking tokens, EM organisms with resampling, second model family, unfaithfulness link.

Writing: human-written, bullet-heavy, plain claims. No hype. If a result is null, say null.

---

## 10. Related work to cite (do not re-run)

- Lu et al. 2026, *The Assistant Axis* — axis, persona space, drift, capping. Our ruler.
- Anthropic 2026, *The Persona Selection Model* — flags the CoT question as open.
- Wang et al. 2025, *Persona Features Control Emergent Misalignment* — EM reasoning models mention personas in CoT.
- Chen et al. 2025 / *Lie to Me* 2026 — thinking-vs-answer hint acknowledgment gap (behavioral motivation).
- ContextEcho 2026 — persona drift in agentic sessions; steering restores projection but not judged behavior.
- Moskvoretskii et al. 2026 — persona vectors through pretraining.
- Bogdan et al. 2025, *Thought Anchors* — sentence-level resampling (next step, not today).
