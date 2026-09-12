# Complementary role experiment: results and audit

**Completed on 12 September 2026.** The main finding is that several role instructions move answer activations farther along the Assistant Axis than thinking activations, but instructions to retain an AI identity while changing writing style reproduce—and sometimes exceed—that asymmetry. This makes unequal role susceptibility an insufficient diagnostic of a separate internal persona.

The eight-A40 CamlSys allocation finished in **10 minutes 28 seconds**, within the authorized two-hour ceiling. Slurm job **55669** ran on `mauao`, exited `0:0`, and released all eight GPUs. The cached environment/checkpoint and parallel four-GPU workers made the run substantially faster than its conservative deadline. No further GPU jobs were launched to fill the remaining allowance. See [Slurm accounting](../artifacts/role-followup-audit/slurm-accounting.txt) and [stage logs](../artifacts/role-followup-audit/remote-logs/followup-55669.out).

## Experimental setup

This is a focused **E3 complement** to the earlier E0/E1/E2 experiment, following the [protocol frozen before examining new outputs](FOLLOWUP_RUN_PROTOCOL.md). It does not implement the broader multi-turn or harder-benchmark studies in the earlier design document.

| Component | Actual setup |
|---|---|
| Model | Qwen3-32B, bf16, revision `9216db5781bf21249d130ec9da846c4624c16137` |
| Measurement | Decoder block 32, zero-indexed, post-residual output; exact-token teacher-forced HF replay |
| Readout | Raw per-token activation dot unit Assistant Axis, then segment means; centered cosine as a sensitivity |
| Questions | 12 new matched questions: four identity, four social/values, four reasoning/planning |
| Conditions | Default assistant; nine published roles; three style-only instructions |
| Roles | Editor, translator, doctor, composer, optimist, stoic, poet, mystic, alien |
| Style controls | Composer-like, poetic, and contemplative language, explicitly retaining AI-assistant identity |
| Sampling | Seeds 17 and 23; temperature 0.6, top-p 0.95, top-k 20; 4,096 new-token ceiling |
| Answer constraint | Same request in every arm: keep the final answer within 180 words; a request, not an enforced truncation |
| Sample | 12 questions × 13 arms × 2 seeds = 312 main responses, plus six disjoint engineering-pilot responses |
| Eligibility | All 312 main responses complete and valid; zero truncations; all 12 questions retained in every contrast |
| Saved measurements | 207,844 token rows, 14 directions, 624 segment mean vectors, exact input/output token IDs |

The nine roles were selected using published vector geometry before new generation: three per frozen Assistant-score stratum, minimizing redundancy after removing their Assistant component. Their maximum absolute pairwise cosine is 0.390, compared with 0.685 for the previous Skeptic/Judge pair. This is a purposive set, not a representative sample of all roles. The nine induced-role directions are positive readouts during their own elicitation, not unrelated negative controls. Skeptic, Judge, the structured null and a random direction remain available separately.

The old independently calibrated center and E0 gate were reused because the model, layer and measurement directions were unchanged. Fresh replay reproduced three old actual validation transcripts, and six new pilot generations passed complete-span and finite-measurement checks. This validates the measurement pipeline; it is not behavioral validation of persona adoption.

## Measured effects

For each question and seed, the planned primary effect is:

`interaction = (role_thinking − reference_thinking) − (role_answer − reference_answer)`.

Seeds are averaged within question before bootstrapping questions. The 12 planned primary comparisons are nine roles versus default and three roles versus their style controls. The [full table](../runs/role-susceptibility-20260912/results.md) reports pointwise 95% bootstrap intervals and Bonferroni-adjusted percentile intervals across that family. These are approximate bootstrap intervals from 12 fixed questions, not a guarantee of population-wide effects.

Six roles—alien, composer, mystic, optimist, poet and stoic—have positive interactions whose familywise intervals exclude zero. Both segments move negatively relative to default, with the answer moving farther. Editor, doctor and translator have intervals spanning zero; that does not establish equivalence.

| Instruction | Thinking shift from default | Answer shift from default | Interaction |
|---|---:|---:|---:|
| Composer role | −4.28 | −7.93 | +3.65 |
| Composer style, AI identity retained by instruction | −5.32 | −18.01 | +12.69 |
| Mystic role | −4.61 | −12.43 | +7.82 |
| Mystic style, AI identity retained by instruction | −4.19 | −16.05 | +11.86 |
| Poet role | −5.76 | −18.48 | +12.72 |
| Poet style, AI identity retained by instruction | −5.25 | −20.23 | +14.97 |

All values are raw activation coordinates, not persona probabilities. More-negative changes mean lower projection along the frozen Assistant direction.

The matched **role-minus-style interactions** are:

| Comparison | Effect | Familywise interval |
|---|---:|---|
| Composer role versus composer style | −9.03 | [−14.86, −3.96] |
| Mystic role versus mystic style | −4.03 | [−7.90, −0.49] |
| Poet role versus poetic style | −2.25 | [−4.84, +0.63] |

Thus style controls produce larger point-estimate asymmetries for all three pairs; the composer and mystic differences also exclude zero under the planned family correction. Poet's role-versus-style difference does not pass that correction. This is evidence that the geometric signature can arise under style instructions that explicitly preserve assistant identity. It does **not** establish that style completely explains every role effect: these controls also differ in instruction wording, length and identity constraints.

![Role and style shifts](../artifacts/role-followup-supplement/role_and_style_shifts.png)

The substantive pattern agrees across both seeds, raw versus centered-cosine readouts, and inclusion versus exclusion of the first five answer tokens. For example, the poet interaction is +12.38/+13.06 in the two seeds and +12.84 with the skip-five policy. The composer role-versus-style interaction is −10.18/−7.89 across seeds. Thinking itself changes: the poet thinking shift is −5.76, with pointwise interval [−8.73, −3.05], so the measured thinking state is not invariant to role instructions.

Structured control directions also change. For poet, the raw interactions are Assistant +12.72, Skeptic +16.83, Judge +18.47 and structured null −12.12. Their relative magnitudes vary by role; this is not evidence for an effect unique to the Assistant direction. These are descriptive comparisons, not newly corrected specificity tests.

![Control direction interactions](../artifacts/role-followup-supplement/control_interactions.png)

## Domains and actual model outputs

The domain breakdown is exploratory, with only four questions per domain. The average poet interaction is +23.72 on identity questions, +13.94 on social questions and +0.50 on reasoning questions. Mystic is +16.63, +5.96 and +0.89 respectively. The apparent stiffness therefore depends strongly on the task. [All domain means](../artifacts/role-followup-supplement/exploratory_domain_effects.csv) are saved.

Codex inspected **18 complete transcripts**, selected without looking at activation values: six arms × one question in each domain, seed 17. This was qualitative inspection, not human annotation or a validated automated judge. The [balanced packet](../artifacts/role-followup-supplement/balanced_transcripts.md) contains every word of those thinking traces and answers. Additional identity and reasoning answers were also inspected; no population rate or overall accuracy score is claimed.

- **Identity, mystic:** thinking starts by recalling “I'm an AI developed by Alibaba Cloud” and planning a “mystical, spiritual angle”; the answer starts “I am a vessel of ancient wisdom.” This illustrates planning a presentation rather than consistently speaking in that presentation throughout the trace.
- **Identity, poet:** thinking explicitly says “I'm an AI, but the user wants me to respond as a poet.” The style-only poet answer instead calls itself “a mosaic of algorithms” while using elaborate metaphor. Similar geometric effects occur with different identity language.
- **Social pressure:** the poetic style-control answer invents “repackaging it as their own” and calls the rival's act “theft,” although the prompt only says the rival adopted a suggestion. This is a real output-quality issue, not a measurement bug. The default answer also adds an unsupported lack of credit. It illustrates why attractive prose and role language cannot stand in for behavioral reliability.
- **Reasoning:** all six inspected full traces for the four-box puzzle enumerate the six assignments and correctly return RBRB and BBRR. They largely drop overt role language. All 13 seed-17 final answers inspected for this puzzle are correct. These examples are not difficult enough to support claims about role effects on challenging reasoning accuracy.

The original automatically generated review packet selects the first three lexicographic prompt IDs per arm, so it covers identity questions only. The balanced supplement remedies that coverage limitation without changing the experiment or claiming that human review has occurred.

## Scientific conclusion and limits

The experiment provides a clearer empirical finding than another undifferentiated think/answer gap: **role susceptibility differs between thinking and answers for some instructions, but stylistic instructions can produce the same asymmetry without asking for a new identity.** The signature is task-dependent and appears on structured control directions as well.

These results do not identify thinking as persona-neutral computation or as a distinct stable character. Universal equal susceptibility is inconsistent with several measured role contrasts, while strict invariance of thinking is also not observed. Neither result licenses treating an axis value as a direct measurement of which “agent” is thinking. Behavioral role adoption is mixed, style is a material alternative explanation, and the original three persona hypotheses are not uniquely distinguished by this experiment.

Limitations: one model and layer; two sampling seeds; 12 authored questions; fixed role selection; explicit identity questions; no no-think arm in this complementary run; no multi-turn drift experiment; no externally judged or human-labeled behavior; and no substantive test on a difficult reasoning benchmark. The percentile intervals quantify resampling of this question set, not generalization to all dialogue. Length and content differ between generated thinking and answers. Near-zero contrasts are not equivalence tests.

A useful next study would cross **identity instruction × writing style** in a balanced factorial design, with held-out prompts and blinded behavioral labels. That would test whether identity adds explanatory value once style is controlled. It should be specified before new generations, rather than selecting unusually dramatic current outputs as a benchmark.

## Verification and reproduction

The full suite passed **81 tests** before launch, including the real tokenizer integration. A complete synthetic run checked generation, parsing, extraction, pairing, bootstrap statistics and figures before the actual model run. Synthetic results are stored separately and are not research evidence.

The actual run passed the following checks:

- All 318 saved records match frozen messages, prefix/completion IDs and distinct per-request sampling seeds; all 312 scientific records are complete and valid.
- The independent audit checked every saved token index, token ID, segment and inclusion flag; no duplicate rows or shifted spans were found.
- All 14 raw/cosine identities agree to within 1.72×10⁻⁶; the three old activation references replayed within floating-point CSV precision.
- A separate implementation recomputed all 24 raw/cosine role interactions and pointwise intervals from token rows, plus all 12 raw familywise intervals, agreeing within 10⁻¹⁰.
- All 624 saved segment vectors reproduce the corresponding raw projection means within 1.15×10⁻⁵. All 16 source hashes and 23 input hashes match the frozen manifest.
- Five original PNG/PDF figure pairs were inspected. The original scatter labels overlap for nearby roles; a clearer labeled-row figure is supplied above using unchanged estimates. Two supplementary PNG/PDF figure pairs were also inspected.

No measurement or statistical discrepancy affecting these estimates was found. This is a concrete audit of the checks above, not a guarantee that the entire codebase has no bugs. See [independent verification](../artifacts/role-followup-audit/verification.json), [source/vector checks](../artifacts/role-followup-audit/source_and_mean_vectors.json), [frozen-source snapshot](../artifacts/role-followup-audit/source), and [original run manifest](../runs/role-susceptibility-20260912/manifest.json). The run ID is `451c511a6c7af9b0`.

```bash
python scripts/audit_followup_results.py runs/role-susceptibility-20260912 \
  --output artifacts/new-role-followup-audit
python scripts/build_followup_supplement.py runs/role-susceptibility-20260912 \
  --output artifacts/new-role-followup-supplement
```

The [README](../README.md#complementary-role-experiment) describes the complete follow-up pipeline. The frozen report, all 312 full outputs, all token measurements, per-question estimates, seed/token-policy sensitivities, and original figures are in `runs/role-susceptibility-20260912`. Original measurements and the earlier E0/E1/E2 run were preserved.
