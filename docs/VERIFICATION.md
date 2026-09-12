# Executed verification

Verification was performed locally on 11 September 2026 with Python 3.14.4, NumPy 2.4.4, pandas 3.0.2, Matplotlib 3.10.8, PyTorch 2.11.0, and Transformers 5.5.0. The host exposed neither CUDA nor MPS. The package installed successfully into a local virtual environment using an editable install. Exact offline plotting/test versions are in `requirements-smoke.lock`; each run has its own runtime manifest.

## End-to-end pipeline

```bash
persona run --config configs/smoke.yaml
python scripts/verify_run.py runs/smoke
```

The two-seed synthetic run passed E0, then completed generation, eligibility selection, E3 mean extraction, E4 fitting, scalar extraction, review export, analysis, and rendering. It produced:

- 74 calibration/E0 transcripts and 168 experimental generation records.
- A retained set of four prompts in each of four domains, nine non-default roles and three shared role questions.
- 196 measured natural/transplanted transcripts and **73,674 token rows**.
- Ten measured directions: Assistant Axis, two orthogonalized role controls, structured null, random, and five answer PCs.
- Nine PNG/PDF figure pairs, plus paired/standardized effects, factorial contrasts, role shifts, boundary/first-five tables, attrition and PCA artifacts.

The independent artifact verifier checked unique record/token identities, finite measured cosine values in range, normalized/absolute positions, negative thinking and positive answer boundary offsets, boundary exclusion, required conditions, figure outputs, and **four prompt observations—not tokens or doubled seed counts—in each pooled domain interval**. Figures were inspected for readable axes, control overlays and synthetic watermarks.

A second run with `pca=false`, `experiments=[E1,E2,E3]`, and `exclude_first_answer_tokens=false` passed the same pipeline and verifier, producing five directions and seven figure pairs. This exercises the E4-omitted and all-answer-token sensitivity paths.

The optional judge attachment was exercised with all 1,050 canonical sentences labeled by the explicitly unvalidated offline heuristic. Analysis recognized the heuristic provenance and left human agreement/calibration unavailable. No API was called, no human labels were filled, and no review was marked completed. The main smoke report may display this heuristic color baseline; it is a software test only.

## Model and token integration

Tests use a small randomly initialized `Qwen3ForCausalLM` on CPU. They compare hook projections against directly captured post-block activations, ensure the LM head never runs, and verify that discarding suffix tokens after a measured transplant leaves its activations unchanged.

The actual Qwen3-32B tokenizer was downloaded at revision `9216db5781bf21249d130ec9da846c4624c16137`, without pretrained model weights. Tests verify its thinking toggle, empty no-think block, atomic delimiter IDs and `special=False` metadata, exact source-ID copying, and user-message tag isolation. A compatibility failure discovered during this check—Transformers 5 returning `BatchEncoding` from `apply_chat_template`—was corrected and covered by the test.

To reproduce this optional check:

```bash
python -m pip install -e '.[models,test]'
python scripts/download_test_tokenizer.py
PERSONA_QWEN_TOKENIZER_DIR=data/assets/test-tokenizer python -m pytest -q
```

The released Assistant Axis, default, Skeptic and Judge tensors were also inspected directly using `torch.load(..., weights_only=True)`. Each was a plain `[64, 5120]` tensor, matching the importer and fixed layer convention.

## Regression coverage

The final suite completed with **58 passed in 5.98 seconds**, including the optional real-tokenizer integration. Tests cover causal spans and malformed/truncated outputs; known centered geometry and orthogonalization; translated, compressed and collapsed role clouds; paired bootstrap recovery and seed averaging; E2 factorial recovery; default-matched E3 shifts; noisy/degenerate slope handling; refusal flags from answer-only labels; missing judge/PCA handling; immutable input/config caches and E0 gates; leakage/stratification; annotation identity/agreement; figure generation; aggregation provenance/cohort guards; and preservation of human notes.

The final smoke command was then run again. It generated zero new transcripts, reused all 242 cached generation records and all 196 projection shards, reproduced identical `e1_effects.csv` bytes, and preserved the human notes and blank annotation template. These checks are recorded in `runs/smoke/verification.json`. Aggregate regression tests also verify duplicate rejection, runtime/cohort mismatch rejection, retained E0/source provenance, complete judge coverage, and reanalysis from saved arguments.

## What was not run

No pretrained 8B/32B generation, vLLM CUDA execution, paid judge, GPU rental, or full research corpus collection was performed. Real-model E0 has not been established. Synthetic E0 success, designed gaps, and random-model hook checks are not evidence for H1, H2, or H3. The repository provides the executable study; substantive experiments and real human review still require the stated data and hardware.

## Subsequent CamlSys experiment and audit

The sections above document the initial local verification, before GPU execution. Their statements about no pretrained run and the 58-test count are historical, superseded by the completed CamlSys E0/E1/E2 study and the **71-test** audit on 11 September 2026. See [results and audit](RESULTS_AND_AUDIT.md) for the actual 354,533-token run, independent GPU replay, corrected reporting bugs, and remaining scientific limitations. The updated end-to-end synthetic run is `runs/audit-smoke`; it again passed artifact verification with 73,674 tokens, ten directions and nine figure pairs. Real E3/E4 and human behavioral validation remain unrun.
