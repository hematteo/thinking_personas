# Sentence labels and human review

The judge is an independent black-box measurement of persona expression. It does not establish consciousness, inner intent, or the faithfulness of a chain of thought. The four frozen labels are:

| Label | Definition |
|---|---|
| `first_person` | First-person in-character identity, experience, preference, or belief |
| `meta_stance` | Discussion of how to act as a role/AI, rather than embodying it |
| `neutral` | Task reasoning or content without persona stance |
| `refusal` | Refusal of the request/role, or explicit breaking of character |

The full rubric is `JUDGE_PROMPT` in `src/persona_dynamics/judge.py`, version `persona-sentences-v1`; its SHA-256 is saved with outputs. Ambiguous overlaps use refusal > meta-stance > first-person > neutral. A first-person planning phrase such as “I will calculate” is neutral under this rubric. Segmentation is deterministic punctuation/newline splitting, with stable character offsets and previous/next-sentence context. It may split abbreviations or code; assess this during calibration.

Extraction automatically writes `RUN/review/sentences.jsonl`, an empty `manual_calibration.csv`, and `review.md`/`review.csv`. At least 30 calibration sentences are sampled across role/default and think/answer/domain groups without consulting labels. The manual review pack requests at least 20 distinct prompt CoTs per domain and one CoT from each of at least 20 distinct roles. Small smoke runs explicitly report sampling shortfalls. Exports never imply that a human has completed the work.

You can regenerate a deterministic review pack independently:

```bash
source .venv/bin/activate
python -m persona_dynamics.judge export \
  --transcripts runs/qwen3-32b/review_transcripts.jsonl \
  --output runs/qwen3-32b/review --seed 2026
```

Do this before editing annotation files: regenerating the pack resets its blank annotation templates. Hand-label at least 30 sampled sentences in the `manual_label` column, recording annotator and notes. Read the sampled full CoTs in `review.md` and complete `review.csv`. A blank field remains unfinished work. Freeze the rubric before comparing it to geometry. If you revise the rubric after calibration, version it and assess agreement on a newly held-out human set rather than reporting the tuning sample as independent validation.

The optional client makes OpenAI-compatible chat-completion requests only when explicitly invoked. It uses an environment key, temperature zero, fixed prompt text, strict sentence-ID/label parsing, bounded retries, and raw-response caching. It requires 30 imported human labels first. Choose a fixed model snapshot supported by the selected provider:

```bash
python -m persona_dynamics.judge api \
  --sentences runs/qwen3-32b/review/sentences.jsonl \
  --manual runs/qwen3-32b/review/manual_calibration.csv \
  --model YOUR_PINNED_JUDGE_MODEL \
  --output runs/qwen3-32b/api_labels.jsonl \
  --cache runs/qwen3-32b/judge_cache
```

Set `OPENAI_API_KEY` in your shell beforehand. For another compatible provider use `--base-url https://YOUR_PROVIDER/v1 --api-key-env YOUR_KEY_VARIABLE`. The command intentionally sends sentence text, neighboring sentences, and the role's system prompt to that provider and may incur charges. No other pipeline command calls this API. Secrets are never written to configs or raw caches. Cache keys cover endpoint, model/settings, exact context, and rubric; cached provider model and fingerprint are retained when supplied. A pinned name and a cache improve reproducibility but do not guarantee a remote provider's deterministic behavior.

External tools may instead produce JSONL with one `sentence_id` and `label` per canonical sentence, plus optional judge provenance such as `model`, `prompt_version`, and `label_source`. Unknown, duplicate, or invalid labels fail validation; supplied sentence text must match exactly. Attach complete labels and optional real human annotations:

```bash
python -m persona_dynamics.judge attach \
  --run-dir runs/qwen3-32b \
  --labels runs/qwen3-32b/api_labels.jsonl \
  --manual runs/qwen3-32b/review/manual_calibration.csv
python -m persona_dynamics analyze runs/qwen3-32b
```

`attach` writes sentence-level `judge_labels.csv` with transcript/prompt/role/segment identity for the main analysis, plus `judge_attachment.json` with calibration coverage, accuracy, Cohen's kappa, and the confusion matrix. It rejects incomplete prediction coverage to prevent silent selective annotation. Manual labels are included only where a person supplied them. No automatic “validated judge” badge is assigned, and completing sentence calibration does not mark full-CoT review complete.

To inspect agreement without attaching labels:

```bash
python -m persona_dynamics.judge agreement \
  --sentences runs/qwen3-32b/review/sentences.jsonl \
  --manual runs/qwen3-32b/review/manual_calibration.csv \
  --predictions runs/qwen3-32b/api_labels.jsonl \
  --output runs/qwen3-32b/judge_agreement.json
```

Accuracy and kappa use only exact overlapping IDs and report unmatched IDs. Kappa is undefined when both raters use one identical category. Report the sample size and label distribution beside agreement; 30 sentences can provide only a coarse reliability check. Treat refusal flags separately and compare the primary role analysis with refusal-excluded sensitivity results. Do not discard inconvenient samples without reporting that choice.

For code integration, use `sentence_records(transcripts)`, `import_labels(path, sentences, manual=False)`, `merge_sentence_labels(sentences, labels, manual=None)`, then pass the resulting sentence DataFrame to `analysis.analyze(..., judge=...)`. `summarize_labels` is a separate convenience for transcript × segment fractions; it is not the input schema for the main analysis. Multiple seeds and sentences do not become independent statistical samples: aggregate through the prompt or role as described in the analysis protocol.

`heuristic_labels` is a deliberately simple regex **unvalidated offline baseline** for testing data flow. It never generates human labels and must never be reported as calibrated judge evidence.
