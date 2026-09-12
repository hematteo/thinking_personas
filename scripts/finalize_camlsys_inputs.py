"""Apply a frozen, pre-generation semantic screen to public advice candidates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
from persona_dynamics.data import read_jsonl, validate_prompts, write_jsonl
from persona_dynamics.io import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("prepared")
args = parser.parse_args()
root = Path(args.prepared)
review = json.loads((root / "advice_review.json").read_text())
candidates = {}
for name, expected_hash in review["candidate_files"].items():
    assert file_hash(root / name) == expected_hash
    candidates.update({r["prompt_id"]: r for r in read_jsonl(root / name)})
accepted = review["accepted_prompt_ids"]
assert len(accepted) == len(set(accepted)) and len(accepted) >= 63
rows = [dict(candidates[key]) for key in accepted]
rows.sort(key=lambda r: r["prompt_id"])
random.Random("2026:advice").shuffle(rows)
ranks = Counter()
for i, row in enumerate(rows):
    split = "calibration" if i < 8 else "e0" if i < 13 else "eval"
    row.update(split=split, candidate_rank=ranks[split], selection_seed=2026,
               selection_filter="Codex-semantic-input-screen-before-generation")
    ranks[split] += 1
existing = read_jsonl(root / "prompts.jsonl")
rows += [row for row in existing if row["domain"] == "math"]
validate_prompts(rows)
archive = root / "rejected_initial_filter"
archive.mkdir(exist_ok=True)
for name in ("prompts.jsonl", "manifest.json"):
    if (archive / name).exists():
        raise RuntimeError("Inputs have already been finalized; do not rerun")
    shutil.copy2(root / name, archive / name)
manifest = json.loads((root / "manifest.json").read_text())
manifest["counts"] = dict(Counter(f"{r['domain']}/{r['split']}" for r in rows))
manifest["advice_review"] = review
manifest["assumptions"] = [
    "Advice is a convenience sample of 63 semantically screened public WildChat first turns, not a representative random sample.",
    "Screening covers personal, interpersonal and practical open-ended advice including work, learning, health, legal procedures and leisure; rejects fiction, coding/math tasks, pure rewriting and obvious duplicate templates.",
    "Codex performed the semantic input screen before any model outputs; this is not independent human validation and does not count as CoT/judge annotation.",
    "Prompt-level inference does not establish independence across underlying WildChat users.",
    "Eight calibration and five E0 prompts are held out per domain; profile uses first three E0 questions and 50 advice evaluation candidates."]
write_jsonl(root / "prompts.jsonl", rows)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(manifest["counts"])
