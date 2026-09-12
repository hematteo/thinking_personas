"""Freeze public two-domain inputs before any model outputs are observed."""
from collections import Counter
import json
from pathlib import Path
import random

from datasets import load_dataset
from huggingface_hub import HfApi
from persona_dynamics.data import ADVICE_PATTERN, text_hash, validate_prompts, write_jsonl


def main():
    root = Path("data/prepared")
    if (root / "prompts.jsonl").exists():
        raise RuntimeError("Refusing to replace frozen prompts")
    api = HfApi()
    repos = ["allenai/WildChat-1M", "openai/gsm8k"]
    revisions = {repo: api.dataset_info(repo).sha for repo in repos}
    rows, seen = [], set()
    scanned = 0
    for item in load_dataset(repos[0], revision=revisions[repos[0]], split="train", streaming=True):
        scanned += 1
        conversation = item.get("conversation", [])
        if not conversation or conversation[0].get("role") != "user":
            continue
        first = conversation[0]
        text = first.get("content", "")
        language = item.get("language", first.get("language", "English"))
        if language not in ("English", "english", "en") or not 40 <= len(text) <= 4000 or not ADVICE_PATTERN.search(text):
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        rows.append(dict(prompt_id=f"advice-{digest[:16]}", domain="advice", text=text,
                         source=f"hf:{repos[0]}@{revisions[repos[0]]}/train",
                         source_id=str(item.get("conversation_hash", item.get("conversation_id", scanned - 1))),
                         selection_filter="first-turn-personal-keyword-v1"))
        if len(rows) == 80:
            break
        if scanned >= 100000:
            raise RuntimeError("Insufficient advice candidates within fixed scan budget")
    if len(rows) != 80:
        raise RuntimeError("Need 80 deduplicated public advice candidates")
    gsm = load_dataset(repos[1], "main", revision=revisions[repos[1]], split="test")
    for i, item in enumerate(gsm):
        rows.append(dict(prompt_id=f"math-{i}", domain="math", text=item["question"],
                         source=f"hf:{repos[1]}@{revisions[repos[1]]}/main/test",
                         source_id=str(i), reference=item["answer"]))
    result = []
    for domain in ("advice", "math"):
        group = sorted((r for r in rows if r["domain"] == domain), key=lambda r: r["prompt_id"])
        random.Random(f"2026:{domain}").shuffle(group)
        ranks = Counter()
        for i, row in enumerate(group):
            split = "calibration" if i < 8 else "e0" if i < 13 else "eval"
            row.update(split=split, candidate_rank=ranks[split], selection_seed=2026)
            ranks[split] += 1
            result.append(row)
    write_jsonl(root / "prompts.jsonl", validate_prompts(result))
    manifest = dict(revisions=revisions, selection_seed=2026, advice_scanned=scanned,
                    counts=dict(Counter(f"{r['domain']}/{r['split']}" for r in result)),
                    assumptions=["Advice is a convenience sample: first 80 distinct matching English first turns in pinned WildChat order, then shuffled before partitioning.",
                                 "The keyword filter is not a validated advice classifier; manual domain review remains required.",
                                 "Eight calibration and five E0 prompts per domain are held out; the profile uses three E0 questions from the frozen order."])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
