"""Collect stricter public advice candidates for review before generation."""
import json
from pathlib import Path
import re
from datasets import load_dataset
from persona_dynamics.data import text_hash, write_jsonl

root = Path("data/prepared")
revision = json.loads((root / "manifest.json").read_text())["revisions"]["allenai/WildChat-1M"]
include = re.compile(r"\b(my (?:boyfriend|girlfriend|husband|wife|partner|friend|family|parents|mother|father|sister|brother|boss|job|career|relationship)|i (?:feel|am feeling|struggle|am struggling)|should i (?:break up|quit|tell|leave|stay|accept|apologize))\b", re.I)
exclude = re.compile(r"\b(roleplay|role play|pretend|act as|story|fiction|novel|character|lyrics|poem|rewrite|translate|translation|write a|write an|write me|programming|python|javascript|code|jailbreak|dan|machiavellian|unfiltered|erotic|porn|sex scene|reality shifting)\b", re.I)
extra_exclude = re.compile(r"dragon ball|monster hunter|\b(letter|essay|prayer|pray|dialog|cmake|paraphrase|correct|grammar|paragraph|rewrite|novelist|socrates|socratic|act like|as if you were|as my girlfriend|onlyfans|give me a.*liner)\b", re.I)
direct = re.compile(r"^(?:(?:hi|hello|hey)[,!. ]*)?(?:i[ '\u2019]|my |how (?:do|can|should) i|what (?:should|can) i|should i|could you (?:give me|help me)|can you (?:give me|help me))", re.I)
rows, seen = [], set()
for index, item in enumerate(load_dataset("allenai/WildChat-1M", revision=revision, split="train", streaming=True)):
    if index >= 400000:
        break
    conversation = item.get("conversation", [])
    if not conversation or conversation[0].get("role") != "user":
        continue
    first = conversation[0]
    text = first.get("content", "")
    if item.get("language", first.get("language", "English")) not in ("English", "english", "en"):
        continue
    if not 40 <= len(text) <= 1800 or not include.search(text) or exclude.search(text):
        continue
    if extra_exclude.search(text) or not direct.search(text.strip()):
        continue
    digest = text_hash(text)
    if digest in seen:
        continue
    seen.add(digest)
    rows.append(dict(prompt_id=f"advice-{digest[:16]}", domain="advice", text=text,
                     source=f"hf:allenai/WildChat-1M@{revision}/train", source_id=str(item.get("conversation_hash", index)),
                     source_row=index, selection_filter="personal-context-v2-plus-manual-review"))
    if len(rows) % 20 == 0:
        write_jsonl(root / "advice_candidates_v3.jsonl", rows)
        print({"candidates": len(rows), "scanned": index + 1}, flush=True)
    if len(rows) == 100:
        break
write_jsonl(root / "advice_candidates_v3.jsonl", rows)
print({"candidate_count": len(rows), "rows_scanned": index + 1}, flush=True)
