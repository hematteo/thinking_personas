"""Reuse a frozen run only for increasing its fixed-order candidate cap."""
import json
from pathlib import Path
import shutil

from persona_dynamics.io import file_hash, read_jsonl, write_json
from persona_dynamics.pipeline import initialize, make_requests, require_gate


def extend(config, source):
    source = Path(source)
    require_gate(source)
    old = json.loads((source / "config.json").read_text())
    new = config.to_dict()
    allowed = {"name", "output_dir", "candidate_limit_per_domain"}
    if any(old[k] != new[k] for k in new if k not in allowed):
        raise ValueError("Candidate extension must preserve every scientific/generation setting")
    if new["candidate_limit_per_domain"] <= old["candidate_limit_per_domain"]:
        raise ValueError("Candidate cap must strictly increase")
    if Path(config.output_dir).resolve() == source.resolve():
        raise ValueError("Extension must use a new run directory")
    target = Path(config.output_dir)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Extension target must be empty")
    target, _, _, child_id = initialize(config)
    parent = json.loads((source / "manifest.json").read_text())
    child = json.loads((target / "manifest.json").read_text())
    for key in ("inputs", "code", "numerical_versions", "python"):
        if parent[key] != child[key]:
            raise ValueError(f"Cannot reuse artifacts with changed {key}")
    old_requests = {}
    for phase in ("gate", "experiments"):
        old_requests.update({r["record_id"]: r for r in read_jsonl(source / f"requests_{phase}.jsonl")})
    # The gate is identical: preserve its original measured result and ruler.
    for name in ("e0.json", "e0_tokens.csv.gz", "directions.npz"):
        shutil.copy2(source / name, target / name)
    requests = make_requests(config, "gate") + make_requests(config, "experiments")
    (target / "transcripts").mkdir()
    hashes = {}
    for request in requests:
        rid = request["record_id"]
        if rid not in old_requests:
            continue
        original = old_requests[rid]
        if {k: v for k, v in original.items() if k != "run_id"} != {k: v for k, v in request.items() if k != "run_id"}:
            raise ValueError(f"Changed cached request {rid}")
        path = source / "transcripts" / f"{rid}.json"
        record = json.loads(path.read_text())
        assert record["run_id"] == parent["run_id"]
        hashes[rid] = file_hash(path)
        record.update(run_id=child_id, source_run_id=parent["run_id"])
        write_json(target / "transcripts" / path.name, record)
    write_json(target / "extension.json", {
        "source": str(source), "source_run_id": parent["run_id"], "source_manifest_sha256": file_hash(source / "manifest.json"),
        "source_transcript_sha256": hashes, "reused_transcripts": len(hashes),
        "old_cap": old["candidate_limit_per_domain"], "new_cap": new["candidate_limit_per_domain"],
        "reason": "Original math cohort had 36 eligible pairs versus required 40. Extend the previously frozen candidate order; preserve thresholds, seed, ruler and all existing outputs.",
        "selection": "First 40 eligible prompts in the original fixed order; no projection/effect outcomes used"})
    print(f"Extended run: reused {len(hashes)} transcripts; {len(requests)-len(hashes)} new requests", flush=True)
