"""Fetch pinned upstream artifacts; no upstream model code is executed."""
from __future__ import annotations

import io
from pathlib import Path
import urllib.request
import zipfile

from .io import file_hash, write_json

UPSTREAM_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"
VECTORS_COMMIT = "3b3b788432ad33e3a28d9ff08e88a530c0740814"


def fetch_assets(output="data/assets", prepared="data/prepared", seed=2026):
    from huggingface_hub import snapshot_download
    from .data import import_upstream_assets
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    upstream = root / "assistant-axis"
    if not (upstream / "data/extraction_questions.jsonl").exists():
        url = f"https://codeload.github.com/safety-research/assistant-axis/zip/{UPSTREAM_COMMIT}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for info in archive.infolist():
            relative = Path(*Path(info.filename).parts[1:])
            if not relative.parts or ".." in relative.parts or relative.is_absolute():
                continue
            # Retain the source methodology and data, not executable installation hooks.
            if relative.parts[0] not in {"data", "pipeline", "assistant_axis", "README.md", "LICENSE"}:
                continue
            destination = upstream / relative
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
    snapshot_download("lu-christina/assistant-axis-vectors", repo_type="dataset", revision=VECTORS_COMMIT,
        local_dir=root / "vectors", allow_patterns=["qwen-3-32b/assistant_axis.pt",
        "qwen-3-32b/default_vector.pt", "qwen-3-32b/role_vectors/*.pt"])
    roles = import_upstream_assets(upstream, root / "vectors/qwen-3-32b", layer=32,
                                  output_dir=prepared, seed=seed)
    files = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*.pt")}
    write_json(root / "manifest.json", {"upstream_commit": UPSTREAM_COMMIT,
               "vectors_commit": VECTORS_COMMIT, "sha256": files, "selection_seed": seed,
               "axis_model": "Qwen/Qwen3-32B", "layer": 32})
    return roles
