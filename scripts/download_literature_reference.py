"""Download the pinned public Qwen role-vector release; never model weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = "lu-christina/assistant-axis-vectors"
REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
GITHUB_REVISION = "a98961956072224eaf244eb289d6c01700b63795"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/literature-reference"))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "reference_layer32.npz"
    cache_manifest = out / "reference_layer32_manifest.json"
    if cache.exists() and cache_manifest.exists():
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == json.loads(cache_manifest.read_text())["sha256"]
        print("Verified compact layer-32 reference cache", flush=True)
        return
    tree_path = out / "hf_tree.json"
    if not tree_path.exists():
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/qwen-3-32b?recursive=true&limit=1000"
        with urllib.request.urlopen(url, timeout=60) as response:
            tree_path.write_bytes(response.read())
    tree = json.loads(tree_path.read_text())
    entries = [e for e in tree if "/role_vectors/" in e["path"] or
               e["path"] in ["qwen-3-32b/default_vector.pt", "qwen-3-32b/assistant_axis.pt"]]
    assert len(entries) == 277, "Unexpected pinned release inventory"

    def download(entry):
        target = out / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = entry["lfs"]["oid"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return entry["path"], expected
        for attempt in range(4):
            try:
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['path']}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                assert len(content) == entry["size"]
                assert hashlib.sha256(content).hexdigest() == expected
                target.write_bytes(content)
                return entry["path"], expected
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    hashes = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (path, digest) in enumerate(pool.map(download, entries), 1):
            hashes[path] = digest
            if i % 50 == 0 or i == len(entries):
                print(f"Verified {i}/{len(entries)} public vector files", flush=True)
    (out / "vector_download_manifest.json").write_text(json.dumps({
        "repository": REPO, "revision": REVISION, "role_count": 275,
        "hashes": hashes, "sha256_checked_against_pinned_hf_lfs_metadata": True,
    }, indent=2) + "\n")
    sources = {
        "tokenizer.json": f"https://huggingface.co/Qwen/Qwen3-32B/resolve/{MODEL_REVISION}/tokenizer.json",
        "notebooks__pca.ipynb": f"https://raw.githubusercontent.com/safety-research/assistant-axis/{GITHUB_REVISION}/notebooks/pca.ipynb",
        "assistant_axis__pca.py": f"https://raw.githubusercontent.com/safety-research/assistant-axis/{GITHUB_REVISION}/assistant_axis/pca.py",
    }
    provenance = {}
    for name, url in sources.items():
        target = out / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        provenance[name] = {"url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (out / "source_download_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # The full tensors contain 64 layers. Keep only the requested layer after
    # checking every original hash, to avoid retaining 181 MB unnecessarily.
    import numpy as np
    import torch
    paths = sorted((out / "qwen-3-32b/role_vectors").glob("*.pt"))
    def layer(path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(value.shape) == (64, 5120)
        return value[32].float().numpy()
    np.savez_compressed(cache, role_vectors=np.stack([layer(p) for p in paths]),
                        labels=np.array([p.stem for p in paths]),
                        default=layer(out / "qwen-3-32b/default_vector.pt"),
                        assistant=layer(out / "qwen-3-32b/assistant_axis.pt"))
    cache_manifest.write_text(json.dumps({"sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "layer": 32, "repository": REPO, "revision": REVISION,
        "conversion": "Original bf16 tensors loaded with weights_only=True, layer 32 converted exactly to float32.",
        "original_hashes": hashes}, indent=2) + "\n")
    for entry in entries:
        (out / entry["path"]).unlink()
    print("Saved compact layer-32 cache; removed downloaded 64-layer tensors", flush=True)


if __name__ == "__main__":
    main()
