"""Download only the pinned tokenizer/config, never model weights, for integration tests."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from persona_dynamics.io import file_hash, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/assets/test-tokenizer")
    args = parser.parse_args()
    revision = "9216db5781bf21249d130ec9da846c4624c16137"
    files = ["config.json", "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json"]
    snapshot_download("Qwen/Qwen3-32B", revision=revision, local_dir=args.output, allow_patterns=files)
    write_json(Path(args.output) / "provenance.json", {"model": "Qwen/Qwen3-32B", "revision": revision,
               "files": {name: file_hash(Path(args.output) / name) for name in files}})
    print(f"PERSONA_QWEN_TOKENIZER_DIR={args.output} python -m pytest -q")
