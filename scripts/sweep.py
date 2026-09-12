"""Run explicit named ablations without pooling incompatible experimental rulers.

Example: python scripts/sweep.py --config configs/smoke.yaml --grid configs/ablations.json
Each grid entry is {name, overrides}; all runs retain the E0 gate.
"""
import argparse
import json
from pathlib import Path

from persona_dynamics.config import load_config, Config
from persona_dynamics.pipeline import run_pipeline
from persona_dynamics.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", default="runs/ablations")
    args = parser.parse_args()
    base = load_config(args.config)
    plan = json.loads(Path(args.grid).read_text())
    names = [p["name"] for p in plan]
    if len(set(names)) != len(names) or any("/" in n or n in {".", ".."} for n in names):
        raise ValueError("Ablations need unique simple directory names")
    results = []
    for entry in plan:
        config = Config(**{**base.to_dict(), **entry["overrides"], "name": entry["name"],
                          "output_dir": str(Path(args.output) / entry["name"])}).validate()
        root = run_pipeline(config)
        results.append({"name": entry["name"], "overrides": entry["overrides"], "results": str(root / "results.md")})
    write_json(Path(args.output) / "index.json", results)


if __name__ == "__main__":
    main()
