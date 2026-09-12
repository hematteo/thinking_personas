"""Verify a completed run's artifacts without trusting its narrative report."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(root):
    root = Path(root)
    e0 = json.loads((root / "e0.json").read_text())
    assert e0["passed"] and all(x["passed"] for x in e0["per_seed"])
    tokens = pd.read_csv(root / "tokens.csv.gz", low_memory=False)
    assert not tokens.duplicated(["record_id", "token_idx"]).any()
    assert tokens.token_idx.ge(0).all()
    assert tokens.norm_pos.between(0, 1).all()
    measured = [c for c in tokens if c.startswith("cos_") and tokens[c].notna().any()]
    assert {"cos_assistant", "cos_ctrl_1", "cos_ctrl_2", "cos_null", "cos_random"}.issubset(measured)
    assert np.isfinite(tokens[measured].to_numpy()).all()
    assert (tokens[measured].abs() <= 1 + 1e-6).all().all()
    assert not tokens.loc[tokens.segment.eq("boundary"), "primary_include"].any()
    assert tokens.loc[tokens.segment.eq("think"), "pos_from_boundary"].lt(0).all()
    assert tokens.loc[tokens.segment.eq("answer"), "pos_from_boundary"].gt(0).all()
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["synthetic"] == (config["backend"] == "fixture")
    assert summary["seeds"] == config["seeds"]
    assert summary["n_token_rows"] == len(tokens)
    extraction = json.loads((root / "extraction.json").read_text())
    e2_complete = extraction.get("transplants_included", bool(tokens.experiment.eq("E2").any()))
    if "E2" in config["experiments"] and e2_complete:
        required = {"cot_as_answer", "cot_as_cot", "answer_as_cot", "answer_as_answer", "cot_in_scratch", "answer_in_scratch"}
        if config.get("e2_stepbystep", True):
            required.add("nothink_stepbystep")
        assert required.issubset(tokens.condition.unique())
    for figure in summary["figures"]:
        assert (root / figure).stat().st_size > 1000
        assert (root / figure).with_suffix(".pdf").stat().st_size > 1000
    gaps = pd.read_csv(root / "e1_effects.csv")
    for cohort in json.loads((root / "cohort.json").read_text()):
        n = gaps.loc[gaps.domain.eq(cohort["domain"]) & gaps.seed_scope.eq("pooled"), "n"]
        assert n.eq(cohort["selected"]).all(), "CI observations must be prompts, not tokens or seeds"
    print(json.dumps({"verified": True, "synthetic": summary["synthetic"], "tokens": len(tokens),
                      "directions": len(measured), "figures": len(summary["figures"]),
                      "e2_transplants_complete": e2_complete}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    verify(parser.parse_args().run_dir)
