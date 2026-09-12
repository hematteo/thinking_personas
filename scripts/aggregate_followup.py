"""Descriptive aggregation across fixed role sets, retaining question clusters.

This is post hoc to the frozen study. Intervals resample questions only; they
do not describe uncertainty over role populations, models, or future seeds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_dir, args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    spec = manifest["spec"]
    seeds = spec["seeds"]
    questions = [json.loads(x) for x in (root / "questions.jsonl").read_text().splitlines()]
    arms = json.loads((root / "arms.json").read_text())
    roles = [a["arm"] for a in arms if a["arm"].startswith("role_")]
    styles = [a["arm"] for a in arms if a["arm"].startswith("style_")]
    matched = [a.replace("style_", "role_", 1) for a in styles]
    pairs = pd.read_csv(root / "paired_question_effects.csv")
    pairs = pairs[pairs.policy.eq("all_content")]
    groups = {
        "nine_fixed_roles_vs_default": [(a, "default") for a in roles],
        "three_matched_roles_vs_default": [(a, "default") for a in matched],
        "three_styles_vs_default": [(a, "default") for a in styles],
        "three_matched_roles_vs_style": list(zip(matched, styles)),
    }
    summary, domains, observations = [], [], []
    for name, comparisons in groups.items():
        wanted = set(comparisons)
        selected = pairs[[pair in wanted for pair in zip(pairs.arm, pairs.reference_arm)]]
        for metric in ["dot_assistant", "cos_assistant"]:
            for contrast in ["think_shift", "answer_shift", "segment_interaction"]:
                g = selected[selected.metric.eq(metric) & selected.contrast.eq(contrast)]
                assert not g.duplicated(["prompt_id", "seed", "arm", "reference_arm"]).any()
                assert set(g.seed) == set(seeds)
                assert g.groupby("prompt_id").size().eq(len(comparisons) * len(seeds)).all()
                assert g.prompt_id.nunique() == len(questions)
                # Same fixed role and seed weights within every question.
                q = g.groupby(["prompt_id", "domain"], sort=True).value.mean().reset_index()
                identity = dict(group=name, metric=metric, contrast=contrast)
                observations.extend([{**identity, **row} for row in q.to_dict("records")])
                values = q.value.to_numpy()
                rng = np.random.default_rng(spec["bootstrap_seed"])
                boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                summary.append({**identity, "n_roles": len(comparisons), "n_seeds": len(seeds),
                                "n_questions": len(q), "mean": values.mean(),
                                "pointwise_ci_low": lo, "pointwise_ci_high": hi})
                for domain, d in q.groupby("domain"):
                    domains.append({**identity, "domain": domain, "n_questions": len(d), "mean": d.value.mean()})
    pd.DataFrame(summary).to_csv(out / "fixed_role_aggregates.csv", index=False)
    pd.DataFrame(domains).to_csv(out / "fixed_role_domain_means.csv", index=False)
    pd.DataFrame(observations).to_csv(out / "question_aggregates.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "run_id": manifest["run_id"], "synthetic": manifest["synthetic"],
        "post_hoc": True, "groups": groups,
        "weighting": "Equal fixed roles and seeds within question; equal questions",
        "intervals": "10000 question bootstrap draws; pointwise 95%; no multiplicity adjustment",
        "scope": "Fixed role sets and model; not role-population or model-population inference",
        "original_study_pooled": False,
    }, indent=2) + "\n")
    print(pd.DataFrame(summary).query("metric == 'dot_assistant'").to_string(index=False))


if __name__ == "__main__":
    main()
