"""CPU-only descriptive diagnostics and balanced transcript export.

These supplement the frozen primary analysis; domain breakdowns are exploratory.
No behavioral labels or human review status are inferred by this script.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    assert not manifest["synthetic"], "This supplement is for the actual model run."
    effects = pd.read_csv(root / "effects.csv")
    pairs = pd.read_csv(root / "paired_question_effects.csv")
    means = pd.read_csv(root / "segment_means.csv")
    attrition = pd.read_csv(root / "attrition.csv")
    role_mask = effects.arm.str.startswith("role_")
    primary = effects[role_mask & effects.metric.eq("dot_assistant")
                      & effects.policy.eq("all_content")
                      & effects.contrast.eq("segment_interaction")
                      & effects.seed_scope.eq("pooled")].copy()
    primary["low_coverage"] = primary.n.lt(8)
    primary.to_csv(out / "primary_interactions.csv", index=False)
    effects[role_mask & effects.metric.isin(["dot_assistant", "cos_assistant"])
            & effects.contrast.eq("segment_interaction")].to_csv(
                out / "metric_seed_token_sensitivities.csv", index=False)

    chosen = pairs[pairs.metric.isin(["dot_assistant", "cos_assistant"])
                   & pairs.policy.eq("all_content")]
    group = ["arm", "reference_arm", "comparison", "metric", "contrast", "domain"]
    question = chosen.groupby(group + ["prompt_id"], as_index=False).value.mean()
    domain = question.groupby(group).value.agg(["count", "mean", "min", "max"])
    domain.rename(columns={"count": "n_questions"}).to_csv(out / "exploratory_domain_effects.csv")
    attrs = attrition.groupby(["arm", "domain"])
    attrs.agg(requested=("record_id", "count"), eligible=("eligible", "sum"),
              think_min=("think_token_count", "min"),
              think_median=("think_token_count", "median"),
              think_max=("think_token_count", "max"),
              answer_median=("answer_token_count", "median")).to_csv(out / "coverage_and_lengths.csv")
    means[means.segment.isin(["think", "answer"])].groupby(["arm", "segment"])[
        ["resid_norm", "centered_norm"]].mean().to_csv(out / "descriptive_activation_norms.csv")

    controls = ["dot_assistant", "dot_ctrl_1", "dot_ctrl_2", "dot_null", "dot_random"]
    s = effects[role_mask & effects.reference_arm.eq("default")
                & effects.policy.eq("all_content") & effects.seed_scope.eq("pooled")
                & effects.contrast.eq("segment_interaction") & effects.metric.isin(controls)]
    matrix = s.pivot(index="arm", columns="metric", values="mean")[controls]
    matrix.to_csv(out / "control_interaction_means.csv")
    limit = np.max(np.abs(matrix.to_numpy()))
    fig, ax = plt.subplots(figsize=(8, 5.8), layout="constrained")
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(matrix)), [x[5:] for x in matrix.index])
    ax.set_xticks(range(len(controls)), ["Assistant", "Skeptic", "Judge", "Structured null", "Random"])
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > .65 * limit else "black")
    ax.set_title("Descriptive segment interactions across unit directions\nRole − default; paired questions, seeds averaged")
    fig.colorbar(im, ax=ax, label="Thinking shift − answer shift · raw units")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"control_interactions.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # A labeled-row alternative to the frozen scatter, whose near-zero roles
    # overlap visually. This changes presentation only, using the same estimates.
    s = effects[effects.reference_arm.eq("default") & effects.policy.eq("all_content")
                & effects.seed_scope.eq("pooled") & effects.metric.eq("dot_assistant")
                & effects.contrast.isin(["think_shift", "answer_shift"])]
    order = [a["arm"] for a in json.loads((root / "arms.json").read_text()) if a["arm"] != "default"]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    y = np.arange(len(order))
    for contrast, offset, color, label in [("think_shift", -.13, "#2962a3", "Thinking"),
                                         ("answer_shift", .13, "#b58a28", "Answer")]:
        frame = s[s.contrast.eq(contrast)].set_index("arm").loc[order]
        ax.errorbar(frame["mean"], y + offset,
                    xerr=[frame["mean"] - frame.ci_low, frame.ci_high - frame["mean"]],
                    fmt="o", capsize=2, color=color, label=label, markersize=5)
    ax.set_yticks(y, [a.replace("role_", "").replace("style_", "style: ") for a in order])
    ax.invert_yaxis()
    ax.axvline(0, color="#555", lw=.8)
    ax.axhline(8.5, color="#bbb", lw=.8)
    ax.set(xlabel="Change from default · raw Assistant projection",
           title="Role and style interventions\nMatched questions; 95% pointwise intervals")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.legend(loc="lower left")
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"role_and_style_shifts.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    records = [json.loads(p.read_text()) for p in (root / "transcripts").glob("*.json")]
    # IDs chosen during generation, before inspecting activation results.
    review_prompts = ["role2-identity-01", "role2-social-04", "role2-reasoning-04"]
    review_arms = ["default", "role_alien", "role_doctor", "role_mystic", "role_poet", "style_poet"]
    selected = sorted([r for r in records if r["seed"] == 17 and r["prompt_id"] in review_prompts
                       and r["arm"] in review_arms], key=lambda r: (r["prompt_id"], r["arm"]))
    assert len(selected) == 18
    lines = ["# Balanced qualitative inspection packet", "",
             "18 full transcripts: six arms × three questions, seed 17. Selected without projection values. "
             "This is a purposive qualitative sample, not an estimated role-adoption rate. "
             "The original automated review packet selects identity questions only; this supplement covers all three domains. "
             "No human annotation or agreement score has been completed.", ""]
    for r in selected:
        lines += [f"## {r['prompt_id']} · {r['arm']} · seed {r['seed']}", "",
                  f"Record: `{r['record_id']}`", "", "Prompt:", r["messages"][-1]["content"], "",
                  "Thinking:", r["think_text"], "", "Answer:", r["answer_text"], ""]
    (out / "balanced_transcripts.md").write_text("\n".join(lines))
    (out / "balanced_transcripts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in selected))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": manifest["run_id"], "supplement_is_exploratory": True,
        "domain_inference": "Descriptive only; four fixed questions per domain",
        "human_review_complete": False, "qualitative_record_ids": [r["record_id"] for r in selected]
    }, indent=2) + "\n")
    print(json.dumps({"primary_interactions": len(primary), "low_coverage": int(primary.low_coverage.sum()),
                      "review_transcripts": len(selected), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
