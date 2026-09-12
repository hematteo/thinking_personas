"""Publication-style figures from scalar projections and prompt-level tables.

All uncertainty is computed after token aggregation. PNG and vector PDF files
are written together; PNG names are returned for the generated Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import bootstrap_summary, direction_columns


BLUE, GOLD, ORANGE, OLIVE, PINK = "#2962a3", "#b58a28", "#c76536", "#73823f", "#a35782"
PALETTE = [BLUE, GOLD, ORANGE, OLIVE, PINK]
LABELS = {"cos_assistant": "Assistant axis", "cos_ctrl_1": "Role control 1", "cos_ctrl_2": "Role control 2",
          "cos_null": "Structured null", "cos_random": "Random", **{f"cos_pc{i}": f"Answer PC{i}" for i in range(1, 6)}}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "axes.axisbelow": True, "savefig.dpi": 160, "pdf.fonttype": 42})


def _controls(tokens):
    return [d for d in ["cos_assistant", "cos_ctrl_1", "cos_ctrl_2", "cos_null", "cos_random"]
            if d in direction_columns(tokens)]


def _save(fig, output_dir, name, synthetic=False):
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    if synthetic:
        fig.text(.5, -.018, "Synthetic pipeline validation — no research conclusions", ha="center", va="top", fontsize=9, color="#923928")
    fig.savefig(path/f"{name}.png", bbox_inches="tight")
    fig.savefig(path/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}.png"


def _panels(n, width=5, height=3.6, columns=2, sharey=False, sharex=False):
    cols = min(columns, max(1, n))
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width*cols, height*rows), squeeze=False,
                             sharey=sharey, sharex=sharex, layout="constrained")
    for ax in axes.flat[n:]:
        ax.remove()
    return fig, list(axes.flat[:n])


def _curve(data, x, directions, bootstrap_samples, seed):
    """One observation per prompt after equal-weight averaging over seeds."""
    keys = ["seed", "prompt_id", "domain", x]
    per_seed = data.groupby(keys, dropna=False)[directions].mean().reset_index()
    prompts = per_seed.groupby(["prompt_id", "domain", x], dropna=False)[directions].mean().reset_index()
    rows = []
    for (domain, position), sub in prompts.groupby(["domain", x], dropna=False):
        for direction in directions:
            rows.append(dict(domain=domain, position=position, direction=direction,
                             **bootstrap_summary(sub[direction], bootstrap_samples, seed)))
    return pd.DataFrame(rows)


def _line(ax, curve, direction, color):
    rows = curve.loc[curve.direction.eq(direction)].sort_values("position")
    if rows.empty:
        return
    ax.plot(rows.position, rows["mean"], label=LABELS.get(direction, direction), color=color,
            lw=2 if direction == "cos_assistant" else 1.2,
            linestyle="-" if direction == "cos_assistant" else "--")
    lo, hi = pd.to_numeric(rows.ci_low), pd.to_numeric(rows.ci_high)
    if lo.notna().any():
        ax.fill_between(rows.position.to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), color=color, alpha=.12)


def plot_trajectory(tokens, output_dir, bootstrap_samples=1000, seed=0, synthetic=False):
    data = tokens.loc[tokens.experiment.eq("E1") & tokens.condition.eq("natural_think") & tokens.primary_include &
                      tokens.segment.isin(["think", "answer"])].copy()
    if data.empty or "norm_pos" not in data:
        return None
    directions = _controls(tokens)
    data["bin"] = np.minimum((data.norm_pos*20).astype(int), 19)/20+.025+data.segment.eq("answer").astype(int)
    curves = _curve(data, "bin", directions, bootstrap_samples, seed)
    facets = sorted(data.domain.unique())
    fig, axes = _panels(len(facets), sharey=True)
    for ax, domain in zip(axes, facets):
        for direction, color in zip(directions, PALETTE):
            _line(ax, curves.loc[curves.domain.eq(domain)], direction, color)
        reference = tokens.loc[tokens.experiment.eq("E1") & tokens.domain.eq(domain) &
                               tokens.condition.eq("natural_nothink") & tokens.primary_include & tokens.segment.eq("answer")]
        if not reference.empty:
            value = reference.groupby(["seed", "prompt_id"]).cos_assistant.mean().groupby("prompt_id").mean().mean()
            ax.axhline(value, ls=":", lw=1.5, color="#303030", label="No-think answer (Assistant)")
        ax.axvline(1, color="#555555", lw=1)
        ax.set(title=domain.capitalize(), xlabel="Normalized position: thinking → answer", ylabel="Centered activation cosine", xlim=(0, 2))
        ax.set_xticks([0, .5, 1, 1.5, 2], ["0", ".5", "</think>", ".5", "1"])
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Thinking and answer trajectories · pointwise 95% prompt bootstrap intervals")
    return _save(fig, output_dir, "01_trajectory", synthetic)


def plot_transplants(tokens, output_dir, levels, synthetic=False):
    subset = levels.loc[levels.seed_scope.eq("pooled") & levels.domain.eq("advice") & levels.segment.isin(["think", "answer"]) &
                        levels.experiment.isin(["E1", "E2"]) & levels.role.eq("default")].copy()
    if subset.empty or not subset.experiment.eq("E2").any():
        return None
    order = ["natural_think:think", "natural_think:answer", "natural_nothink:answer", "nothink_stepbystep:answer",
             "cot_as_cot:think", "cot_as_answer:answer", "answer_as_answer:answer", "answer_as_cot:think", "cot_in_scratch:answer",
             "answer_in_scratch:answer", "cot_in_special:answer", "answer_in_special:answer"]
    subset["cell"] = subset.condition+":"+subset.segment
    order = [c for c in order if c in set(subset.cell)]
    directions = _controls(tokens)
    fig, axes = _panels(len(directions), width=3.5, height=6.5, columns=5, sharey=True, sharex=True)
    for ax, direction, color in zip(axes, directions, PALETTE):
        rows = subset.loc[subset.direction.eq(direction)].set_index("cell").reindex(order)
        y = np.arange(len(order))
        ax.barh(y, rows["mean"], color=color, alpha=.8, height=.65)
        lo, hi = pd.to_numeric(rows.ci_low), pd.to_numeric(rows.ci_high)
        valid = rows["mean"].notna() & lo.notna() & hi.notna()
        ax.errorbar(rows.loc[valid, "mean"], y[valid], xerr=[np.maximum(0, rows.loc[valid, "mean"]-lo[valid]),
                     np.maximum(0, hi[valid]-rows.loc[valid, "mean"])], fmt="none", ecolor="#333333", capsize=2)
        ax.axvline(0, color="#444444", lw=.8)
        ax.set(title=LABELS.get(direction, direction), xlabel="Mean cosine", yticks=y,
               yticklabels=[c.replace("_", " ").replace(":", " · ") for c in order])
        ax.set_ylim(len(order)-.5, -.5)
    fig.suptitle("Advice tag transplants · 95% prompt bootstrap intervals")
    return _save(fig, output_dir, "02_transplants", synthetic)


def plot_role_stiffness(tokens, output_dir, role_shifts, synthetic=False):
    if role_shifts.empty or not {"think", "answer"}.issubset(role_shifts):
        return None
    directions = _controls(tokens)
    directions = [d for d in directions if d in set(role_shifts.direction)]
    if not directions:
        return None
    fig, axes = _panels(len(directions), width=4, height=4, columns=3, sharex=True, sharey=True)
    finite = role_shifts[["think", "answer"]].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        plt.close(fig)
        return None
    low, high = min(0, finite.min()), max(0, finite.max())
    pad = max((high-low)*.08, .005)
    judge_points = None
    for ax, direction in zip(axes, directions):
        rows = role_shifts.loc[role_shifts.direction.eq(direction)].dropna(subset=["think", "answer"])
        judge = "meta_stance_rate" in rows and rows.meta_stance_rate.notna().any()
        if judge:
            rated = rows.loc[rows.meta_stance_rate.notna()]
            judge_points = ax.scatter(rated.answer, rated.think, c=rated.meta_stance_rate, cmap="cividis", vmin=0, vmax=1,
                                      s=35, edgecolors="#333333", linewidths=.4)
            unrated = rows.loc[rows.meta_stance_rate.isna()]
            if not unrated.empty:
                ax.scatter(unrated.answer, unrated.think, color="#aaaaaa", marker="x", label="Unjudged")
                ax.legend(fontsize=8)
        else:
            ax.scatter(rows.answer, rows.think, color=BLUE, s=35, alpha=.8)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], ":", color="#333333")
        ax.axhline(0, color="#777777", lw=.7)
        ax.axvline(0, color="#777777", lw=.7)
        ax.set(title=f"{LABELS.get(direction, direction)} · {len(rows)} roles", xlabel="Default − role answer cosine",
               ylabel="Default − role thinking cosine", xlim=(low-pad, high+pad), ylim=(low-pad, high+pad))
        if direction == "cos_assistant" and not rows.empty:
            distance = (rows.think-rows.answer).abs()
            for i in list(dict.fromkeys([*distance.nlargest(2).index, distance.idxmin()])):
                row = rows.loc[i]
                right = row.answer > (low + high) / 2
                ax.annotate(row.role, (row.answer, row.think), xytext=(-5 if right else 5, 6),
                            ha="right" if right else "left", textcoords="offset points", fontsize=8)
    if judge_points is not None:
        fig.colorbar(judge_points, ax=axes, label="CoT meta-stance fraction", shrink=.7)
    fig.suptitle("Role stiffness · paired extraction questions, one point per role")
    return _save(fig, output_dir, "03_role_stiffness", synthetic)


def plot_gap_histogram(tokens, output_dir, gaps, synthetic=False):
    data = gaps.loc[gaps.contrast.eq("think_minus_answer")]
    if data.empty:
        return None
    data = data.groupby(["domain", "prompt_id", "direction"], dropna=False).value.mean().reset_index()
    facets = sorted(data.domain.unique())
    fig, axes = _panels(len(facets), sharex=True, sharey=True)
    limits = (data.value.min(), data.value.max())
    span = max(limits[1]-limits[0], .01)
    bins = np.linspace(limits[0]-.02*span, limits[1]+.02*span, min(21, max(6, int(np.sqrt(len(data)))+1)))
    for ax, domain in zip(axes, facets):
        for direction, color in zip(_controls(tokens), PALETTE):
            values = data.loc[data.domain.eq(domain) & data.direction.eq(direction), "value"]
            ax.hist(values, bins=bins, histtype="step", color=color, lw=1.7, label=LABELS.get(direction, direction))
        ax.axvline(0, color="#555555", lw=.8)
        ax.set(title=domain.capitalize(), xlabel="Thinking − answer cosine per prompt", ylabel="Prompts")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gap distributions · seeds averaged within prompt")
    return _save(fig, output_dir, "05_gap_histogram", synthetic)


def plot_boundary(tokens, output_dir, bootstrap_samples=1000, seed=0, synthetic=False):
    if "pos_from_boundary" not in tokens:
        return None
    keys = ["run_id", "seed", "model", "prompt_id", "domain", "condition", "role"]
    data = tokens.loc[tokens.experiment.eq("E1") & tokens.condition.eq("natural_think")].copy()
    if data.empty:
        return None
    eligible = data.groupby(keys, dropna=False).primary_include.transform("any")
    data = data.loc[eligible & data.pos_from_boundary.between(-30, 30)]
    if data.empty:
        return None
    directions = _controls(tokens)
    curve = _curve(data, "pos_from_boundary", directions, bootstrap_samples, seed)
    facets = sorted(data.domain.unique())
    fig, axes = _panels(len(facets), sharey=True, sharex=True)
    for ax, domain in zip(axes, facets):
        for direction, color in zip(directions, PALETTE):
            _line(ax, curve.loc[curve.domain.eq(domain)], direction, color)
        ax.axvline(0, color="#333333", lw=1)
        first5 = data.loc[data.domain.eq(domain) & data.segment.eq("answer") &
                          data.segment_idx.between(0, 4), "pos_from_boundary"]
        if not first5.empty:
            ax.axvspan(first5.min()-.5, first5.max()+.5, color="#777777", alpha=.1)
        ax.set(title=domain.capitalize(), xlabel="Token position from </think>", ylabel="Centered activation cosine", xlim=(-30, 30))
    axes[0].legend(fontsize=8)
    fig.suptitle("Boundary detail · first-five content offset range shaded; 95% prompt intervals")
    return _save(fig, output_dir, "06_boundary_zoom", synthetic)


def plot_specificity(tokens, output_dir, levels, synthetic=False):
    data = levels.loc[levels.seed_scope.eq("pooled") & levels.experiment.eq("E1") & levels.condition.eq("natural_think") &
                      levels.role.eq("default") & levels.segment.isin(["think", "answer"])]
    if data.empty:
        return None
    directions = direction_columns(tokens)
    facets = sorted(data.domain.unique())
    fig, axes = _panels(len(facets), width=max(6, len(directions)*.75), height=4.2, sharey=True)
    for ax, domain in zip(axes, facets):
        for segment, offset, color in [("think", -.19, BLUE), ("answer", .19, GOLD)]:
            rows = data.loc[data.domain.eq(domain) & data.segment.eq(segment)].set_index("direction").reindex(directions)
            x = np.arange(len(directions))+offset
            ax.bar(x, rows["mean"], .36, label=segment.capitalize(), color=color)
            lo, hi = pd.to_numeric(rows.ci_low), pd.to_numeric(rows.ci_high)
            valid = rows["mean"].notna() & lo.notna() & hi.notna()
            ax.errorbar(x[valid], rows.loc[valid, "mean"], yerr=[np.maximum(0, rows.loc[valid, "mean"]-lo[valid]),
                        np.maximum(0, hi[valid]-rows.loc[valid, "mean"])], fmt="none", ecolor="#333333", capsize=2)
        ax.axhline(0, color="#444444", lw=.8)
        ax.set(title=domain.capitalize(), ylabel="Centered activation cosine", xticks=np.arange(len(directions)),
               xticklabels=[LABELS.get(d, d) for d in directions])
        ax.tick_params(axis="x", rotation=40)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    axes[0].legend()
    fig.suptitle("Direction specificity · 95% prompt intervals; compare paired gaps in specificity.csv")
    return _save(fig, output_dir, "10_direction_specificity", synthetic)


def plot_persona_space(output_dir, synthetic=False):
    path = Path(output_dir)/"persona_space.json"
    if not path.exists():
        return []
    report = json.loads(path.read_text())
    coordinates = pd.DataFrame(report.get("coordinates", []))
    outputs = []
    if not coordinates.empty and {"pc1", "pc2", "role", "segment"}.issubset(coordinates):
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        for segment, face in [("answer", BLUE), ("think", "none")]:
            rows = coordinates.loc[coordinates.segment.eq(segment)]
            ax.scatter(rows.pc1, rows.pc2, facecolors=face, edgecolors=BLUE, label=segment.capitalize(), s=45)
        paired = coordinates.pivot_table(index="role", columns="segment", values=["pc1", "pc2"])
        if all((pc, seg) in paired for pc in ["pc1", "pc2"] for seg in ["think", "answer"]):
            distance = ((paired[("pc1", "think")]-paired[("pc1", "answer")])**2+
                        (paired[("pc2", "think")]-paired[("pc2", "answer")])**2)
            for role in list(dict.fromkeys([*distance.nlargest(3).index, "default"])):
                rows = coordinates.loc[coordinates.role.eq(role)]
                for _, row in rows.iterrows():
                    label = f"Default ({row.segment})" if role == "default" else role
                    ax.annotate(label, (row.pc1, row.pc2), xytext=(4, 4), textcoords="offset points", fontsize=8)
                    if role == "default":
                        ax.scatter([row.pc1], [row.pc2], marker="*", s=140,
                                   facecolor=GOLD if row.segment == "answer" else "none", edgecolor="#333333")
        explained = report.get("answer_explained_variance_ratio", [])
        axis_label = lambda i: f"Answer-space PC{i+1}" + (f" ({explained[i]:.1%} variance)" if len(explained)>i else "")
        ax.set(xlabel=axis_label(0), ylabel=axis_label(1), title="Persona space · same answer-space PCA frame")
        ax.legend()
        outputs.append(_save(fig, output_dir, "08_persona_space", synthetic))
    alignment = np.asarray(report.get("pc_alignment", []), dtype=float)
    if "pc_alignment" in report:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
        if alignment.ndim == 2 and alignment.size and np.isfinite(alignment).any():
            cmap = plt.get_cmap("cividis").copy()
            cmap.set_bad("#eeeeee")
            im = ax.imshow(np.ma.masked_invalid(alignment), cmap=cmap, vmin=0, vmax=1)
            ax.set(xlabel="Thinking-space PC", ylabel="Answer-space PC", title="PC direction agreement · absolute cosine",
                   xticks=np.arange(alignment.shape[1]), xticklabels=np.arange(1, alignment.shape[1]+1),
                   yticks=np.arange(alignment.shape[0]), yticklabels=np.arange(1, alignment.shape[0]+1))
            for (i, j), value in np.ndenumerate(alignment):
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "n/a", ha="center", va="center", color="white" if value<.5 else "#222222")
            fig.colorbar(im, ax=ax, label="Absolute cosine")
        else:
            ax.text(.5, .5, "PC agreement unavailable\n"+report.get("think_pca_status", "Insufficient finite components"),
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
        outputs.append(_save(fig, output_dir, "09_pc_agreement", synthetic))
    return outputs


def plot_norm_diagnostic(output_dir, diagnostics, synthetic=False):
    if diagnostics.empty or not {"resid_norm", "dot_assistant"}.issubset(diagnostics):
        return None
    data = diagnostics.loc[diagnostics.experiment.eq("E1") & diagnostics.condition.eq("natural_think")]
    if data.empty:
        return None
    data = data.groupby(["prompt_id", "domain", "segment"])[["resid_norm", "dot_assistant"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for segment, color, marker in [("think", BLUE, "o"), ("answer", GOLD, "^")]:
        rows = data.loc[data.segment.eq(segment)]
        ax.scatter(rows.resid_norm, rows.dot_assistant, color=color, marker=marker, label=segment.capitalize(), alpha=.7, s=23)
    ax.set(xlabel="Mean residual activation norm", ylabel="Mean raw Assistant dot product", title="Norm diagnostic · one point per prompt and segment")
    ax.legend()
    return _save(fig, output_dir, "appendix_norm_diagnostic", synthetic)


def make_plots(tokens: pd.DataFrame, output_dir: Path, tables: dict | None = None,
               bootstrap_samples: int = 1000, seed: int = 0) -> list[str]:
    """Render all supported core figures; unavailable experiments are omitted."""
    _style()
    if tokens.empty:
        return []
    if tables is None:
        # The public entry point can also be called after analyze() using its CSVs.
        tables = {}
        for name in ["segment_levels", "e1_prompt_contrasts", "role_shifts", "norm_position_diagnostics"]:
            path = Path(output_dir)/f"{name}.csv"
            if path.exists():
                try:
                    tables[name] = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    tables[name] = pd.DataFrame()
    synthetic = bool(tokens.get("synthetic", pd.Series(False, index=tokens.index)).fillna(False).any()) or tokens.model.astype(str).str.contains("fixture|simulator", case=False).any()
    # Keep separate models separate: every headline panel must use one ruler.
    models = list(tokens.model.unique())
    if len(models) > 1:
        outputs = []
        for i, model in enumerate(models):
            model_dir = Path(output_dir)/f"model_{i}"
            model_dir.mkdir(exist_ok=True)
            model_tables = {k: v.loc[v.model.eq(model)] if "model" in v else v for k, v in tables.items()}
            outputs += [f"model_{i}/{p}" for p in make_plots(tokens.loc[tokens.model.eq(model)], model_dir, model_tables, bootstrap_samples, seed)]
        return outputs
    outputs = [plot_trajectory(tokens, output_dir, bootstrap_samples, seed, synthetic),
               plot_boundary(tokens, output_dir, bootstrap_samples, seed, synthetic)]
    if "segment_levels" in tables and not tables["segment_levels"].empty:
        outputs += [plot_transplants(tokens, output_dir, tables["segment_levels"], synthetic),
                    plot_specificity(tokens, output_dir, tables["segment_levels"], synthetic)]
    if "e1_prompt_contrasts" in tables and not tables["e1_prompt_contrasts"].empty:
        outputs.append(plot_gap_histogram(tokens, output_dir, tables["e1_prompt_contrasts"], synthetic))
    if "role_shifts" in tables:
        outputs.append(plot_role_stiffness(tokens, output_dir, tables["role_shifts"], synthetic))
    if "norm_position_diagnostics" in tables:
        outputs.append(plot_norm_diagnostic(output_dir, tables["norm_position_diagnostics"], synthetic))
    outputs += plot_persona_space(output_dir, synthetic)
    return sorted(p for p in outputs if p)
