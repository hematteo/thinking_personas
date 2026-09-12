"""Prompt/role-level, paired analysis of scalar token projections.

Tokens are never independent observations. Seed-specific estimates pair within a
seed and prompt; pooled estimates first average those paired effects over seeds
within each prompt. These CIs quantify prompt sampling, conditional on the seeds
run, and are not confidence intervals over the population of model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDENTITY = ["run_id", "seed", "model", "experiment", "prompt_id", "domain", "condition", "role"]
PAIR = ["seed", "model", "prompt_id", "domain"]
EXPECTED_E2 = ["natural_think", "natural_nothink", "nothink_stepbystep", "cot_as_cot", "cot_as_answer",
               "answer_as_cot", "cot_in_scratch", "answer_in_scratch", "cot_in_special",
               "answer_in_special", "answer_as_answer"]


def direction_columns(tokens: pd.DataFrame) -> list[str]:
    """Only measured cosine directions; unavailable PC columns stay unavailable."""
    return [c for c in tokens if c.startswith("cos_") and pd.to_numeric(tokens[c], errors="coerce").notna().any()]


def bootstrap_summary(values: Iterable[float], bootstrap_samples: int = 1000,
                      seed: int = 0) -> dict:
    """Mean and paired standardized effect dz, with percentile bootstrap CIs.

    Input must contain one paired difference per independent prompt/role for dz
    to mean a paired effect. Fewer than two units cannot identify uncertainty;
    zero-variance differences do not identify a finite standardized effect.
    """
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    x = np.asarray(list(values), dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Bootstrap observations must be finite; resolve missing measurements before analysis")
    result = dict(n=len(x), mean=None, ci_low=None, ci_high=None, sd=None,
                  dz=None, dz_ci_low=None, dz_ci_high=None)
    if not len(x):
        return result
    result["mean"] = float(x.mean())
    if len(x) < 2:
        return result
    sd = float(x.std(ddof=1))
    result["sd"] = sd
    if sd > np.finfo(float).eps:
        result["dz"] = result["mean"] / sd
    rng = np.random.default_rng(seed)
    means, effects = [], []
    for start in range(0, bootstrap_samples, 256):
        draws = x[rng.integers(0, len(x), size=(min(256, bootstrap_samples-start), len(x)))]
        draw_mean = draws.mean(axis=1)
        draw_sd = draws.std(axis=1, ddof=1)
        means.extend(draw_mean.tolist())
        effects.extend(np.divide(draw_mean, draw_sd, out=np.full(len(draw_sd), np.nan),
                                 where=draw_sd > np.finfo(float).eps).tolist())
    result["ci_low"], result["ci_high"] = map(float, np.quantile(means, [.025, .975]))
    finite_effects = np.asarray(effects)[np.isfinite(effects)]
    # Degenerate resamples are common with tiny n; do not imply a reliable dz CI.
    if result["dz"] is not None and len(finite_effects) >= .9 * bootstrap_samples:
        result["dz_ci_low"], result["dz_ci_high"] = map(float, np.quantile(finite_effects, [.025, .975]))
    return result


def _prepare(tokens: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "model", "prompt_id", "domain", "condition", "role", "experiment",
                "segment", "segment_idx", "cos_assistant", "primary_include"}
    if missing := required.difference(tokens.columns):
        raise ValueError(f"Token table lacks required columns: {sorted(missing)}")
    out = tokens.copy()
    measured = list(dict.fromkeys(["cos_assistant"] + direction_columns(out) +
                    [c for c in out if c.startswith("cos_") and not c.startswith("cos_pc")]))
    for column in measured:
        values = pd.to_numeric(out[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Measured direction {column} contains non-finite values")
        out[column] = values
    if {"record_id", "token_idx"}.issubset(out) and out.duplicated(["record_id", "token_idx"]).any():
        raise ValueError("Duplicate record/token observations would distort segment weighting")
    if out["primary_include"].isna().any() or not out["primary_include"].isin([True, False]).all():
        raise ValueError("primary_include must contain non-null booleans")
    out["primary_include"] = out["primary_include"].astype(bool)
    if "run_id" not in out:
        out["run_id"] = "run"
    out["role"] = out["role"].fillna("default")
    return out


def segment_means(tokens: pd.DataFrame) -> pd.DataFrame:
    """One row per transcript/segment/direction, with explicit diagnostic spans."""
    directions = direction_columns(tokens)
    spans = {
        "think": tokens.primary_include & tokens.segment.eq("think"),
        "answer": tokens.primary_include & tokens.segment.eq("answer"),
        "boundary": tokens.segment.eq("boundary"),
        "answer_first5": tokens.segment.eq("answer") & tokens.segment_idx.between(0, 4),
        "answer_all": tokens.segment.eq("answer"),
        "think_all": tokens.segment.eq("think"),
    }
    if "norm_pos" in tokens:
        spans["think_early"] = tokens.primary_include & tokens.segment.eq("think") & tokens.norm_pos.le(.25)
        spans["think_late"] = tokens.primary_include & tokens.segment.eq("think") & tokens.norm_pos.ge(.75)
    # Diagnostics are reported even for excluded transcripts, with eligibility.
    eligible = tokens.groupby(IDENTITY, dropna=False).primary_include.any().rename("eligible")
    rows = []
    for label, mask in spans.items():
        sub = tokens.loc[mask]
        if sub.empty:
            continue
        grouped = sub.groupby(IDENTITY, dropna=False)
        means = grouped[directions].mean().join(grouped.size().rename("n_tokens")).join(eligible).reset_index()
        means["segment"] = label
        rows.append(means.melt(id_vars=IDENTITY + ["segment", "n_tokens", "eligible"],
                               value_vars=directions, var_name="direction", value_name="value"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=IDENTITY+["segment", "n_tokens", "eligible", "direction", "value"])


def _paired(frame: pd.DataFrame, left_mask: pd.Series, right_mask: pd.Series,
            keys: list[str], name: str) -> pd.DataFrame:
    """Inner pairing prevents unmatched generations from changing the contrast."""
    left = frame.loc[left_mask].groupby(keys, dropna=False).value.mean().rename("left")
    right = frame.loc[right_mask].groupby(keys, dropna=False).value.mean().rename("right")
    paired = pd.concat([left, right], axis=1).dropna().reset_index()
    paired["value"] = paired.left - paired.right
    paired["contrast"] = name
    return paired


def _summarize(frame: pd.DataFrame, groups: list[str], unit: str,
               bootstrap_samples: int, seed: int) -> pd.DataFrame:
    columns = groups + ["seed_scope", "n", "mean", "ci_low", "ci_high", "sd", "dz", "dz_ci_low", "dz_ci_high"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    output = []
    # Averages repeated run IDs within seed; each seed gets equal weight per unit.
    grain = list(dict.fromkeys(groups + ["seed", unit]))
    by_seed = frame.groupby(grain, dropna=False).value.mean().reset_index()
    pooled = by_seed.groupby(list(dict.fromkeys(groups + [unit])), dropna=False).value.mean().reset_index()
    for scope, data, grouping in [("pooled", pooled, groups), ("per_seed", by_seed, groups + ["seed"])]:
        for group, sub in data.groupby(grouping, dropna=False, sort=True):
            group = group if isinstance(group, tuple) else (group,)
            row = dict(zip(grouping, group))
            row["seed_scope"] = "pooled" if scope == "pooled" else str(row.pop("seed"))
            row.update(bootstrap_summary(sub.value, bootstrap_samples, seed))
            output.append(row)
    return pd.DataFrame(output, columns=columns)


def e1_contrasts(means: pd.DataFrame) -> pd.DataFrame:
    frame = means.loc[means.experiment.eq("E1") & means.role.eq("default") & means.eligible]
    keys = PAIR + ["direction"]
    natural = frame.condition.eq("natural_think")
    parts = [
        _paired(frame, natural & frame.segment.eq("think"), natural & frame.segment.eq("answer"),
                keys, "think_minus_answer"),
        _paired(frame, natural & frame.segment.eq("think"), natural & frame.segment.eq("answer_all"),
                keys, "think_minus_answer_all_sensitivity"),
        _paired(frame, natural & frame.segment.eq("answer"),
                frame.condition.eq("natural_nothink") & frame.segment.eq("answer"),
                keys, "think_mode_answer_minus_nothink_answer"),
        _paired(frame, natural & frame.segment.eq("think_late"), natural & frame.segment.eq("think_early"),
                keys, "late_minus_early_think"),
    ]
    return pd.concat(parts, ignore_index=True)


def e2_contrasts(means: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    frame = means.loc[means.experiment.isin(["E1", "E2"]) & means.domain.eq("advice") & means.eligible]
    frame = frame.loc[frame.segment.isin(["think", "answer"])]
    frame = frame.copy()
    frame["cell"] = frame.condition + ":" + frame.segment
    matched_plain = "answer_as_answer:answer" in set(frame.cell)
    cot_think = "cot_as_cot:think" if "cot_as_cot:think" in set(frame.cell) else "natural_think:think"
    answer_plain = "answer_as_answer:answer" if matched_plain else "natural_think:answer"
    comparisons = {
        "cot_tag_think_minus_plain": (cot_think, "cot_as_answer:answer"),
        "answer_tag_think_minus_plain": ("answer_as_cot:think", answer_plain),
        "content_cot_minus_answer_in_think": (cot_think, "answer_as_cot:think"),
        "content_cot_minus_answer_in_plain": ("cot_as_answer:answer", answer_plain),
        "cot_scratch_minus_plain": ("cot_in_scratch:answer", "cot_as_answer:answer"),
        "answer_scratch_minus_plain": ("answer_in_scratch:answer", answer_plain),
        "cot_special_minus_plain": ("cot_in_special:answer", "cot_as_answer:answer"),
        "answer_special_minus_plain": ("answer_in_special:answer", answer_plain),
        "cot_think_minus_scratch": (cot_think, "cot_in_scratch:answer"),
        "answer_think_minus_scratch": ("answer_as_cot:think", "answer_in_scratch:answer"),
        "cot_think_minus_special": (cot_think, "cot_in_special:answer"),
        "answer_think_minus_special": ("answer_as_cot:think", "answer_in_special:answer"),
        "stepbystep_minus_nothink": ("nothink_stepbystep:answer", "natural_nothink:answer"),
    }
    keys = PAIR + ["direction"]
    parts = [_paired(frame, frame.cell.eq(a), frame.cell.eq(b), keys, name)
             for name, (a, b) in comparisons.items()]
    wide = frame.pivot_table(index=keys, columns="cell", values="value", aggfunc="mean")
    cells = [cot_think, "cot_as_answer:answer", "answer_as_cot:think", answer_plain]
    if all(c in wide for c in cells):
        complete = wide.dropna(subset=cells)
        ct, cp, at, ap = (complete[c] for c in cells)
        for name, difference in {
            "factorial_tag_main": ((ct-cp)+(at-ap))/2,
            "factorial_content_main": ((ct-at)+(cp-ap))/2,
            "factorial_tag_content_interaction": (ct-cp)-(at-ap),
        }.items():
            part = difference.rename("value").reset_index()
            part["contrast"] = name
            parts.append(part)
    return pd.concat(parts, ignore_index=True), matched_plain


def specificity_contrasts(gaps: pd.DataFrame) -> pd.DataFrame:
    """Paired |Assistant gap| - |control gap|, not independent group CIs."""
    keys = PAIR + ["contrast"]
    wide = gaps.pivot_table(index=keys, columns="direction", values="value", aggfunc="mean")
    parts = []
    if "cos_assistant" in wide:
        for control in wide:
            if control == "cos_assistant":
                continue
            paired = wide[["cos_assistant", control]].dropna()
            part = (paired.cos_assistant.abs()-paired[control].abs()).rename("value").reset_index()
            part["control"] = control
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keys+["value", "control"])


def e3_role_shifts(means: pd.DataFrame) -> pd.DataFrame:
    frame = means.loc[means.experiment.eq("E3") & means.segment.isin(["think", "answer"]) & means.eligible]
    keys = PAIR + ["segment", "direction"]
    defaults = frame.loc[frame.role.eq("default")].groupby(keys, dropna=False).value.mean().rename("default")
    roles = frame.loc[~frame.role.eq("default")].groupby(keys+["role"], dropna=False).value.mean().rename("role_value").reset_index()
    shifts = roles.merge(defaults.reset_index(), on=keys, how="inner")
    shifts["value"] = shifts.default - shifts.role_value
    return shifts


def role_regression(role_rows: pd.DataFrame, bootstrap_samples: int = 1000, seed: int = 0) -> dict:
    """Descriptive thinking shift ~ intercept + slope × answer shift, by role.

    Both axes are estimated quantities, so OLS can have errors-in-variables
    attenuation. This is not an equivalence test or a causal stiffness parameter.
    """
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    pairs = role_rows[["answer", "think"]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    result = {"n_roles": len(pairs), "slope": None, "intercept": None, "r_squared": None,
              "slope_ci_low": None, "slope_ci_high": None, "intercept_ci_low": None, "intercept_ci_high": None}
    def fit(values):
        x, y = values[:, 0], values[:, 1]
        denominator = ((x-x.mean())**2).sum()
        if denominator <= 1e-16:
            return None
        slope = ((x-x.mean())*(y-y.mean())).sum()/denominator
        return float(slope), float(y.mean()-slope*x.mean())
    if len(pairs) < 3 or (estimate := fit(pairs)) is None:
        return result
    result["slope"], result["intercept"] = estimate
    y = pairs[:, 1]
    total = ((y-y.mean())**2).sum()
    if total > 1e-16:
        residual = y-(result["intercept"]+result["slope"]*pairs[:, 0])
        result["r_squared"] = float(1-(residual**2).sum()/total)
    rng = np.random.default_rng(seed)
    draws = [fit(pairs[rng.integers(0, len(pairs), len(pairs))]) for _ in range(bootstrap_samples)]
    finite = [draw for draw in draws if draw is not None]
    if len(finite) >= .9*bootstrap_samples:
        bounds = np.quantile(finite, [.025, .975], axis=0)
        result.update(slope_ci_low=float(bounds[0, 0]), slope_ci_high=float(bounds[1, 0]),
                      intercept_ci_low=float(bounds[0, 1]), intercept_ci_high=float(bounds[1, 1]))
    return result


def _judge_tables(judge: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    if judge is None or judge.empty:
        return pd.DataFrame(), {"available": False, "limitation": "No sentence judge labels supplied."}
    frame = judge.copy()
    if "sentence_id" in frame and frame.sentence_id.duplicated().any():
        raise ValueError("Judge annotations contain duplicate sentence IDs")
    label_col = "label" if "label" in frame else "judge_label"
    if label_col not in frame:
        return pd.DataFrame(), {"available": False, "limitation": "Judge labels lack label/judge_label column."}
    normalized = frame[label_col].astype(str).str.lower().str.replace("_", " ")
    frame["meta_stance_rate"] = normalized.str.contains("meta")
    frame["first_person_rate"] = normalized.str.contains("first person|in.character", regex=True)
    frame["refusal_rate"] = normalized.str.contains("refusal|breaking|broken", regex=True)
    recognized = frame[["meta_stance_rate", "first_person_rate", "refusal_rate"]].any(axis=1) | normalized.eq("neutral")
    if not recognized.all():
        raise ValueError("Unknown judge labels: " + ", ".join(sorted(frame.loc[~recognized, label_col].astype(str).unique())))
    keys = [c for c in ["seed", "model", "prompt_id", "domain", "condition", "role", "segment"] if c in frame]
    if not keys:
        return pd.DataFrame(), {"available": False, "limitation": "Judge rows lack transcript identity."}
    rates = frame.groupby(keys, dropna=False)[["meta_stance_rate", "first_person_rate", "refusal_rate"]].mean().reset_index()
    info: dict = {"available": True, "n_sentences": len(frame), "n_transcripts": len(rates),
                  "refusal_policy": "Flagged in role table; unfiltered geometry retained as sensitivity analysis."}
    info["label_sources"] = sorted(frame.label_source.fillna("unspecified").astype(str).unique().tolist()) if "label_source" in frame else ["unspecified"]
    info["heuristic_only"] = all("heuristic" in source.lower() for source in info["label_sources"])
    if info["heuristic_only"]:
        info["source_caveat"] = "Unvalidated heuristic labels check plumbing only; they are not independent model judgments."
    if "human_label" in frame:
        hand = frame.dropna(subset=["human_label", label_col])
        if len(hand):
            a, b = hand.human_label.astype(str), hand[label_col].astype(str)
            agreement = float((a == b).mean())
            pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
            expected = sum(pa.get(k, 0)*pb.get(k, 0) for k in set(pa.index) | set(pb.index))
            info.update(n_hand_labeled=len(hand), agreement=agreement,
                        cohen_kappa=(agreement-expected)/(1-expected) if expected < 1 else None)
    if info.get("n_hand_labeled", 0) < 30:
        info["limitation"] = "Fewer than the required 30 human-labeled calibration sentences."
    return rates, info


def analyze(tokens: pd.DataFrame, output_dir: Path, bootstrap_samples: int = 1000,
            seed: int = 0, judge: pd.DataFrame | None = None) -> dict:
    """Write reproducible tables, reports and figures; never select H1/H2/H3 automatically."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = _prepare(tokens)
    means = segment_means(tokens)
    e1 = e1_contrasts(means)
    e2, matched_plain = e2_contrasts(means)
    specificity = specificity_contrasts(e1)
    role_shifts = e3_role_shifts(means)
    judge_rates, judge_info = _judge_tables(judge)
    groups = ["model", "domain", "direction", "contrast"]
    tables = {
        "segment_means": means,
        "e1_prompt_contrasts": e1,
        "e1_effects": _summarize(e1, groups, "prompt_id", bootstrap_samples, seed),
        "e2_prompt_contrasts": e2,
        "e2_effects": _summarize(e2, groups, "prompt_id", bootstrap_samples, seed),
        "specificity": _summarize(specificity, ["model", "domain", "contrast", "control"], "prompt_id", bootstrap_samples, seed),
        "role_prompt_shifts": role_shifts,
    }
    eligible_means = means.loc[means.eligible]
    tables["segment_levels"] = _summarize(eligible_means, ["model", "experiment", "domain", "condition", "role", "segment", "direction"],
                                          "prompt_id", bootstrap_samples, seed)
    role_keys = ["model", "domain", "role", "segment", "direction"]
    if not role_shifts.empty:
        # Equal question weight within seed, equal seed weight within role.
        by_role_seed = role_shifts.groupby(["seed"]+role_keys, dropna=False).value.mean().reset_index()
        by_role = by_role_seed.groupby(role_keys, dropna=False).value.mean().reset_index()
        role_wide = by_role.pivot_table(index=["model", "domain", "role", "direction"], columns="segment", values="value").reset_index()
        tables["role_shifts"] = role_wide
        role_effects = _summarize(by_role_seed, ["model", "domain", "segment", "direction"], "role", bootstrap_samples, seed)
        tables["role_effects"] = role_effects
        seed_wide = by_role_seed.pivot_table(index=["seed", "model", "domain", "role", "direction"], columns="segment", values="value").reset_index()
        if {"think", "answer"}.issubset(seed_wide):
            difference = seed_wide.dropna(subset=["think", "answer"]).copy()
            difference["value"] = difference.think-difference.answer
            tables["role_stiffness_effects"] = _summarize(difference, ["model", "domain", "direction"], "role", bootstrap_samples, seed)
            regression_rows = []
            for (model, domain, direction), sub in role_wide.groupby(["model", "domain", "direction"]):
                regression_rows.append(dict(model=model, domain=domain, direction=direction,
                                            **role_regression(sub, bootstrap_samples, seed)))
            tables["role_regression"] = pd.DataFrame(regression_rows)
        if not judge_rates.empty and "role" in judge_rates:
            judge_keys = [c for c in ["model", "domain", "role"] if c in judge_rates]
            rates = judge_rates
            if "segment" in rates:
                rates = rates.loc[rates.segment.eq("think")]
            rate_means = rates.groupby(judge_keys, dropna=False)[["meta_stance_rate", "first_person_rate"]].mean().reset_index()
            all_segment_refusals = judge_rates.groupby(judge_keys, dropna=False).refusal_rate.agg(["mean", "max"]).rename(
                columns={"mean": "refusal_rate", "max": "refusal_flag"}).reset_index()
            all_segment_refusals["refusal_flag"] = all_segment_refusals.refusal_flag.gt(0)
            rate_means = rate_means.merge(all_segment_refusals, on=judge_keys, how="outer")
            tables["role_shifts"] = tables["role_shifts"].merge(rate_means, on=judge_keys, how="left")
    else:
        tables["role_shifts"] = pd.DataFrame(columns=["model", "domain", "role", "direction", "think", "answer"])
    tables.setdefault("role_effects", pd.DataFrame(columns=["model", "domain", "segment", "direction", "seed_scope", "n", "mean", "ci_low", "ci_high"]))
    tables.setdefault("role_regression", pd.DataFrame(columns=["model", "domain", "direction", "n_roles", "slope", "intercept", "slope_ci_low", "slope_ci_high"]))
    tables["judge_rates"] = judge_rates
    if not judge_rates.empty and {"prompt_id", "domain", "role"}.issubset(judge_rates):
        baseline = judge_rates.loc[judge_rates.role.eq("default")].copy()
        if "seed" not in baseline:
            baseline["seed"] = 0
        baseline_groups = [c for c in ["model", "domain", "condition", "segment"] if c in baseline]
        long = baseline.melt(id_vars=list(dict.fromkeys(baseline_groups+["seed", "prompt_id"])),
                             value_vars=["meta_stance_rate", "first_person_rate", "refusal_rate"],
                             var_name="measure", value_name="value")
        tables["judge_baseline"] = _summarize(long, baseline_groups+["measure"], "prompt_id", bootstrap_samples, seed)
        # Descriptive links between black-box labels and geometry; no causal or
        # incremental-validity conclusion is inferred from these correlations.
        associations = []
        if {"meta_stance_rate", "think", "answer"}.issubset(tables["role_shifts"]):
            for (model, domain, direction), sub in tables["role_shifts"].groupby(["model", "domain", "direction"]):
                for segment in ["think", "answer"]:
                    paired = sub[[segment, "meta_stance_rate"]].dropna()
                    correlation = paired[segment].corr(paired.meta_stance_rate) if len(paired) >= 3 and paired.nunique().min() > 1 else np.nan
                    associations.append(dict(model=model, domain=domain, direction=direction, segment=segment,
                                             n_roles=len(paired), pearson_r=correlation))
        tables["judge_geometry_associations"] = pd.DataFrame(associations)
    # The boundary and first five tokens get a dedicated auditable table, together
    # with all-answer sensitivity estimates instead of data-dependent exclusion.
    tables["boundary_first5"] = tables["segment_levels"].loc[
        tables["segment_levels"].segment.isin(["boundary", "answer_first5", "answer_all", "answer"])]
    survival = tokens.loc[tokens.experiment.eq("E1") & tokens.condition.eq("natural_think")].groupby(
        ["seed", "model", "domain", "prompt_id"], dropna=False).agg(eligible=("primary_include", "any"),
                                                                  n_think=("is_think", "sum")) if "is_think" in tokens else pd.DataFrame()
    tables["survival"] = survival.reset_index()
    diagnostic_cols = [c for c in ["token_idx", "resid_norm", "centered_norm", "dot_assistant", "cos_assistant"] if c in tokens]
    if diagnostic_cols:
        tables["norm_position_diagnostics"] = tokens.loc[tokens.primary_include].groupby(IDENTITY+["segment"], dropna=False)[diagnostic_cols].mean().reset_index()
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    e0_path = output_dir / "e0.json"
    e0 = json.loads(e0_path.read_text()) if e0_path.exists() else {"status": "not supplied"}
    synthetic = bool(tokens.get("synthetic", pd.Series(False, index=tokens.index)).fillna(False).any()) or tokens.model.astype(str).str.contains("fixture|simulator", case=False).any()
    present_e2 = set(tokens.loc[tokens.experiment.isin(["E1", "E2"]) & tokens.domain.eq("advice"), "condition"])
    summary = {
        "synthetic": bool(synthetic), "n_token_rows": len(tokens),
        "experiments_present": sorted(tokens.experiment.unique().tolist()),
        "n_prompts": int(tokens[["model", "domain", "prompt_id"]].drop_duplicates().shape[0]),
        "seeds": sorted(tokens.seed.unique().tolist()), "directions": direction_columns(tokens),
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "ci_method": "95% percentile paired bootstrap over prompts; seeds averaged within prompt; E3 CIs over roles.",
        "standardized_effect": "dz = mean paired difference / sample SD of paired differences; undefined for zero variance or n < 2.",
        "specificity_estimand": "Mean across prompts of |Assistant paired gap| minus |control paired gap|; read alongside signed group gaps.",
        "e0": e0, "judge": judge_info, "e2_matched_answer_plain": matched_plain,
        "missing_e2_conditions": [c for c in EXPECTED_E2 if c not in present_e2],
        "decision": "Descriptive only. No automated assignment of H1, H2 or H3.",
        "limitations": [
            "Cosine near zero and a nonsignificant role shift do not establish persona neutrality or equivalence.",
            "A persona interpretation requires gaps exceeding structured role/null/PC controls; random directions alone are insufficient.",
            "E2 must be evaluated jointly for tag, content, scratch and special-token controls; content tracking argues against a tag-keyed persona state.",
            "Transplants alter prefixes and may alter absolute position; they are not isolated interventions on a latent persona.",
            "Paired dz depends on prompt heterogeneity; raw cosine gaps and structured-control comparisons are primary.",
            "Role-regression CIs bootstrap whole roles. OLS uses two noisy estimated shifts and can suffer errors-in-variables attenuation; it is descriptive, not a causal or equivalence test.",
            "Boundary/first-five means are separate diagnostics. Primary inclusion comes from the configured rule, not an analysis-time outlier decision.",
            "Matched E2 transplants include every source-content token in each cell; natural E1/E3 baselines retain the configured first-five exclusion. E2 uses cot_as_cot and answer_as_answer when available.",
            "Per-seed tables assess seed stability; pooled CIs condition on the observed seeds and are pointwise, not simultaneous.",
            "Manual inspection of 20 CoTs per measured E1 domain remains required; 20 role CoTs are additionally required when E3 is run.",
        ],
    }
    cohort_path = output_dir / "cohort.json"
    if cohort_path.exists():
        summary["cohort"] = json.loads(cohort_path.read_text())
    attrition_path = output_dir / "attrition.csv"
    if attrition_path.exists():
        attrition = pd.read_csv(attrition_path)
        if {"domain", "condition", "exclusion_reason"}.issubset(attrition):
            summary["attrition"] = attrition.groupby(["domain", "condition", "exclusion_reason"], dropna=False).size().rename("n").reset_index().to_dict("records")
    persona_path = output_dir / "persona_space.json"
    if persona_path.exists():
        persona = json.loads(persona_path.read_text())
        summary["persona_space"] = {k: persona[k] for k in ["cloud_geometry", "think_pca_status", "answer_rank", "think_rank", "frame_center", "weighting"] if k in persona}
    if not matched_plain:
        summary["limitations"].append("The answer/plain factorial cell uses the natural answer following its CoT; prefix and position confound the tag contrast.")
    if synthetic:
        summary["limitations"].insert(0, "SYNTHETIC PIPELINE VALIDATION ONLY: these values cannot test the research hypotheses.")
    review_path = output_dir / "review" / "review.csv"
    domains = sorted(tokens.loc[tokens.experiment.eq("E1"), "domain"].unique().tolist())
    required_roles = 20 if tokens.experiment.eq("E3").any() else 0
    review_info = {"complete": False, "reviewed_per_domain": {}, "reviewed_roles": 0,
                   "required_per_domain": 20, "required_domains": domains, "required_roles": required_roles}
    if review_path.exists():
        reviewed = pd.read_csv(review_path).fillna("")
        if {"reviewed", "role", "domain", "prompt_id"}.issubset(reviewed):
            reviewed = reviewed.loc[reviewed.reviewed.astype(str).str.strip().str.lower().isin(["true", "yes", "1", "done"])]
            # Count only prompts/roles that actually have measured thinking data.
            known = tokens.loc[tokens.primary_include & tokens.segment.eq("think") &
                               tokens.experiment.isin(["E1", "E3"]), ["domain", "prompt_id", "role"]].drop_duplicates()
            reviewed = reviewed.merge(known, on=["domain", "prompt_id", "role"], how="inner")
            review_info["reviewed_per_domain"] = reviewed.loc[reviewed.role.eq("default")].groupby("domain").prompt_id.nunique().to_dict()
            review_info["reviewed_roles"] = int(reviewed.loc[~reviewed.role.eq("default")].role.nunique())
            review_info["complete"] = bool((len(domains) or required_roles) and all(review_info["reviewed_per_domain"].get(d, 0) >= 20 for d in domains)
                                                   and review_info["reviewed_roles"] >= required_roles)
    summary["manual_review"] = review_info
    summary["reporting_readiness"] = {"e0_passed": bool(e0.get("passed", False)), "human_review_complete": review_info["complete"],
                                      "judge_available": judge_info["available"], "human_judge_calibration_complete": judge_info.get("n_hand_labeled", 0) >= 30,
                                      "synthetic_only": bool(synthetic)}
    from .plots import make_plots
    summary["figures"] = make_plots(tokens, output_dir, tables=tables, bootstrap_samples=bootstrap_samples, seed=seed)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    _write_report(summary, tables, output_dir)
    return summary


def _write_report(summary: dict, tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    lines = ["# Persona geometry experiment results", ""]
    if summary["synthetic"]:
        lines += ["**SYNTHETIC PIPELINE VALIDATION — not evidence about model personas.**", ""]
    lines += [summary["decision"], "", f"Token rows: {summary['n_token_rows']:,}; prompts: {summary['n_prompts']}; seeds: {summary['seeds']}.",
              "", summary["ci_method"], "", "## Paired thinking − answer cosine gaps", "",
              "| Model | Domain | Direction | Prompts | Gap | 95% CI | Paired dz |", "|---|---|---|---:|---:|---|---:|"]
    effects = tables["e1_effects"]
    for _, row in effects.loc[effects.seed_scope.eq("pooled") & effects.contrast.eq("think_minus_answer")].iterrows():
        fmt = lambda v: "unavailable" if pd.isna(v) else f"{v:.4f}"
        lines.append(f"| {row.model} | {row.domain} | {row.direction} | {row.n} | {fmt(row['mean'])} | [{fmt(row.ci_low)}, {fmt(row.ci_high)}] | {fmt(row.dz)} |")
    lines += ["", "All directions and paired specificity contrasts are in `specificity.csv`; raw and standardized intervals are in `e1_effects.csv`.",
              "Boundary, first-five answer tokens, configured main answers and all-answer sensitivity estimates are separately tabulated in `boundary_first5.csv`.",
              "", "## Mechanism and role tests", "",
              "`e2_effects.csv` contains within-prompt tag, content, tag×content, scratch and special-block contrasts. Missing conditions: " +
              (", ".join(summary["missing_e2_conditions"]) or "none") + ".", ""]
    if not tables["role_shifts"].empty:
        lines += ["`role_shifts.csv` pairs each role with default on the same extraction question and averages questions/seeds within role; `role_effects.csv` uses roles as the sampling units.",
                  "`role_regression.csv` reports descriptive thinking-shift versus answer-shift regressions with role-bootstrap intervals.", ""]
    else:
        lines += ["E3 role intervention was not measured; empty role tables provide no evidence about role stiffness.", ""]
    if "persona_space" not in summary:
        lines += ["E4 persona-space/PCA analysis was not measured.", ""]
    lines += ["", "## Calibration and interpretation", "", "E0 status: `"+json.dumps(summary["e0"])+"`.", "",
              "Sentence judge: `"+json.dumps(summary["judge"])+"`.", ""]
    if "cohort" in summary:
        lines += ["Selected cohorts (candidate counts and survivors before selection):", "", "| Domain | Candidates | Eligible all seeds | Selected | Target |",
                  "|---|---:|---:|---:|---:|"]
        for row in summary["cohort"]:
            lines.append(f"| {row.get('domain')} | {row.get('candidates')} | {row.get('eligible_all_seeds')} | {row.get('selected')} | {row.get('requested')} |")
        lines += ["", "Condition-level exclusions and reasons are retained in `attrition.csv`; `survival.csv` describes the extracted token table only.", ""]
    if "judge_baseline" in tables:
        lines += ["The independent sentence-label baseline is in `judge_baseline.csv` (prompt bootstrap); `judge_geometry_associations.csv` reports descriptive role-level correlations. These correlations do not establish incremental validity or causation.", ""]
    lines += ["Human review status: `"+json.dumps(summary["manual_review"])+"`.", ""]
    if "persona_space" in summary:
        persona = summary["persona_space"]
        lines += ["## Persona-space diagnostics", "", "Full-space cloud metrics are descriptive and use the same roles that fit the PCA; they have no bootstrap intervals.", "",
                  "Thinking-space PCA status: "+str(persona.get("think_pca_status", "not recorded"))+".", "",
                  "| Metric | Value |", "|---|---:|"]
        for name, value in persona.get("cloud_geometry", {}).items():
            if isinstance(value, (int, float)) or value is None:
                lines.append(f"| {name.replace('_', ' ')} | {'unavailable' if value is None else format(value, '.5g')} |")
        lines.append("")
    lines += ["- "+item for item in summary["limitations"]]
    lines += ["", "## Figures", ""]
    for figure in summary["figures"]:
        lines += [f"![{Path(figure).stem}]({figure})", ""]
    notes = output_dir / "human_notes.md"
    if not notes.exists():
        notes.write_text("# Human interpretation notes\n\nRecord reviewed prompt IDs, anomalous generations, bimodal gap examples, role refusals, and changes from the frozen protocol here. This file is preserved when analysis is regenerated. Mark completed transcript reviews in review/review.csv.\n")
    lines += ["## Human review notes", "", "Record interpretation in [human_notes.md](human_notes.md), which analysis regeneration preserves; mark completed full-transcript reviews in `review/review.csv`.", "",
              "## Next steps", "", "Multi-turn drift; thinking-only steering/capping; emergent-misalignment organisms; unfaithfulness-gap correlation; second model family.", ""]
    (output_dir / "results.md").write_text("\n".join(lines))
