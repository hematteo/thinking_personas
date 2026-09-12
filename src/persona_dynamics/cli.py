"""Command-line entry points for staging, ablations, and artifact analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Config, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run E0 then experiments and analysis, with process isolation")
    run.add_argument("--config", default="configs/smoke.yaml")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    worker = commands.add_parser("worker", help="Run one restartable stage from a frozen config")
    worker.add_argument("--config", required=True)
    worker.add_argument("--stage", choices=["generate-gate", "gate", "generate-experiments", "extract-e1", "extract", "analyze"], required=True)
    analyze = commands.add_parser("analyze", help="Rebuild tables/figures without model inference")
    analyze.add_argument("run_dir")
    aggregate = commands.add_parser("aggregate", help="Pool disjoint seeds from compatible runs; never duplicate samples")
    aggregate.add_argument("run_dirs", nargs="+")
    aggregate.add_argument("--output", required=True)
    assets = commands.add_parser("fetch-assets", help="Download pinned published axes and role prompts")
    assets.add_argument("--output", default="data/assets")
    assets.add_argument("--prepared", default="data/prepared")
    assets.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    from . import pipeline
    if args.command == "run":
        config = load_config(args.config, args.set)
        root = pipeline.run_pipeline(config)
        print(f"Complete: {root.resolve() / 'results.md'}")
    elif args.command == "worker":
        config = load_config(args.config)
        if args.stage.startswith("generate-"):
            pipeline.generate_phase(config, args.stage.removeprefix("generate-"))
        elif args.stage == "gate":
            pipeline.calibrate_and_gate(config)
        elif args.stage == "extract":
            pipeline.extract_experiments(config)
        elif args.stage == "extract-e1":
            pipeline.extract_experiments(config, include_transplants=False)
        else:
            pipeline.analyze_run(config)
    elif args.command == "analyze":
        if (Path(args.run_dir) / "aggregation.json").exists():
            _analyze_aggregate(Path(args.run_dir))
        else:
            config = Config(**json.loads((Path(args.run_dir) / "config.json").read_text())).validate()
            if Path(config.output_dir).resolve() != Path(args.run_dir).resolve():
                raise ValueError("Stored output_dir resolves to a different run. Analyze from the original working directory so the frozen run identity is preserved.")
            pipeline.analyze_run(config)
    elif args.command == "fetch-assets":
        from .assets import fetch_assets
        print(fetch_assets(args.output, args.prepared, args.seed))
    else:
        _aggregate(args.run_dirs, args.output)


def _aggregate(run_dirs, output):
    import pandas as pd
    import shutil
    from .io import file_hash, write_json
    from .pipeline import require_gate
    directories = [Path(d).resolve() for d in run_dirs]
    if not directories:
        raise ValueError("At least one source run is required")
    configs, manifests, frames, gates = [], [], [], []
    for directory in directories:
        require_gate(directory)
        gates.append(json.loads((directory / "e0.json").read_text()))
        config = json.loads((directory / "config.json").read_text())
        manifests.append(json.loads((directory / "manifest.json").read_text()))
        for key in ["output_dir", "name", "seeds"]:
            config.pop(key, None)
        configs.append(config)
        frames.append(pd.read_csv(directory / "tokens.csv.gz"))
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Cannot aggregate runs with different models, layers, filters, or generation settings")
    if any(m["inputs"] != manifests[0]["inputs"] or m["code"] != manifests[0]["code"] for m in manifests[1:]):
        raise ValueError("Cannot aggregate changed inputs or code")
    if any(not isinstance(m.get("numerical_versions"), dict) for m in manifests):
        raise ValueError("Cannot verify numerical runtime: source manifest lacks numerical_versions")
    if any(m["numerical_versions"] != manifests[0]["numerical_versions"] or m.get("python") != manifests[0].get("python")
           for m in manifests[1:]):
        raise ValueError("Cannot aggregate changed numerical runtime or Python versions")
    # Learned PC frames can differ by seed: pooled PCs require the same frame.
    direction_name = "directions_with_pcs.npz" if configs[0]["pca"] and "E4" in configs[0]["experiments"] else "directions.npz"
    ruler_hashes = [file_hash(d / direction_name) for d in directories]
    if len(set(ruler_hashes)) > 1:
        raise ValueError("Runs have different calibrated/PCA rulers. Run seeds together or disable E4 and share a frozen calibration.")
    tokens = pd.concat(frames, ignore_index=True)
    identity = ["model", "experiment", "domain", "prompt_id", "condition", "role"]
    key = ["seed", *identity, "token_idx"]
    if tokens.duplicated(key).any():
        raise ValueError("Overlapping seed/prompt/token observations; pooling would duplicate samples")
    retained = None
    seen_seeds = set()
    for frame in frames:
        current_seeds = set(frame.seed.unique())
        if seen_seeds & current_seeds:
            raise ValueError("Source runs must have disjoint seeds; overlapping seeds are not independent replications")
        seen_seeds.update(current_seeds)
        for _, seed_frame in frame.groupby("seed", dropna=False):
            if "primary_include" in seed_frame:
                if not seed_frame.primary_include.isin([True, False]).all():
                    raise ValueError("primary_include must contain boolean inclusion flags")
                seed_frame = seed_frame.loc[seed_frame.primary_include]
            cohort = set(seed_frame[identity].fillna("default").itertuples(index=False, name=None))
            if retained is not None and cohort != retained:
                raise ValueError("Retained transcript cohorts or role/question coverage differ across seeds; run seeds jointly with a shared cohort")
            retained = cohort
    if not retained:
        raise ValueError("Cannot aggregate an empty retained cohort")
    judge_paths = [d / "judge_labels.csv" for d in directories]
    judge = pd.concat([pd.read_csv(p) for p in judge_paths], ignore_index=True) if all(p.exists() for p in judge_paths) else None
    if judge is not None and "sentence_id" in judge and judge.sentence_id.duplicated().any():
        raise ValueError("Duplicate judge sentence IDs across source runs")
    sources = [{"path": str(d), "run_id": m["run_id"], "seeds": sorted(f.seed.unique().tolist()),
                "manifest": m} for d, m, f in zip(directories, manifests, frames)]
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Aggregate output must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"source_runs": [str(d) for d in directories], "sources": sources,
                  "unit": "prompt with seeds averaged within prompt; same retained transcript identities in every seed",
                  "analysis": {"bootstrap_samples": configs[0]["bootstrap_samples"], "seed": configs[0]["analysis_seed"]},
                  "config_without_run_identity": configs[0], "direction_file": direction_name, "direction_sha256": ruler_hashes[0],
                  "judge_status": "all source annotations pooled" if judge is not None else "unavailable: not every source has sentence labels",
                  "missing_judge_sources": [str(d) for d, p in zip(directories, judge_paths) if not p.exists()]}
    write_json(out / "e0.json", {"passed": True, "synthetic": any(m.get("synthetic", False) for m in manifests),
                                 "criterion": "Every source passed its frozen E0; no new pooled gate test is inferred",
                                 "source_gates": [{"source_run": str(d), "run_id": m["run_id"], "gate": gate}
                                                  for d, m, gate in zip(directories, manifests, gates)]})
    source_cohorts = [(d, json.loads((d / "cohort.json").read_text()) if (d / "cohort.json").exists() else []) for d in directories]
    pooled_cohorts = []
    retained_frame = pd.DataFrame(sorted(retained), columns=identity)
    for domain, group in retained_frame.loc[retained_frame.experiment.eq("E1")].groupby("domain"):
        records = [{"source_run": str(d), "cohort": row} for d, rows in source_cohorts for row in rows if row.get("domain") == domain]
        candidate_counts = {row["cohort"].get("candidates") for row in records}
        prompts = sorted(group.prompt_id.unique().tolist())
        pooled_cohorts.append({"domain": domain, "candidates": next(iter(candidate_counts)) if len(candidate_counts) == 1 else None,
                               "eligible_all_seeds": None, "selected": len(prompts), "selected_prompt_ids": prompts,
                               "requested": configs[0].get("target_per_domain"), "minimum": configs[0].get("min_per_domain"),
                               "note": "Retained cohort is identical; full candidate intersection cannot be inferred from survivor counts alone.",
                               "source_cohorts": records})
    write_json(out / "cohort.json", pooled_cohorts)
    for filename in ["attrition.csv", "role_retention.csv"]:
        if all((d / filename).exists() for d in directories):
            source_tables = []
            for d in directories:
                try:
                    source_tables.append(pd.read_csv(d / filename).assign(source_run=str(d)))
                except pd.errors.EmptyDataError:
                    pass  # E1-only runs write an empty role-retention artifact.
            if source_tables:
                pd.concat(source_tables, ignore_index=True).to_csv(out / filename, index=False)
    shutil.copyfile(directories[0] / direction_name, out / direction_name)
    persona_paths = [d / "persona_space.json" for d in directories]
    if all(p.exists() for p in persona_paths):
        reports = [json.loads(p.read_text()) for p in persona_paths]
        source_dir = out / "source_persona_reports"
        source_dir.mkdir()
        for i, report in enumerate(reports):
            write_json(source_dir / f"source_{i}.json", report)
        if all(report == reports[0] for report in reports[1:]):
            write_json(out / "persona_space.json", reports[0])
            provenance["persona_space_status"] = "Identical source persona report and shared measured directions preserved"
        else:
            # Directions alone do not establish identical affine PCA centers.
            # Only average scores when the complete stored PCA frame is shared.
            frame_paths = [d / "persona_frame.npz" for d in directories]
            if all(p.exists() for p in frame_paths) and len({file_hash(p) for p in frame_paths}) == 1:
                coordinates = []
                for source, report in zip(sources, reports):
                    frame = pd.DataFrame(report["coordinates"])
                    frame["seed_weight"] = len(source["seeds"])
                    coordinates.append(frame)
                coordinates = pd.concat(coordinates, ignore_index=True)
                pcs = [c for c in coordinates if c.startswith("pc")]
                weighted = coordinates.copy()
                weighted[pcs] = weighted[pcs].multiply(weighted.seed_weight, axis=0)
                sums = weighted.groupby(["model", "role", "segment"])[pcs+["seed_weight"]].sum()
                pooled = sums[pcs].divide(sums.seed_weight, axis=0).reset_index()
                report = {"coordinates": pooled.to_dict("records"),
                          "answer_explained_variance_ratio": reports[0].get("answer_explained_variance_ratio", []),
                          "pc_alignment": None, "think_pca_status": "Unavailable: pooled full-space thinking PCA was not refitted",
                          "frame_center": "Identical source answer PCA frame", "weighting": "Source seed-count weighted coordinates; no refit",
                          "aggregation_note": "Full-space cloud geometry and thinking-PC agreement cannot be averaged across source runs and are omitted."}
                write_json(out / "persona_space.json", report)
                shutil.copyfile(frame_paths[0], out / "persona_frame.npz")
                provenance["persona_space_status"] = report["aggregation_note"]
            else:
                provenance["persona_space_status"] = "Source reports retained; pooled E4 omitted because identical affine PCA frames were not established"
    elif any(p.exists() for p in persona_paths):
        provenance["persona_space_status"] = "Unavailable: not every source has a persona-space report"
    if judge is not None:
        judge.to_csv(out / "judge_labels.csv", index=False)
    review_paths = [d / "review" / "review.csv" for d in directories]
    if all(p.exists() for p in review_paths):
        (out / "review").mkdir()
        pd.concat([pd.read_csv(p).assign(source_run=str(d)) for d, p in zip(directories, review_paths)], ignore_index=True).to_csv(out / "review" / "review.csv", index=False)
    write_json(out / "aggregation.json", provenance)
    tokens.to_csv(out / "tokens.csv.gz", index=False)
    return _analyze_aggregate(out)


def _analyze_aggregate(directory):
    """Rebuild an aggregate's analysis from its saved settings and annotations."""
    import pandas as pd
    from .analysis import analyze
    from .pipeline import require_gate
    directory = Path(directory)
    require_gate(directory)
    metadata = json.loads((directory / "aggregation.json").read_text())
    judge_path = directory / "judge_labels.csv"
    judge = pd.read_csv(judge_path) if judge_path.exists() else None
    return analyze(pd.read_csv(directory / "tokens.csv.gz"), directory, judge=judge, **metadata["analysis"])
