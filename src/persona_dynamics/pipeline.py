"""Staged, restartable experiments. E0 is a hard gate before E1–E4 generation."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import fixture_activations, generate_batches, model_for, tokenizer_for
from .config import Config
from .io import digest, file_hash, read_jsonl, stable_seed, write_json, write_jsonl


def _paths(config):
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _inputs(config):
    from .data import load_prompts, load_roles, offline_fixture_prompts, offline_fixture_roles
    if config.backend == "fixture":
        prompts = offline_fixture_prompts(seed=config.analysis_seed, per_domain=config.target_per_domain + 2)
        roles = offline_fixture_roles(count=max(config.role_count, 10))
    else:
        prompts, roles = load_prompts(config.prompts_path), load_roles(config.roles_path)
    prompts = [p for p in prompts if p["domain"] in config.domains]
    return prompts, roles


def initialize(config):
    root = _paths(config)
    prompts, roles = _inputs(config)
    inputs = {"prompts": digest(prompts), "roles": digest(roles)}
    if config.backend != "fixture":
        for p in [config.axis_path, config.default_vector_path, *config.control_paths]:
            inputs[p] = file_hash(p)
    code = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    numerical_versions = {}
    for name in ["numpy", "torch", "transformers", "vllm"]:
        try:
            numerical_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            numerical_versions[name] = None
    identity = {"config": config.to_dict(), "inputs": inputs, "code": code, "numerical_versions": numerical_versions}
    run_id = digest(identity)[:16]
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["run_id"] != run_id:
            raise ValueError(f"Run directory {root} has different config, inputs, or code. Use a new output_dir.")
    else:
        versions = {}
        for name in ["numpy", "pandas", "matplotlib", "torch", "transformers", "vllm", "datasets", "persona-dynamics"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        manifest = {**identity, "run_id": run_id, "synthetic": config.backend == "fixture",
                    "python": sys.version, "platform": platform.platform(), "versions": versions,
                    "layer_convention": "zero-indexed decoder block output, post residual, pre final norm",
                    "warning": "SYNTHETIC PIPELINE VERIFICATION; NOT SCIENTIFIC EVIDENCE" if config.backend == "fixture" else None}
        write_json(manifest_path, manifest)
        write_json(root / "config.json", config.to_dict())
        write_jsonl(root / "prompts.jsonl", prompts)
        write_jsonl(root / "roles.jsonl", roles)
    return root, prompts, roles, run_id


def _role_name(role):
    return role.get("role", role.get("name"))


def _request(config, tokenizer, prompt, condition, seed, role=None, experiment="E1"):
    role_name = _role_name(role) if role else "default"
    text = prompt["text"]
    if condition == "nothink_stepbystep":
        text += "\nReason step by step before answering."
    system = role["system_prompt"] if role else "You are an AI Assistant."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    enable_thinking = condition in {"natural_think", "role_prompt"}
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=enable_thinking, return_dict=False)
    if len(prefix) + config.max_tokens > config.max_context:
        raise ValueError(f"Prompt {prompt['prompt_id']} exceeds fixed context budget; never silently truncate")
    r = {"prompt_id": prompt["prompt_id"], "domain": prompt["domain"], "text": text,
         "source": prompt.get("source"), "messages": messages, "system_prompt": system,
         "condition": condition, "seed": seed, "role": role_name, "model": config.model,
         "experiment": experiment, "enable_thinking": enable_thinking, "prefix_ids": list(prefix),
         "role_score": float(role.get("precomputed_score", 0)) if role else 1.0,
         "synthetic": config.backend == "fixture"}
    r["generation_seed"] = stable_seed(seed, prompt["prompt_id"], condition, role_name)
    r["record_id"] = digest([seed, prompt["prompt_id"], condition, role_name, experiment])[:24]
    return r


def _partition(prompts, name):
    # dataset split (e.g. test) is separate from experimental partition.
    return [p for p in prompts if p.get("partition", p.get("split")) in ({"e0", "E0"} if name == "e0" else {name})]


def make_requests(config, phase):
    root, prompts, roles, run_id = initialize(config)
    tokenizer = tokenizer_for(config)
    from .transcripts import token_contract
    write_json(root / "tokenizer_contract.json", token_contract(tokenizer))
    requests = []
    if phase == "gate":
        calibration = _partition(prompts, "calibration")[:config.calibration_count]
        e0_prompts = _partition(prompts, "e0")[:config.e0_question_count]
        if len(calibration) < config.calibration_count or len(e0_prompts) < config.e0_question_count:
            raise ValueError("Not enough disjoint calibration/E0 prompts; prepare more data")
        available = {_role_name(r): r for r in roles}
        e0_roles = [available[r] for r in config.e0_roles if r in available]
        if len(e0_roles) < 6:
            if config.backend == "fixture":
                e0_roles = roles[:10]
            else:
                raise ValueError("E0 requires at least six of the frozen validation roles in roles.jsonl")
        for p in calibration:
            requests.append(_request(config, tokenizer, p, "natural_nothink", config.control_seed, experiment="calibration"))
        for seed in config.seeds:
            for p in e0_prompts:
                for role in [None, *e0_roles]:
                    requests.append(_request(config, tokenizer, p, "natural_nothink", seed, role, "E0"))
    else:
        require_gate(root)
        if any(e in config.experiments for e in ["E1", "E2"]):
            eval_prompts = _partition(prompts, "eval")
            if not eval_prompts:
                raise ValueError("No eval partition in input prompts")
            missing_domains = set(config.domains) - {p["domain"] for p in eval_prompts}
            if missing_domains:
                raise ValueError(f"Requested domains lack evaluation prompts: {sorted(missing_domains)}")
            eval_prompts.sort(key=lambda p: (p["domain"] != "advice", p["domain"], p.get("candidate_rank", 0), p["prompt_id"]))
            counts = {}
            limited = []
            for p in eval_prompts:
                counts[p["domain"]] = counts.get(p["domain"], 0) + 1
                if counts[p["domain"]] <= config.candidate_limit_per_domain:
                    limited.append(p)
            eval_prompts = limited
            for seed in config.seeds:
                for p in eval_prompts:
                    for condition in ["natural_think", "natural_nothink"]:
                        requests.append(_request(config, tokenizer, p, condition, seed))
                    if p["domain"] == "advice" and "E2" in config.experiments and config.e2_stepbystep:
                        requests.append(_request(config, tokenizer, p, "nothink_stepbystep", seed, experiment="E2"))
        if "E3" in config.experiments:
            from .data import select_roles
            if config.role_count % 3:
                raise ValueError("role_count must be divisible by three for balanced axis strata")
            selected = select_roles(roles, per_stratum=config.role_count // 3, seed=config.analysis_seed)
            write_jsonl(root / "selected_roles.jsonl", selected)
            questions = selected[0]["questions"][:config.role_question_count]
            if len(questions) < config.role_question_count:
                raise ValueError("Role data has too few extraction questions")
            for seed in config.seeds:
                for q in questions:
                    p = {**q, "domain": "role_extraction", "prompt_id": q.get("prompt_id", q.get("id")),
                         "text": q.get("text", q.get("question"))}
                    for role in [None, *selected]:
                        requests.append(_request(config, tokenizer, p, "role_prompt", seed, role, "E3"))
    for r in requests:
        r["run_id"] = run_id
    write_jsonl(root / f"requests_{phase}.jsonl", requests)
    return requests


def generate_phase(config, phase):
    from .transcripts import build_transcript
    root = _paths(config)
    requests = make_requests(config, phase)
    tokenizer = tokenizer_for(config)
    cache = root / "transcripts"
    cache.mkdir(exist_ok=True)
    pending = [r for r in requests if not (cache / f"{r['record_id']}.json").exists()]
    for r, completion, finish_reason in generate_batches(config, pending, tokenizer):
        record = build_transcript(tokenizer, r, r["condition"], completion, r["prefix_ids"],
                                  r["seed"], r["role"], r["experiment"])
        record.update({k: r[k] for k in ["record_id", "run_id", "generation_seed", "synthetic", "role_score", "messages"]})
        record.update({"finish_reason": finish_reason, "generated_ids": completion,
                       "prefix_ids": r["prefix_ids"], "text": tokenizer.decode(completion, skip_special_tokens=False)})
        write_json(cache / f"{r['record_id']}.json", record)
    print(f"{phase}: generated {len(pending)}, reused {len(requests) - len(pending)} transcripts", flush=True)


def phase_records(config, phase):
    root = _paths(config)
    requests = read_jsonl(root / f"requests_{phase}.jsonl")
    return [json.loads((root / "transcripts" / f"{r['record_id']}.json").read_text()) for r in requests]


def _vector(path, layer):
    import torch
    tensor = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or layer >= len(tensor):
        raise ValueError(f"Expected upstream [layers, hidden] tensor in {path}")
    return tensor[layer].float().numpy()


def _initial_bundle(config):
    from .geometry import DirectionBundle
    if config.backend == "fixture":
        d = config.synthetic_dimension
        axis, ctrl1, ctrl2 = np.eye(d)[:3]
        center = np.zeros(d); center[-1] = 10
    else:
        axis = _vector(config.axis_path, config.layer)
        center = _vector(config.default_vector_path, config.layer)
        ctrl1, ctrl2 = [_vector(p, config.layer) - center for p in config.control_paths[:2]]
    axis = axis / np.linalg.norm(axis)
    controls = []
    for v in [ctrl1, ctrl2]:
        v = v - np.dot(v, axis) * axis
        if np.linalg.norm(v) < 1e-8:
            raise ValueError("Control direction collinear with assistant axis")
        controls.append(v / np.linalg.norm(v))
    random = np.random.default_rng(config.control_seed).normal(size=len(axis))
    return DirectionBundle(center=center, directions={"assistant": axis, "ctrl_1": controls[0],
        "ctrl_2": controls[1], "random": random}, metadata={"model": config.model, "layer": config.layer,
        "control_seed": config.control_seed, "centering": "pending independent calibration"})


def _measure(config, record, bundle, model, keep_means=False):
    from .geometry import extract_record
    from .transcripts import annotate_transcript
    if config.backend == "fixture":
        hidden = fixture_activations(record, config.synthetic_dimension)
        scalars = bundle.project(hidden)
        rows = annotate_transcript(record)
        for row in rows:
            i = row["token_idx"]
            row.update({key: float(value[i]) for key, value in scalars.items()})
        means = {}
        for segment in ["think", "answer"]:
            idx = [r["token_idx"] for r in rows if r["segment"] == segment and
                   (segment != "answer" or not config.exclude_first_answer_tokens or r["segment_idx"] >= config.first_answer_tokens)]
            if idx:
                means[segment] = hidden[idx].mean(axis=0)
    else:
        rows, means = extract_record(model, record, bundle, config.layer, keep_segment_means=keep_means,
                                    answer_skip=config.first_answer_tokens if config.exclude_first_answer_tokens else 0)
    for row in rows:
        row.update({k: record.get(k) for k in ["run_id", "record_id", "seed", "prompt_id", "domain", "condition", "role", "model", "experiment", "synthetic"]})
        row["primary_include"] = row["segment"] in {"think", "answer"} and (
            bool(record.get("source_record_id")) or row["segment"] != "answer" or
            not config.exclude_first_answer_tokens or row["segment_idx"] >= config.first_answer_tokens)
    return rows, means


def _valid(record, min_think=0, answer_skip=5):
    spans = record.get("spans", [])
    reason = record.get("failure_reason")
    # A token-budget stop can also leave an unclosed thinking block. Report the
    # causal termination reason first; parser diagnostics remain on the record.
    if record.get("finish_reason") == "length":
        return False, "truncated"
    if not record.get("valid", True):
        return False, reason or "malformed"
    counts = {s: sum(x["end"] - x["start"] for x in spans if x["segment"] == s) for s in ["think", "answer"]}
    if counts["answer"] <= answer_skip:
        return False, "missing_or_short_answer"
    if min_think and counts["think"] < min_think:
        return False, "short_think"
    return True, "eligible"


def calibrate_and_gate(config):
    root, _, _, _ = initialize(config)
    if (root / "e0.json").exists():
        require_gate(root)
        print("Reused frozen calibration and passed E0", flush=True)
        return
    records = phase_records(config, "gate")
    model = None if config.backend == "fixture" else model_for(config)
    bundle = _initial_bundle(config)
    calibration = []
    for r in records:
        if r["experiment"] != "calibration":
            continue
        if not _valid(r, answer_skip=config.first_answer_tokens if config.exclude_first_answer_tokens else 0)[0]:
            raise ValueError(f"Invalid calibration transcript {r['record_id']}")
        rows, means = _measure(config, r, bundle, model, keep_means=True)
        count = sum(x["primary_include"] and x["segment"] == "answer" for x in rows)
        calibration.append((r["prompt_id"], means["answer"], count))
    # Token-weighted independent default answer center, never recomputed on eval data.
    center = np.average(np.array([x[1] for x in calibration]), axis=0, weights=[x[2] for x in calibration])
    rng = np.random.default_rng(config.control_seed)
    order = rng.permutation(len(calibration)); half = len(order) // 2
    values = np.array([x[1] for x in calibration])
    null = values[order[:half]].mean(0) - values[order[half:]].mean(0)
    from .geometry import DirectionBundle
    bundle = DirectionBundle(center, {**bundle.directions, "null": null},
          {**bundle.metadata, "centering": "token-weighted held-out default no-think answer mean",
           "calibration_prompt_ids": [x[0] for x in calibration],
           "null_halves": [[calibration[i][0] for i in part] for part in [order[:half], order[half:]]]})
    bundle.save(root / "directions.npz")
    all_rows = []
    role_scores = {}
    for r in records:
        if r["experiment"] != "E0":
            continue
        if not _valid(r, answer_skip=config.first_answer_tokens if config.exclude_first_answer_tokens else 0)[0]:
            raise ValueError(f"Invalid E0 transcript {r['record_id']}; stop and debug")
        rows, _ = _measure(config, r, bundle, model)
        all_rows.extend(rows)
        role_scores[r["role"]] = r["role_score"]
    df = pd.DataFrame(all_rows)
    df.to_csv(root / "e0_tokens.csv.gz", index=False)
    means = df[df.primary_include & (df.segment == "answer")].groupby(["seed", "prompt_id", "role"]).cos_assistant.mean().unstack("role")
    tests = []
    for seed in config.seeds:
        sub = means.loc[seed]
        role_columns = [c for c in sub if c != "default"]
        paired = sub["default"] - sub[role_columns].mean(axis=1)
        draws = rng.choice(paired.to_numpy(), size=(config.bootstrap_samples, len(paired)), replace=True).mean(axis=1)
        observed = sub[role_columns].mean().rank().to_numpy()
        expected = pd.Series({r: role_scores[r] for r in role_columns}).rank().to_numpy()
        corr = float(np.corrcoef(observed, expected)[0, 1]) if np.std(expected) and np.std(observed) else 0.0
        low, high = np.quantile(draws, [.025, .975])
        passed = bool(low > config.e0_min_default_advantage and corr >= config.e0_min_rank_correlation)
        tests.append({"seed": seed, "passed": passed, "default_advantage": float(paired.mean()),
                      "ci_low": float(low), "ci_high": float(high), "role_rank_spearman": corr,
                      "n_questions": len(paired), "n_roles": len(role_columns)})
    result = {"passed": all(t["passed"] for t in tests), "per_seed": tests,
              "criterion": "paired 95% bootstrap lower CI > frozen minimum AND role rank correlation >= frozen threshold",
              "synthetic": config.backend == "fixture"}
    write_json(root / "e0.json", result)
    require_gate(root)
    print("E0 passed for every seed; calibration frozen", flush=True)


def require_gate(root):
    path = Path(root) / "e0.json"
    if not path.exists() or not json.loads(path.read_text()).get("passed"):
        raise RuntimeError("E0 has not passed. Stop and debug the axis; no E1–E4 generation/extraction is allowed.")


def select_records(config, records):
    """Freeze a complete-seed paired cohort using token lengths, never projections."""
    root = _paths(config)
    by_key = {(r["seed"], r["prompt_id"], r["condition"], r["role"]): r for r in records}
    attrition = []
    for r in records:
        needs_think = r["condition"] in {"natural_think", "role_prompt"}
        valid, reason = _valid(r, config.min_think_tokens if needs_think else 0,
                               config.first_answer_tokens if config.exclude_first_answer_tokens else 0)
        r["eligible"], r["exclusion_reason"] = valid, reason
        attrition.append({k: r.get(k) for k in ["record_id", "seed", "prompt_id", "domain", "condition", "role", "experiment", "think_token_count", "answer_token_count", "finish_reason", "eligible", "exclusion_reason"]})
    selected = set()
    cohort = []
    prompts = read_jsonl(root / "prompts.jsonl")
    for domain in sorted({r["domain"] for r in records if r["experiment"] == "E1"}):
        candidates = sorted([p for p in _partition(prompts, "eval") if p["domain"] == domain],
                            key=lambda p: (p.get("candidate_rank", 0), p["prompt_id"]))[:config.candidate_limit_per_domain]
        eligible = []
        for p in candidates:
            conditions = ["natural_think", "natural_nothink"]
            rows = [by_key.get((s, p["prompt_id"], c, "default")) for s in config.seeds for c in conditions]
            if all(r is not None and r["eligible"] for r in rows):
                eligible.append(p["prompt_id"])
        chosen = eligible[:config.target_per_domain]
        selected.update(chosen)
        cohort.append({"domain": domain, "candidates": len(candidates), "eligible_all_seeds": len(eligible),
                       "selected": len(chosen), "requested": config.target_per_domain,
                       "minimum": config.min_per_domain, "shortfall": max(0, config.target_per_domain - len(chosen)),
                       "selected_prompt_ids": chosen})
    pd.DataFrame(attrition).to_csv(root / "attrition.csv", index=False)
    write_json(root / "cohort.json", cohort)
    if any(c["selected"] < config.min_per_domain for c in cohort):
        raise RuntimeError("Too few paired prompts survived. See attrition.csv/cohort.json; add frozen reserve candidates in a NEW run.")
    included = [r for r in records if r["experiment"] != "E3" and r["prompt_id"] in selected and r["eligible"]]
    # E3: every role/question comparison has its matched default, across all seeds.
    role_records = [r for r in records if r["experiment"] == "E3"]
    pairs = set()
    for r in role_records:
        q, role = r["prompt_id"], r["role"]
        rows = [by_key.get((s, q, "role_prompt", who)) for s in config.seeds for who in {"default", role}]
        if all(v is not None and v["eligible"] for v in rows):
            pairs.add((q, role))
    included.extend(r for r in role_records if (r["prompt_id"], r["role"]) in pairs)
    role_counts = pd.DataFrame([{"role": role, "paired_questions": len({q for q, who in pairs if who == role})}
                                for role in sorted({r["role"] for r in role_records})])
    role_counts.to_csv(root / "role_retention.csv", index=False)
    if "E3" in config.experiments and len([x for x in pairs if x[1] != "default"]) < 3:
        raise RuntimeError("Fewer than three valid paired role/question samples; E3 cannot be evaluated")
    return included


def extract_experiments(config, include_transplants=True):
    from .geometry import DirectionBundle, fit_persona_space
    from .transcripts import build_transplants
    root, _, _, _ = initialize(config)
    require_gate(root)
    tokenizer = tokenizer_for(config)
    all_records = phase_records(config, "experiments")
    records = select_records(config, all_records)
    if not include_transplants:
        if set(config.experiments) - {"E1", "E2"}:
            raise ValueError("extract-e1 is reserved for E1/E2-only runs; it cannot skip a configured E3/E4")
        if (root / "extraction.json").exists() and json.loads((root / "extraction.json").read_text()).get("transplants_included", False):
            raise ValueError("Full E2 extraction already exists; refusing to overwrite it with E1-only output")
        records = [r for r in records if r["experiment"] == "E1"]
    for r in list(records):
        if include_transplants and r["condition"] == "natural_think" and r["domain"] == "advice" and "E2" in config.experiments:
            records.extend(build_transplants(r, tokenizer))
    model = None if config.backend == "fixture" else model_for(config)
    bundle = DirectionBundle.load(root / "directions.npz")
    if config.pca and "E4" in config.experiments:
        role_records = [r for r in records if r["experiment"] == "E3"]
        planned_questions = {r["prompt_id"] for r in all_records if r["experiment"] == "E3"}
        complete_roles = {role for role in {r["role"] for r in role_records}
                          if {r["prompt_id"] for r in role_records if r["role"] == role} == planned_questions}
        role_records = [r for r in role_records if r["role"] in complete_roles]
        if "default" not in complete_roles or len(complete_roles) < 7:
            raise RuntimeError("E4 needs default plus >=6 roles completing the SAME planned question set. See role_retention.csv. "
                               "Run a preregistered E1–E3-only configuration if E4 is infeasible.")
        vectors = []
        means_dir = root / "segment_means"
        means_dir.mkdir(exist_ok=True)
        for r in role_records:
            path = means_dir / f"{r['record_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
            else:
                _, means = _measure(config, r, bundle, model, keep_means=True)
                saved = [{**{k: r[k] for k in ["prompt_id", "role", "model", "seed"]},
                          "segment": segment, "vector": vector.tolist()} for segment, vector in means.items()]
                write_json(path, saved)
            vectors.extend(saved)
        frame, report = fit_persona_space(vectors, n_components=5)
        report["synthetic"] = config.backend == "fixture"
        report["common_question_ids"] = sorted(planned_questions)
        report["excluded_incomplete_roles"] = sorted({r["role"] for r in records if r["experiment"] == "E3"} - complete_roles)
        report["extraction_passes"] = "E3 mean-only precursor plus final scalar pass; all other transcripts one scalar pass"
        write_json(root / "persona_space.json", report)
        np.savez_compressed(root / "persona_frame.npz", **frame)
        bundle = DirectionBundle(bundle.center, {**bundle.directions,
            **{f"pc{i + 1}": pc for i, pc in enumerate(frame["components"])}}, bundle.metadata)
        bundle.save(root / "directions_with_pcs.npz")
    shard_dir = root / "token_shards"
    shard_dir.mkdir(exist_ok=True)
    for i, r in enumerate(records):
        path = shard_dir / f"{r['record_id']}.csv.gz"
        if not path.exists():
            rows, _ = _measure(config, r, bundle, model)
            for row in rows:
                for pc in range(1, 6):
                    row.setdefault(f"cos_pc{pc}", np.nan)
            tmp = path.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
            tmp.replace(path)
        if (i + 1) % 50 == 0:
            print(f"Extracted {i + 1}/{len(records)} transcripts", flush=True)
    # Concatenate only the explicit cohort, never stale unrelated shards.
    tables = [pd.read_csv(shard_dir / f"{r['record_id']}.csv.gz") for r in records]
    tokens = pd.concat(tables, ignore_index=True)
    tokens.to_csv(root / "tokens.csv.gz", index=False)
    review_records = []
    for r in records:
        if r.get("source_record_id"):
            continue
        item = dict(r)
        item["transcript_id"] = r["record_id"]
        for segment in ["think", "answer"]:
            ids = [token for span in r["spans"] if span["segment"] == segment
                   for token in r["input_ids"][span["start"]:span["end"]]]
            item[f"{segment}_text"] = tokenizer.decode(ids, skip_special_tokens=False)
        review_records.append(item)
    write_jsonl(root / "review_transcripts.jsonl", review_records)
    from .judge import export_review
    if not (root / "review" / "sentences.jsonl").exists():
        export_review(review_records, root / "review", seed=config.analysis_seed,
                      domains=config.domains if "E1" in config.experiments else [],
                      require_roles="E3" in config.experiments)
    write_json(root / "extraction.json", {"transcripts": len(records), "tokens": len(tokens),
              "transplants_included": bool(include_transplants and "E2" in config.experiments),
              "conditions": sorted(tokens.condition.unique().tolist()), "synthetic": config.backend == "fixture",
              "unsupported_controls": sorted({c for r in records for c in r.get("unsupported_controls", [])})})
    print(f"Wrote {len(tokens):,} token rows from {len(records)} transcripts", flush=True)


def analyze_run(config):
    from .analysis import analyze
    root, _, _, _ = initialize(config)
    require_gate(root)
    tokens = pd.read_csv(root / "tokens.csv.gz", low_memory=False)
    judge_path = root / "judge_labels.csv"
    judge = pd.read_csv(judge_path) if judge_path.exists() else None
    report = analyze(tokens, root, bootstrap_samples=config.bootstrap_samples,
                     seed=config.analysis_seed, judge=judge)
    return report


def run_pipeline(config):
    """Process isolation guarantees generation and HF weights never coexist."""
    root, _, _, _ = initialize(config)
    config_path = (root / "config.json").resolve()
    for stage in ["generate-gate", "gate", "generate-experiments", "extract", "analyze"]:
        subprocess.run([sys.executable, "-m", "persona_dynamics", "worker", "--config", str(config_path),
                        "--stage", stage], check=True)
    return root
