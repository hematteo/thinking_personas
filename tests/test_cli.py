"""Aggregation must preserve the scientific cohort and its provenance."""

import json

import pandas as pd
import pytest

from persona_dynamics.cli import _aggregate, main


def source_run(tmp_path, name, seed, prompt="p0", *, experiment="E1", role="default"):
    root = tmp_path/name
    root.mkdir()
    config = dict(name=name, output_dir=str(root), seeds=[seed], pca=False, experiments=["E1"],
                  bootstrap_samples=20, analysis_seed=7, target_per_domain=1, min_per_domain=1)
    manifest = dict(run_id=name, inputs={"prompts": "frozen-prompts"}, code={"pipeline.py": "frozen-code"},
                    numerical_versions={"numpy": "2.0", "torch": None, "transformers": None, "vllm": None},
                    python="3.11.0", synthetic=True)
    for filename, value in [("config.json", config), ("manifest.json", manifest),
                            ("e0.json", {"passed": True, "synthetic": True, "per_seed": [{"seed": seed, "passed": True}]}),
                            ("cohort.json", [{"domain": "advice", "candidates": 3, "eligible_all_seeds": 2,
                                               "selected": 1, "requested": 1, "selected_prompt_ids": [prompt]}])]:
        (root/filename).write_text(json.dumps(value))
    rows = [dict(run_id=name, seed=seed, model="fixture", experiment=experiment, domain="advice",
                 prompt_id=prompt, condition="natural_think" if experiment == "E1" else "role_prompt", role=role,
                 token_idx=i, segment=segment, primary_include=True, cos_assistant=.2+i*.1)
            for i, segment in enumerate(["think", "answer"])]
    pd.DataFrame(rows).to_csv(root/"tokens.csv.gz", index=False)
    (root/"directions.npz").write_bytes(b"same-frozen-ruler")
    return root


def test_aggregate_rejects_duplicate_observations(tmp_path):
    a, b = source_run(tmp_path, "a", 0), source_run(tmp_path, "b", 0)
    with pytest.raises(ValueError, match="duplicate samples"):
        _aggregate([a, b], tmp_path/"pooled")
    assert not (tmp_path/"pooled").exists()


@pytest.mark.parametrize("experiment,role", [("E1", "default"), ("E3", "skeptic")])
def test_aggregate_rejects_changed_prompt_or_role_question_cohort(tmp_path, experiment, role):
    a = source_run(tmp_path, "a", 0, "p0", experiment=experiment, role=role)
    b = source_run(tmp_path, "b", 1, "p1", experiment=experiment, role=role)
    with pytest.raises(ValueError, match="cohorts or role/question coverage"):
        _aggregate([a, b], tmp_path/"pooled")
    assert not (tmp_path/"pooled").exists()


def test_aggregate_rejects_changed_runtime(tmp_path):
    a, b = source_run(tmp_path, "a", 0), source_run(tmp_path, "b", 1)
    manifest = json.loads((b/"manifest.json").read_text())
    manifest["numerical_versions"]["numpy"] = "3.0"
    (b/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="numerical runtime"):
        _aggregate([a, b], tmp_path/"pooled")


def test_aggregate_preserves_provenance_labels_and_is_reanalyzable(tmp_path, monkeypatch):
    a, b = source_run(tmp_path, "a", 0), source_run(tmp_path, "b", 1)
    persona = {"coordinates": [], "cloud_geometry": {"n_roles": 6}, "frame_center": "fixed answer frame"}
    for i, run in enumerate([a, b]):
        (run/"persona_space.json").write_text(json.dumps(persona))
        pd.DataFrame([dict(sentence_id=f"s{i}", prompt_id="p0", seed=i, label="neutral")]).to_csv(run/"judge_labels.csv", index=False)
    calls = []
    def analyze(tokens, output_dir, **kwargs):
        calls.append((tokens, output_dir, kwargs))
        return {"ok": True}
    monkeypatch.setattr("persona_dynamics.analysis.analyze", analyze)
    out = tmp_path/"pooled"
    assert _aggregate([a, b], out) == {"ok": True}
    assert len(calls[0][0]) == 4
    assert calls[0][2]["seed"] == 7
    assert calls[0][2]["bootstrap_samples"] == 20
    assert len(calls[0][2]["judge"]) == 2
    gate = json.loads((out/"e0.json").read_text())
    assert gate["passed"] and gate["synthetic"]
    assert len(gate["source_gates"]) == 2
    cohort = json.loads((out/"cohort.json").read_text())[0]
    assert cohort["selected_prompt_ids"] == ["p0"]
    assert cohort["eligible_all_seeds"] is None  # Do not fabricate the candidate intersection.
    assert len(cohort["source_cohorts"]) == 2
    assert json.loads((out/"persona_space.json").read_text()) == persona
    metadata = json.loads((out/"aggregation.json").read_text())
    assert metadata["sources"][0]["manifest"]["numerical_versions"]["numpy"] == "2.0"
    main(["analyze", str(out)])
    assert len(calls) == 2
    assert calls[1][2]["bootstrap_samples"] == calls[0][2]["bootstrap_samples"]


def test_partial_judge_coverage_stays_unavailable(tmp_path, monkeypatch):
    a, b = source_run(tmp_path, "a", 0), source_run(tmp_path, "b", 1)
    pd.DataFrame([dict(sentence_id="s0", label="neutral")]).to_csv(a/"judge_labels.csv", index=False)
    calls = []
    monkeypatch.setattr("persona_dynamics.analysis.analyze", lambda *args, **kwargs: calls.append(kwargs))
    out = tmp_path/"pooled"
    _aggregate([a, b], out)
    assert calls[0]["judge"] is None
    assert not (out/"judge_labels.csv").exists()
    assert json.loads((out/"aggregation.json").read_text())["missing_judge_sources"] == [str(b)]


def test_normal_analyze_keeps_frozen_relative_output_identity(tmp_path, monkeypatch):
    run = source_run(tmp_path, "a", 0)
    config = json.loads((run/"config.json").read_text())
    config["output_dir"] = "a"
    (run/"config.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr("persona_dynamics.pipeline.analyze_run", lambda config: calls.append(config.output_dir))
    main(["analyze", str(run)])
    assert calls == ["a"]
