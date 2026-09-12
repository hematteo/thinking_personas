"""Time-limited stages must preserve E1 before optional E2, without new generation."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from persona_dynamics.config import Config, load_config
from persona_dynamics import pipeline


def test_profile_freezes_small_workload_and_structured_controls():
    config = load_config(Path(__file__).parents[1] / "configs/a40_two_hour.yaml")
    assert config.seeds == [0]
    assert config.domains == ["advice", "math"]
    assert config.experiments == ["E1", "E2"] and not config.pca
    assert config.layer == 32 and config.tensor_parallel_size == 4
    assert config.target_per_domain == 40 and config.candidate_limit_per_domain == 50
    assert not config.e2_stepbystep and len(config.control_paths) >= 2
    with pytest.raises(ValueError, match="domains"):
        replace(config, domains=["advice", "advice"]).validate()


def test_e1_can_finish_before_e2_without_regenerating_or_losing_controls(tmp_path):
    config = Config(backend="fixture", model="fixture/persona-state-simulator", revision="synthetic-v1",
                    layer=0, seeds=[0], domains=["advice", "math"], output_dir=str(tmp_path),
                    target_per_domain=2, min_per_domain=2, candidate_limit_per_domain=4,
                    min_think_tokens=12, calibration_count=4, e0_question_count=2,
                    experiments=["E1", "E2"], pca=False, e2_stepbystep=False, bootstrap_samples=30).validate()
    pipeline.generate_phase(config, "gate")
    pipeline.calibrate_and_gate(config)
    pipeline.generate_phase(config, "experiments")
    requests = [json.loads(line) for line in (tmp_path / "requests_experiments.jsonl").read_text().splitlines()]
    assert len(requests) == 16
    assert {r["domain"] for r in requests} == {"advice", "math"}
    assert all(r["experiment"] == "E1" for r in requests)
    pipeline.extract_experiments(config, include_transplants=False)
    e1 = pd.read_csv(tmp_path / "tokens.csv.gz", low_memory=False)
    assert set(e1.experiment) == {"E1"}
    assert e1.prompt_id.nunique() == 4
    assert {"cos_ctrl_1", "cos_ctrl_2", "cos_null", "cos_random"} <= set(e1)
    assert not json.loads((tmp_path / "extraction.json").read_text())["transplants_included"]
    summary = pipeline.analyze_run(config)
    assert len(summary["figures"]) == 5  # E1 trajectory/histogram/specificity + boundary/norm
    assert "cot_as_answer" in summary["missing_e2_conditions"]
    shard_times = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "token_shards").glob("*.gz")}
    transcript_times = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "transcripts").glob("*.json")}
    pipeline.extract_experiments(config)
    full = pd.read_csv(tmp_path / "tokens.csv.gz", low_memory=False)
    assert len(full.record_id.unique()) == 24  # 8 natural + 2 advice × 8 transplants
    assert {"cot_in_scratch", "answer_in_special", "answer_as_answer", "cot_as_cot"} <= set(full.condition)
    assert "nothink_stepbystep" not in set(full.condition)
    assert transcript_times == {p.name: p.stat().st_mtime_ns for p in (tmp_path / "transcripts").glob("*.json")}
    assert all((tmp_path / "token_shards" / name).stat().st_mtime_ns == t for name, t in shard_times.items())
    # CSV infers all-missing source metadata as float before E2 introduces strings.
    pd.testing.assert_frame_equal(e1, full.loc[full.experiment.eq("E1")].reset_index(drop=True)[e1.columns], check_dtype=False)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        pipeline.extract_experiments(config, include_transplants=False)
