import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from persona_dynamics import pipeline
from persona_dynamics.config import Config

spec = importlib.util.spec_from_file_location("extend_candidate_run", Path(__file__).parents[1] / "scripts/extend_candidate_run.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_extension_preserves_ruler_and_tokens_and_rejects_changed_settings(tmp_path):
    source = tmp_path / "source"
    config = Config(backend="fixture", model="fixture/persona-state-simulator", revision="synthetic-v1",
                    layer=0, seeds=[0], domains=["advice", "math"], output_dir=str(source),
                    target_per_domain=2, min_per_domain=2, candidate_limit_per_domain=2,
                    min_think_tokens=12, calibration_count=4, e0_question_count=2,
                    experiments=["E1", "E2"], pca=False, e2_stepbystep=False, bootstrap_samples=30).validate()
    pipeline.generate_phase(config, "gate")
    pipeline.calibrate_and_gate(config)
    pipeline.generate_phase(config, "experiments")
    original = {p.name: p.read_bytes() for p in (source / "transcripts").glob("*.json")}
    child = replace(config, output_dir=str(tmp_path / "child"), candidate_limit_per_domain=4)
    with pytest.raises(ValueError, match="preserve"):
        module.extend(replace(child, min_think_tokens=0), source)
    module.extend(child, source)
    target = Path(child.output_dir)
    assert (source / "directions.npz").read_bytes() == (target / "directions.npz").read_bytes()
    for name, data in original.items():
        old = json.loads(data)
        new = json.loads((target / "transcripts" / name).read_text())
        assert new.pop("source_run_id") == old["run_id"]
        assert new.pop("run_id") != old.pop("run_id")
        assert new == old
    pipeline.generate_phase(child, "experiments")
    assert len(pipeline.phase_records(child, "experiments")) == 16
    assert original == {p.name: p.read_bytes() for p in (source / "transcripts").glob("*.json")}
