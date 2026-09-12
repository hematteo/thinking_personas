import numpy as np
import pytest

from persona_dynamics.geometry import (
    DirectionBundle,
    calibration_geometry,
    extract_record,
    fit_persona_space,
    orthogonalize,
)


def test_centered_cosine_raw_dot_and_norms():
    bundle = DirectionBundle(np.array([10.0, 0.0]), {"assistant": np.array([2.0, 0.0]), "ctrl_1": np.array([0.0, 4.0])})
    result = bundle.project(np.array([[10.0, 2.0], [13.0, 4.0], [10.0, 0.0]]))
    np.testing.assert_allclose(result["cos_assistant"][:2], [0.0, 0.6])
    np.testing.assert_allclose(result["cos_ctrl_1"][:2], [1.0, 0.8])
    np.testing.assert_allclose(result["dot_assistant"], [10.0, 13.0, 10.0])
    np.testing.assert_allclose(result["centered_norm"], [2.0, 5.0, 0.0])
    assert np.isnan(result["cos_assistant"][-1])


def test_bundle_roundtrip_and_validation(tmp_path):
    bundle = DirectionBundle(np.zeros(3), {"assistant": np.array([1, 0, 0]), "ctrl_1": np.array([0, 1, 0])}, {"layer": 32, "model": "Qwen/Qwen3-32B"})
    path = tmp_path / "geometry.npz"
    bundle.save(path)
    restored = DirectionBundle.load(path)
    np.testing.assert_array_equal(restored.center, bundle.center)
    np.testing.assert_array_equal(restored.directions["assistant"], bundle.directions["assistant"])
    assert restored.metadata["layer"] == 32
    with pytest.raises(ValueError, match="dimensions"):
        DirectionBundle(np.zeros(2), {"assistant": np.ones(3)})
    with pytest.raises(ValueError, match="zero"):
        DirectionBundle(np.zeros(2), {"assistant": np.zeros(2)})


def test_control_orthogonalization_rejects_axis_copy():
    axis = np.array([1.0, 2.0, 3.0])
    control = orthogonalize(np.array([3.0, 2.0, 1.0]), axis)
    assert abs(np.dot(control, axis)) < 1e-12
    assert np.linalg.norm(control) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="degenerate"):
        orthogonalize(axis, axis)


def test_calibration_random_halves_are_disjoint_reproducible():
    values = np.arange(24).reshape(6, 4)
    center, null, metadata = calibration_geometry(values, seed=14)
    np.testing.assert_allclose(center, values.mean(axis=0))
    np.testing.assert_allclose(null, calibration_geometry(values, seed=14)[1])
    assert set(metadata["null_half_a"]).isdisjoint(metadata["null_half_b"])
    assert sorted(metadata["null_half_a"] + metadata["null_half_b"]) == list(range(6))


def test_pca_uses_answer_center_and_preserves_translation():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(9, 6))
    shift = np.array([5.0, -3.0, 0.0, 0.0, 0.0, 0.0])
    rows = [{"model": "fixture", "role": f"role{i}", "segment": segment,
             "prompt_id": f"question{q}", "vector": vector + (shift if segment == "think" else 0)}
            for i, vector in enumerate(values) for segment in ("think", "answer") for q in range(2)]
    frame, report = fit_persona_space(rows)
    np.testing.assert_allclose(frame["center"], values.mean(axis=0))
    np.testing.assert_allclose(np.diag(report["pc_alignment"]), np.ones(5), atol=1e-12)
    coords = {(r["role"], r["segment"]): np.array([r[f"pc{i}"] for i in range(1, 6)]) for r in report["coordinates"]}
    np.testing.assert_allclose(coords["role0", "think"] - coords["role0", "answer"], shift @ frame["components"].T)
    metrics = report["cloud_geometry"]
    assert metrics["n_roles"] == len(values)
    assert metrics["think_to_answer_spread_ratio"] == pytest.approx(1.0)
    assert metrics["translation_norm"] == pytest.approx(np.linalg.norm(shift))
    assert metrics["rms_displacement_after_translation"] < 1e-12
    assert metrics["translation_residual_to_answer_spread_ratio"] < 1e-12


@pytest.mark.parametrize("scale", [0.0, 0.25])
def test_pca_describes_compressed_and_collapsed_thinking_clouds(scale):
    rng = np.random.default_rng(30)
    answer = rng.normal(size=(9, 6))
    center = answer.mean(axis=0)
    shift = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    think = center + scale * (answer - center) + shift
    rows = [{"role": f"role{i}", "model": "fixture", "segment": segment, "vector": values[i]}
            for i in range(len(answer)) for segment, values in [("answer", answer), ("think", think)]]
    _, report = fit_persona_space(rows)
    metrics = report["cloud_geometry"]
    assert metrics["think_to_answer_spread_ratio"] == pytest.approx(scale, abs=1e-12)
    assert metrics["translation_norm"] == pytest.approx(np.linalg.norm(shift))
    assert metrics["translation_residual_to_answer_spread_ratio"] == pytest.approx(1 - scale)
    if scale == 0:
        assert report["think_rank"] == 0
        assert report["think_pca_status"] == "unavailable: collapsed cloud"
        assert all(value is None for row in report["pc_alignment"] for value in row)
        assert all(value is None for value in report["think_explained_variance_ratio"])


def test_tiny_qwen_hook_matches_post_block_residual_and_omits_lm_head():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Installed Transformers predates Qwen3")
    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=8, max_position_embeddings=32, attention_dropout=0.0)
    torch.manual_seed(4)
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = [1, 4, 7, 9, 11, 2]
    record = {"valid": True, "input_ids": ids, "boundary_idx": 3,
              "spans": [{"segment": "think", "start": 1, "end": 3}, {"segment": "boundary", "start": 3, "end": 4}, {"segment": "answer", "start": 4, "end": 5}]}
    bundle = DirectionBundle(np.full(16, 0.2), {"assistant": np.arange(1, 17)})
    captured = []
    handle = model.model.layers[0].register_forward_hook(lambda _m, _i, output: captured.append((output[0] if isinstance(output, tuple) else output).detach().clone()))
    with torch.inference_mode():
        model.model(input_ids=torch.tensor([ids]), use_cache=False)
    handle.remove()
    expected = captured[0][0].numpy()

    def reject_logits(*_args, **_kwargs):
        raise AssertionError("The LM head must never run during extraction.")

    model.lm_head.register_forward_pre_hook(reject_logits)
    rows, means = extract_record(model, record, bundle, layer=0, keep_segment_means=True)
    positions = [row["token_idx"] for row in rows]
    direct = bundle.project(expected[positions])
    np.testing.assert_allclose([row["cos_assistant"] for row in rows], direct["cos_assistant"], atol=1e-6)
    np.testing.assert_allclose(means["think"], expected[1:3].mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(means["answer"], expected[4:5].mean(axis=0), atol=1e-7)
    assert len(model.model.layers[0]._forward_hooks) == 0
    _, skipped_means = extract_record(model, record, bundle, layer=0, keep_segment_means=True, answer_skip=1)
    assert "answer" not in skipped_means
    np.testing.assert_allclose(skipped_means["think"], means["think"], atol=1e-7)
