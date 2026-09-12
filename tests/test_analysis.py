"""Scientific aggregation invariants, not rendering snapshots."""

import json

import numpy as np
import pandas as pd
import pytest

from persona_dynamics.analysis import (
    _summarize, analyze, bootstrap_summary, e1_contrasts, e2_contrasts,
    e3_role_shifts, role_regression, segment_means,
)


def token_rows(prompt, think, answer, seed=0, length=8, experiment="E1", role="default", condition="natural_think"):
    rows = []
    for segment, value in [("think", think), ("boundary", 99), ("answer", answer)]:
        count = 1 if segment == "boundary" else length
        for i in range(count):
            rows.append(dict(run_id=f"seed-{seed}", seed=seed, model="fixture/persona-state-simulator",
                             prompt_id=prompt, domain="advice", condition=condition, role=role,
                             experiment=experiment, segment=segment, segment_idx=i,
                             token_idx=i+(length if segment != "think" else 0), token_id=7,
                             is_think=segment == "think", norm_pos=i/max(1, count-1),
                             pos_from_boundary=i-length if segment == "think" else i+1 if segment == "answer" else 0,
                             primary_include=segment != "boundary", resid_norm=10+value,
                             centered_norm=1, dot_assistant=value, cos_assistant=value,
                             cos_ctrl_1=.01, cos_ctrl_2=-.01, cos_null=0., cos_random=.001,
                             synthetic=True))
    return rows


def test_tokens_do_not_become_statistical_replicates():
    tokens = pd.DataFrame(token_rows("long", .2, .6, length=1000)+token_rows("short", .7, .8, length=6))
    gaps = e1_contrasts(segment_means(tokens))
    effects = _summarize(gaps, ["model", "domain", "direction", "contrast"], "prompt_id", 100, 0)
    row = effects.loc[effects.seed_scope.eq("pooled") & effects.direction.eq("cos_assistant") &
                      effects.contrast.eq("think_minus_answer")].iloc[0]
    assert row.n == 2
    assert row["mean"] == pytest.approx(-.25)


def test_seeds_are_averaged_within_prompt_before_bootstrap():
    rows = [dict(model="m", domain="d", direction="cos_assistant", contrast="gap", prompt_id=p, seed=s, value=v)
            for p, s, v in [("p0", 1, 0), ("p0", 2, 1), ("p0", 3, 2), ("p1", 1, 3)]]
    effects = _summarize(pd.DataFrame(rows), ["model", "domain", "direction", "contrast"], "prompt_id", 100, 4)
    pooled = effects.loc[effects.seed_scope.eq("pooled")].iloc[0]
    assert pooled.n == 2
    assert pooled["mean"] == 2
    assert effects.loc[effects.seed_scope.eq("1"), "n"].item() == 2


def test_missing_pairs_are_dropped_and_boundary_is_separate():
    tokens = pd.DataFrame(token_rows("paired", .2, .5)+token_rows("unpaired", .8, .4))
    tokens.loc[tokens.prompt_id.eq("unpaired") & tokens.segment.eq("answer"), "primary_include"] = False
    means = segment_means(tokens)
    gaps = e1_contrasts(means)
    paired = gaps.loc[gaps.direction.eq("cos_assistant") & gaps.contrast.eq("think_minus_answer")]
    assert list(paired.prompt_id) == ["paired"]
    assert paired.value.item() == pytest.approx(-.3)
    assert means.loc[means.segment.eq("boundary") & means.direction.eq("cos_assistant"), "value"].eq(99).all()


def test_factorial_recovers_tag_and_content_with_matched_plain_answer():
    cells = [("natural_think", "think", 5), ("cot_as_answer", "answer", 3),
             ("answer_as_cot", "think", 2), ("answer_as_answer", "answer", 0)]
    means = pd.DataFrame([dict(seed=0, model="m", prompt_id="p", domain="advice", eligible=True,
                              experiment="E1" if condition == "natural_think" else "E2", condition=condition,
                              segment=segment, direction="cos_assistant", value=value)
                          for condition, segment, value in cells])
    contrasts, matched = e2_contrasts(means)
    assert matched
    values = contrasts.set_index("contrast").value
    assert values["factorial_tag_main"] == 2
    assert values["factorial_content_main"] == 3
    assert values["factorial_tag_content_interaction"] == 0


def test_role_shifts_pair_the_same_question_before_role_aggregation():
    rows = []
    for role, prompt, value in [("default", "easy", 10), ("default", "hard", -10),
                                ("skeptic", "easy", 9), ("skeptic", "unmatched", 900)]:
        rows.append(dict(seed=0, model="m", prompt_id=prompt, domain="roles", experiment="E3", role=role,
                         segment="answer", direction="cos_assistant", eligible=True, value=value))
    shifts = e3_role_shifts(pd.DataFrame(rows))
    assert shifts.value.tolist() == [1]
    assert shifts.prompt_id.tolist() == ["easy"]


def test_small_sample_bootstrap_and_reproducibility():
    assert bootstrap_summary([], 100)["n"] == 0
    assert bootstrap_summary([1], 100)["ci_low"] is None
    assert bootstrap_summary([1, 1, 1], 100)["dz"] is None
    a = bootstrap_summary([-3, 1, 2, 4, 8, 10], 200, 7)
    assert a == bootstrap_summary([-3, 1, 2, 4, 8, 10], 200, 7)
    assert a["dz"] == pytest.approx(np.mean([-3, 1, 2, 4, 8, 10])/np.std([-3, 1, 2, 4, 8, 10], ddof=1))


def test_role_slope_recovers_known_compression_and_handles_degenerate_axis():
    data = pd.DataFrame({"answer": np.arange(10)/10, "think": .1+np.arange(10)/40})
    regression = role_regression(data, 100, 3)
    assert regression["n_roles"] == 10
    assert regression["slope"] == pytest.approx(.25)
    assert regression["intercept"] == pytest.approx(.1)
    assert regression["slope_ci_low"] == pytest.approx(.25)
    assert regression["slope_ci_high"] == pytest.approx(.25)
    assert role_regression(pd.DataFrame({"answer": [1, 1, 1], "think": [0, 1, 2]}), 100)["slope"] is None


def test_answer_refusals_flag_roles_even_when_thinking_has_no_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr("persona_dynamics.plots.make_plots", lambda *args, **kwargs: [])
    tokens = pd.DataFrame(token_rows("q", .2, .5, experiment="E3") +
                          token_rows("q", .1, .3, experiment="E3", role="skeptic"))
    labels = pd.DataFrame([dict(model="fixture/persona-state-simulator", seed=0, prompt_id="q", domain="advice",
                               role="skeptic", segment=segment, label=label)
                           for segment, label in [("think", "neutral"), ("answer", "refusal")]])
    summary = analyze(tokens, tmp_path, bootstrap_samples=20, judge=labels)
    roles = pd.read_csv(tmp_path/"role_shifts.csv")
    assert roles.refusal_flag.all()
    assert roles.meta_stance_rate.eq(0).all()
    assert not summary["manual_review"]["complete"]
    assert not summary["reporting_readiness"]["human_judge_calibration_complete"]


def test_analysis_preserves_human_notes_and_reports_collapsed_persona_space(tmp_path, monkeypatch):
    monkeypatch.setattr("persona_dynamics.plots.make_plots", lambda *args, **kwargs: [])
    tokens = pd.DataFrame(token_rows("p", .2, .5))
    (tmp_path/"human_notes.md").write_text("My independently reviewed finding.\n")
    (tmp_path/"persona_space.json").write_text(json.dumps({"think_pca_status": "unavailable: collapsed cloud",
        "cloud_geometry": {"think_to_answer_spread_ratio": 0, "translation_residual_to_answer_spread_ratio": 1}}))
    summary = analyze(tokens, tmp_path, bootstrap_samples=20)
    assert summary["persona_space"]["cloud_geometry"]["think_to_answer_spread_ratio"] == 0
    assert (tmp_path/"human_notes.md").read_text() == "My independently reviewed finding.\n"
    assert "collapsed cloud" in (tmp_path/"results.md").read_text()


def test_collapsed_thinking_pca_has_explicit_unavailable_plot(tmp_path):
    from persona_dynamics.plots import plot_persona_space
    (tmp_path/"persona_space.json").write_text(json.dumps({"pc_alignment": [[None, None], [None, None]],
                                                         "think_pca_status": "unavailable: collapsed cloud"}))
    outputs = plot_persona_space(tmp_path, synthetic=True)
    assert outputs == ["figures/09_pc_agreement.png"]
    assert (tmp_path/outputs[0]).exists()


def test_minimal_analysis_writes_valid_tables_report_and_figures(tmp_path):
    tokens = pd.DataFrame(token_rows("p0", .2, .5)+token_rows("p1", .3, .7))
    (tmp_path/"e0.json").write_text(json.dumps({"passed": True, "synthetic": True}))
    summary = analyze(tokens, tmp_path, bootstrap_samples=20, seed=2)
    assert summary["synthetic"]
    assert summary["e0"]["passed"]
    assert len(summary["figures"]) >= 4
    assert all((tmp_path/figure).is_file() for figure in summary["figures"])
    assert "SYNTHETIC PIPELINE VALIDATION" in (tmp_path/"results.md").read_text()
    boundary = pd.read_csv(tmp_path/"boundary_first5.csv")
    assert {"boundary", "answer_first5", "answer_all", "answer"} <= set(boundary.segment)
    loaded = json.loads((tmp_path/"summary.json").read_text())
    assert "H2" in loaded["decision"]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'invalid'])
def test_analysis_rejects_corrupted_measurements(tmp_path, bad):
    from persona_dynamics.analysis import _prepare
    tokens = pd.DataFrame(token_rows('p', .2, .5))
    tokens['cos_ctrl_1'] = tokens.cos_ctrl_1.astype(object)
    tokens.loc[0, 'cos_ctrl_1'] = bad
    with pytest.raises(ValueError, match='non-finite'):
        _prepare(tokens)


def test_bootstrap_rejects_silent_sample_size_changes():
    with pytest.raises(ValueError, match='finite'):
        bootstrap_summary([1, 2, np.nan], 20)


def test_reduced_scope_review_does_not_require_absent_role_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr('persona_dynamics.plots.make_plots', lambda *a, **kw: [])
    tokens = pd.DataFrame([r for i in range(20) for r in token_rows(f'p{i}', .2, .5)])
    (tmp_path/'review').mkdir()
    reviews=pd.DataFrame([dict(prompt_id=f'p{i}',domain='advice',role='default',reviewed='yes') for i in range(20)])
    reviews.to_csv(tmp_path/'review/review.csv',index=False)
    summary=analyze(tokens,tmp_path,bootstrap_samples=20)
    assert summary['manual_review']['complete']
    assert summary['manual_review']['required_roles']==0
    assert 'E3 role intervention was not measured' in (tmp_path/'results.md').read_text()
    assert 'E4 persona-space/PCA analysis was not measured' in (tmp_path/'results.md').read_text()
    # An invented prompt must not count toward the review quota.
    reviews.loc[0,'prompt_id']='not_in_experiment'
    reviews.to_csv(tmp_path/'review/review.csv',index=False)
    assert not analyze(tokens,tmp_path,bootstrap_samples=20)['manual_review']['complete']


def test_analysis_rejects_entire_missing_control_and_duplicate_token():
    from persona_dynamics.analysis import _prepare
    tokens=pd.DataFrame(token_rows('p',.2,.5))
    tokens['cos_ctrl_1']=np.nan
    with pytest.raises(ValueError,match='non-finite'):
        _prepare(tokens)
    tokens=pd.DataFrame(token_rows('p',.2,.5))
    tokens['record_id']='r'
    tokens['token_idx']=np.arange(len(tokens))
    with pytest.raises(ValueError,match='Duplicate'):
        _prepare(pd.concat([tokens,tokens.iloc[[0]]],ignore_index=True))
