from dataclasses import replace
import json

import pandas as pd
import pytest

from persona_dynamics.config import Config, load_config
from persona_dynamics.io import write_json, write_jsonl
from persona_dynamics import pipeline


def fixture_config(tmp_path, **overrides):
    values = dict(backend='fixture', model='fixture', output_dir=str(tmp_path),
                  seeds=[0, 1], target_per_domain=2, min_per_domain=2,
                  candidate_limit_per_domain=10, calibration_count=8,
                  role_count=6, e0_question_count=4, bootstrap_samples=20,
                  experiments=['E1', 'E2'], pca=False)
    values.update(overrides)
    return Config(**values).validate()


def test_generation_and_extraction_cannot_run_before_e0(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, 'generate_batches', lambda *a, **k: calls.append('generate'))
    monkeypatch.setattr(pipeline, 'model_for', lambda *a, **k: calls.append('model'))
    with pytest.raises(RuntimeError, match='E0 has not passed'):
        pipeline.generate_phase(config, 'experiments')
    with pytest.raises(RuntimeError, match='E0 has not passed'):
        pipeline.extract_experiments(config)
    write_json(tmp_path / 'e0.json', {'passed': False})
    with pytest.raises(RuntimeError, match='E0 has not passed'):
        pipeline.generate_phase(config, 'experiments')
    assert calls == []


def test_frozen_run_rejects_config_or_input_cache_mismatch(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    _, prompts, roles, run_id = pipeline.initialize(config)
    assert pipeline.initialize(config)[3] == run_id
    with pytest.raises(ValueError, match='different config, inputs, or code'):
        pipeline.initialize(replace(config, temperature=.7))
    changed = [dict(row) for row in prompts]
    changed[0]['text'] += ' Changed input.'
    monkeypatch.setattr(pipeline, '_inputs', lambda _: (changed, roles))
    with pytest.raises(ValueError, match='different config, inputs, or code'):
        pipeline.initialize(config)


def test_unknown_configuration_key_and_duplicate_seeds_fail(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('backend: fixture\nlayer_typo: 5\n')
    with pytest.raises(ValueError, match='Unknown config'):
        load_config(path)
    with pytest.raises(ValueError, match='unique'):
        fixture_config(tmp_path, seeds=[1, 1])


def record(prompt='p0', seed=0, condition='natural_think', think=200,
           answer=12, domain='advice', **extra):
    spans = []
    if condition in ('natural_think', 'role_prompt'):
        spans.append({'segment': 'think', 'start': 10, 'end': 10 + think})
    spans.append({'segment': 'answer', 'start': 20 + think, 'end': 20 + think + answer})
    row = {'record_id': f'{prompt}-{seed}-{condition}', 'seed': seed, 'prompt_id': prompt,
           'domain': domain, 'condition': condition, 'role': 'default',
           'experiment': 'E2' if condition == 'nothink_stepbystep' else 'E1',
           'spans': spans, 'valid': True, 'finish_reason': 'stop',
           'think_token_count': think, 'answer_token_count': answer}
    row.update(extra)
    return row


def test_eligibility_rejects_malformed_truncated_short_think_and_uses_skip():
    assert pipeline._valid(record(valid=False), 200)[0] is False
    assert pipeline._valid(record(finish_reason='length'), 200) == (False, 'truncated')
    assert pipeline._valid(record(think=199), 200) == (False, 'short_think')
    assert pipeline._valid(record(think=200), 200) == (True, 'eligible')
    assert pipeline._valid(record(answer=5), 200, answer_skip=5) == (False, 'missing_or_short_answer')
    assert pipeline._valid(record(answer=5), 200, answer_skip=0) == (True, 'eligible')
    assert pipeline._valid(record(answer=0), 200, answer_skip=0)[0] is False


def test_selection_uses_complete_seed_condition_pairs_and_frozen_reserves(tmp_path):
    config = fixture_config(tmp_path)
    prompts = [{'prompt_id': f'p{i}', 'domain': 'advice', 'split': 'eval', 'candidate_rank': i} for i in range(4)]
    write_jsonl(tmp_path / 'prompts.jsonl', prompts)
    records = [record(f'p{i}', seed, condition)
               for i in range(4) for seed in config.seeds
               for condition in ('natural_think', 'natural_nothink', 'nothink_stepbystep')]
    # A failure in only one seed removes p0 from every paired comparison.
    next(r for r in records if r['prompt_id'] == 'p0' and r['seed'] == 1 and r['condition'] == 'natural_think')['finish_reason'] = 'length'
    selected = pipeline.select_records(config, records)
    assert {r['prompt_id'] for r in selected} == {'p1', 'p2'}
    assert len(selected) == 12
    cohort = json.loads((tmp_path / 'cohort.json').read_text())
    assert cohort[0]['selected_prompt_ids'] == ['p1', 'p2']
    assert cohort[0]['eligible_all_seeds'] == 3
    attrition = pd.read_csv(tmp_path / 'attrition.csv')
    assert len(attrition) == len(records)
    assert (attrition.exclusion_reason == 'truncated').sum() == 1


def test_missing_paired_condition_does_not_silently_reduce_sample(tmp_path):
    config = fixture_config(tmp_path, candidate_limit_per_domain=2)
    write_jsonl(tmp_path / 'prompts.jsonl', [{'prompt_id': f'p{i}', 'domain': 'advice', 'split': 'eval', 'candidate_rank': i} for i in range(2)])
    records = [record(f'p{i}', seed, condition) for i in range(2) for seed in config.seeds
               for condition in ('natural_think', 'natural_nothink', 'nothink_stepbystep')]
    records = [r for r in records if not (r['prompt_id'] == 'p1' and r['seed'] == 1 and r['condition'] == 'natural_nothink')]
    # One missing paired no-think condition excludes p1 entirely.
    with pytest.raises(RuntimeError, match='Too few paired prompts survived'):
        pipeline.select_records(config, records)
    assert (tmp_path / 'attrition.csv').exists()
    assert json.loads((tmp_path / 'cohort.json').read_text())[0]['selected'] == 1


def test_length_termination_takes_priority_over_incomplete_block_label():
    from persona_dynamics.pipeline import _valid
    r={'valid':False,'failure_reason':'invalid_think_block','finish_reason':'length','spans':[]}
    assert _valid(r)==(False,'truncated')
    r['finish_reason']='stop'
    assert _valid(r)==(False,'invalid_think_block')


def test_empty_generation_does_not_load_any_backend():
    from persona_dynamics.backends import generate_batches
    # No configuration/backend access is necessary when every request is cached.
    assert list(generate_batches(None, [], None))==[]
