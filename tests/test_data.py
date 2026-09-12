from pathlib import Path
import json

import pytest

from persona_dynamics.data import (
    authored_ai_prompts, candidate_order, import_advice, import_upstream_roles,
    load_prompts, load_roles, offline_fixture_prompts, offline_fixture_roles,
    partition_prompts, prepare_data, select_roles, text_hash, validate_prompts,
    write_jsonl,
)


def test_normalized_overlap_cannot_leak_between_partitions():
    first = {"prompt_id": "a", "domain": "math", "text": "  What IS  １? ", "source": "test", "split": "calibration"}
    second = dict(first, prompt_id="b", text="what is 1?", split="eval")
    with pytest.raises(ValueError, match="overlaps"):
        validate_prompts([first, second])
    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        validate_prompts([first, dict(first)])


def test_partition_and_reserve_order_is_fixed_independent_of_input_order():
    source = offline_fixture_prompts(per_domain=20)
    a = partition_prompts(source, seed=31, calibration_per_domain=2, e0_per_domain=2)
    b = partition_prompts(list(reversed(source)), seed=31, calibration_per_domain=2, e0_per_domain=2)
    assert a == b
    assert a != partition_prompts(source, seed=32, calibration_per_domain=2, e0_per_domain=2)
    assert len({text_hash(r['text']) for r in a}) == len(a)
    assert all(r['split'] == 'eval' for r in a if r['domain'] == 'ai_philosophy')
    eval_rows = candidate_order(a, "math")
    assert [r['candidate_rank'] for r in eval_rows] == list(range(len(eval_rows)))


def test_role_stratification_is_from_frozen_scores():
    rows = offline_fixture_roles(90)
    a = select_roles(rows, seed=4, n_roles=60)
    b = select_roles(list(reversed(rows)), seed=4, n_roles=60)
    assert a == b
    assert len(a) == len({r['role'] for r in a}) == 60
    assert {s: sum(r['stratum'] == s for r in a) for s in ['low', 'mid', 'high']} == {'low': 20, 'mid': 20, 'high': 20}
    assert max(r['precomputed_score'] for r in a if r['stratum'] == 'low') < min(r['precomputed_score'] for r in a if r['stratum'] == 'mid')
    with pytest.raises(ValueError, match="multiple"):
        select_roles(rows, n_roles=5)


def test_local_upstream_exact_system_prompts_and_shared_questions(tmp_path):
    repo = tmp_path / "upstream"
    folder = repo / "data" / "roles" / "instructions"
    folder.mkdir(parents=True)
    scores = []
    for i in range(9):
        name = f"r{i}"
        (folder / f"{name}.json").write_text(json.dumps({"instruction": [{"pos": f"verbatim variant {j} of {name}"} for j in range(5)]}))
        scores.append({"role": name, "precomputed_score": i})
    write_jsonl(repo / "data" / "extraction_questions.jsonl", [{"id": i, "question": f"Shared question {i}?"} for i in range(10)])
    path = import_upstream_roles(scores, tmp_path / "out", seed=2, per_stratum=2, local_repo=repo)
    roles = load_roles(path)
    assert len(roles) == 9
    assert sum(r['e3_selected'] for r in roles) == 6
    assert len(roles[0]['questions']) == 5
    assert roles[0]['questions'] == roles[-1]['questions']
    assert roles[3]['system_prompt'] == "verbatim variant 0 of r3"
    assert len(roles[3]['system_prompts']) == 5


def test_advice_first_turn_filter_and_dedup(tmp_path):
    path = tmp_path / 'advice.jsonl'
    good = "I feel overwhelmed by my job and need advice about making time for friends."
    write_jsonl(path, [
        {"conversation": [{"role": "user", "content": good}]},
        {"conversation": [{"role": "user", "content": good.upper()}]},
        {"conversation": [{"role": "assistant", "content": good}]},
        {"language": "French", "conversation": [{"role": "user", "content": good}]},
        {"conversation": [{"role": "user", "content": "Explain the geography of the continents in detail."}]},
    ])
    rows = import_advice(path)
    assert len(rows) == 1
    assert rows[0]['text'] == good
    assert rows[0]['selection_filter'] == 'first-turn-personal-keyword-v1'


def test_real_prepare_does_not_substitute_fake_data(tmp_path):
    with pytest.raises(ValueError, match="advice"):
        prepare_data(tmp_path)
    fixtures = offline_fixture_prompts(per_domain=20)
    benchmarks = tmp_path / 'benchmarks.jsonl'
    advice = tmp_path / 'advice.jsonl'
    write_jsonl(benchmarks, [r for r in fixtures if r['domain'] in ('math', 'code')])
    write_jsonl(advice, [r for r in fixtures if r['domain'] == 'advice'])
    output = prepare_data(tmp_path / 'prepared', 1, advice_path=advice, curated_advice=True, benchmark_path=benchmarks)
    rows = load_prompts(output)
    ai = [r for r in rows if r['domain'] == 'ai_philosophy']
    assert len(ai) == 50
    assert len(authored_ai_prompts()) == 50
    assert all(r['split'] == 'eval' for r in ai)
    assert all('not sampled' in r['authorship'] for r in ai)
    assert len({r['text_sha256'] for r in rows}) == len(rows)
