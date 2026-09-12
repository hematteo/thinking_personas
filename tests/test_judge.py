import csv
import json

import pytest

from persona_dynamics.data import write_jsonl
from persona_dynamics.judge import (
    LABELS, agreement, export_review, heuristic_labels, import_labels,
    parse_judge_response, run_api, sentence_records, split_sentences, summarize_labels,
)


def transcript(i=0, domain="math", role="default"):
    return {"transcript_id": f"r-{domain}-{role}-{i}", "prompt_id": f"p-{domain}-{role}-{i}",
            "domain": domain, "role": role, "seed": 0,
            "condition": "natural_think" if role == "default" else "role_prompt",
            "text": "<think>First I calculate the sum. The user wants me to act as a pirate.\nI feel ready.</think>\nThe answer is 4.<|im_end|>",
            "messages": [{"role": "user", "content": "What is 2 plus 2?"}]}


def test_sentence_offsets_ids_and_segment_boundaries():
    rows = sentence_records([transcript()])
    assert [r['segment'] for r in rows] == ['think', 'think', 'think', 'answer']
    assert rows[0]['text'] == "First I calculate the sum."
    assert rows[-1]['text'] == "The answer is 4."
    assert rows[0]['sentence_id'] != rows[1]['sentence_id']
    assert rows == sentence_records([transcript()])
    text = "  Hello.  Second!\nThird? "
    for start, end, sentence in split_sentences(text):
        assert text[start:end] == sentence
        assert sentence == sentence.strip()
    with pytest.raises(ValueError, match="Duplicate"):
        sentence_records([transcript(), transcript()])


def test_manual_export_never_fills_human_labels_and_reports_shortfall(tmp_path):
    report = export_review([transcript()], tmp_path)
    assert report['calibration_sentences'] == 4
    assert not report['sampling_requirements_met']
    assert report['human_review_complete'] is False
    with (tmp_path / 'manual_calibration.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['manual_label'] == '' for row in rows)
    assert import_labels(tmp_path / 'manual_calibration.csv', sentence_records([transcript()]), manual=True) == []


def test_full_manual_review_uses_twenty_distinct_prompts_and_roles(tmp_path):
    records = [transcript(i, domain) for domain in ('math', 'code', 'advice', 'ai_philosophy') for i in range(20)]
    records += [transcript(0, "role", f"role-{i}") for i in range(20)]
    report = export_review(records, tmp_path, seed=13)
    assert report['sampling_requirements_met']
    assert report['review_counts'] == {'math': 20, 'code': 20, 'advice': 20, 'ai_philosophy': 20, 'roles': 20}
    assert report['calibration_sentences'] == 30
    assert not report['human_review_complete']


def test_annotation_import_rejects_unknown_duplicate_and_changed_text(tmp_path):
    sentences = sentence_records([transcript()])
    path = tmp_path / 'labels.jsonl'
    label = {'sentence_id': sentences[0]['sentence_id'], 'label': 'neutral'}
    write_jsonl(path, [label, label])
    with pytest.raises(ValueError, match='Duplicate'):
        import_labels(path, sentences)
    write_jsonl(path, [dict(label, sentence_id='unknown')])
    with pytest.raises(ValueError, match='Unknown'):
        import_labels(path, sentences)
    write_jsonl(path, [dict(label, text='tampered')])
    with pytest.raises(ValueError, match='changed'):
        import_labels(path, sentences)


def test_agreement_known_kappa_and_missing_predictions():
    manual = [{'sentence_id': str(i), 'label': label} for i, label in enumerate(['neutral', 'neutral', 'first_person', 'first_person'])]
    predicted = [{'sentence_id': str(i), 'label': label} for i, label in enumerate(['neutral', 'first_person', 'first_person', 'first_person'])]
    result = agreement(manual, predicted)
    assert result['accuracy'] == .75
    assert result['cohen_kappa'] == .5
    assert result['n'] == 4
    assert not result['minimum_30_met']
    assert agreement(manual, predicted[:-1])['missing_judge_ids'] == ['3']
    assert agreement([], [])['cohen_kappa'] is None


def test_heuristic_is_explicitly_unvalidated_and_aggregates_by_transcript():
    sentences = sentence_records([transcript()])
    labels = heuristic_labels(sentences)
    assert all(row['validated'] is False for row in labels)
    assert all(row['label_source'] == 'unvalidated_offline_heuristic' for row in labels)
    summary = summarize_labels(sentences, labels)
    think = next(r for r in summary if r['segment'] == 'think')
    assert think['n_sentences'] == 3
    assert think['meta_stance_rate'] == pytest.approx(1 / 3)
    assert sum(think[f'{label}_rate'] for label in LABELS) == pytest.approx(1)


def test_strict_judge_parser_and_api_requires_human_work(tmp_path):
    assert parse_judge_response('```json\n{"sentence_id":"a","label":"neutral"}\n```', 'a')['label'] == 'neutral'
    with pytest.raises(ValueError):
        parse_judge_response('{"sentence_id":"b","label":"neutral"}', 'a')
    with pytest.raises(ValueError):
        parse_judge_response('{"sentence_id":"a","label":"in-character"}', 'a')
    with pytest.raises(ValueError, match='Hand-label'):
        run_api([], tmp_path / 'out.jsonl', model='unused', cache_dir=tmp_path / 'cache', manual_labels=[])


def test_reduced_scope_export_and_explicit_missing_domain(tmp_path):
    records = [transcript(i, domain) for domain in ('math', 'advice') for i in range(20)]
    report=export_review(records,tmp_path/'reduced')
    assert report['sampling_requirements_met']
    assert report['required_roles']==0
    assert set(report['required_domains'])=={'math','advice'}
    assert not export_review(records,tmp_path/'missing',domains=['math','advice','code'])['sampling_requirements_met']
    assert not export_review(records,tmp_path/'roles',require_roles=True)['sampling_requirements_met']
