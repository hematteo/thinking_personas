import re
import os

import pytest

from persona_dynamics.transcripts import (
    annotate_transcript,
    build_transcript,
    build_transplants,
    token_contract,
)


class Tokenizer:
    """Tiny atomic-tag tokenizer; deliberately follows Qwen special=False tags."""
    tags = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_response>", "</tool_response>"]
    vocab = {tag: index + 1 for index, tag in enumerate(tags)}
    inverse = {value: key for key, value in vocab.items()}
    all_special_ids = [1, 2]
    eos_token_id = 2
    chat_template = "<think></think><tool_response></tool_response>"

    def get_added_vocab(self):
        return self.vocab

    def encode(self, text, add_special_tokens=False):
        parts = re.split("(" + "|".join(re.escape(t) for t in self.tags) + ")", text)
        return [token for part in parts for token in ([self.vocab[part]] if part in self.vocab else [ord(c) + 100 for c in part])]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inverse[i] if i in self.inverse else chr(i - 100) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        text += "<|im_start|>assistant\n"
        if not enable_thinking:
            text += "<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


def record(text="<think>\nReason about this.\n</think>\n\nAnswer here.<|im_end|>", condition="natural_think", prompt_text="Question"):
    tokenizer = Tokenizer()
    prompt = {"id": "p1", "domain": "advice", "text": prompt_text, "model": "fixture"}
    prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], enable_thinking=condition not in ("natural_nothink", "nothink_stepbystep"))
    return tokenizer, build_transcript(tokenizer, prompt, condition, tokenizer.encode(text), prefix, 4)


def test_spans_ignore_prompt_tags_and_exclude_format_and_eos():
    tokenizer, result = record(prompt_text="Explain <think> and </think> tags.")
    assert result["valid"]
    rows = annotate_transcript(result, tokenizer)
    assert "".join(r["token_text"] for r in rows if r["segment"] == "think") == "Reason about this."
    assert "".join(r["token_text"] for r in rows if r["segment"] == "answer") == "Answer here."
    assert [r["token_text"] for r in rows if r["is_boundary"]] == ["</think>"]
    assert all(r["token_idx"] >= result["assistant_start"] for r in rows)
    assert all(r["pos_from_boundary"] < 0 for r in rows if r["is_think"])
    assert sum(r["first5"] for r in rows) == 5
    assert {r["norm_pos"] for r in rows if r["segment"] == "answer"} >= {0.0, 1.0}


@pytest.mark.parametrize("text", ["<think>Unfinished", "Plain answer", "<think>a<think>b</think>answer", "<think>a</think></think>answer", "before<think>a</think>answer", "<think> </think>answer"])
def test_malformed_natural_think_is_rejected(text):
    _, result = record(text=text)
    assert not result["valid"]
    assert result["failure_reason"]
    assert annotate_transcript(result) == []


def test_nothink_parses_empty_block_from_generation_prefix():
    _, result = record(text="Actual answer.<|im_end|>", condition="natural_nothink")
    assert result["valid"]
    assert result["think_token_count"] == 0
    assert result["answer_token_count"] == len("Actual answer.")
    assert result["boundary_idx"] < result["prefix_length"]


def test_transplants_copy_exact_ids_and_match_factorial_prefixes():
    tokenizer, natural = record()
    transplants = {r["condition"]: r for r in build_transplants(natural, tokenizer)}
    assert len(transplants) == 8
    for name, result in transplants.items():
        source = "think" if name.startswith("cot_") else "answer"
        span = next(s for s in natural["spans"] if s["segment"] == source)
        original_ids = natural["input_ids"][span["start"] : span["end"]]
        rows = annotate_transcript(result)
        assert [r["token_id"] for r in rows] == original_ids
        assert result["content_token_ids"] == original_ids
    for context in ("as_cot", "as_answer", "in_scratch", "in_special"):
        cot, answer = transplants[f"cot_{context}"], transplants[f"answer_{context}"]
        assert cot["input_ids"][: cot["prefix_length"]] == answer["input_ids"][: answer["prefix_length"]]
    assert transplants["cot_as_answer"]["think_token_count"] == 0
    assert transplants["answer_as_cot"]["answer_token_count"] == 0


def test_qwen_atomic_control_does_not_require_special_flag():
    contract = token_contract(Tokenizer())
    assert contract["special_control_supported"]
    assert not contract["tags"]["<tool_response>"]["tokenizer_special"]
    assert not contract["tags"]["<think>"]["tokenizer_special"]


def test_unsupported_control_is_omitted_and_audited():
    tokenizer, natural = record()
    tokenizer.chat_template = "<think></think>"
    results = build_transplants(natural, tokenizer)
    assert len(results) == 6
    assert all("cot_in_special" in r["unsupported_controls"] for r in results)


def test_overlap_is_rejected():
    _, result = record()
    result["spans"].append(result["spans"][0])
    with pytest.raises(ValueError, match="overlap"):
        annotate_transcript(result)


@pytest.mark.skipif(not os.environ.get("PERSONA_QWEN_TOKENIZER_DIR"),
                    reason="Optional pinned Qwen tokenizer integration; set PERSONA_QWEN_TOKENIZER_DIR to downloaded tokenizer files")
def test_real_qwen_tokenizer_exact_spans_atomic_controls_and_causal_suffix():
    """No pretrained weights: real tokenizer plus a tiny random causal decoder.

    Verified fixture source: Qwen/Qwen3-32B revision
    9216db5781bf21249d130ec9da846c4624c16137. Download tokenizer/config files only
    outside this test, which never accesses the network or downloads assets.
    """
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    import numpy as np
    from persona_dynamics.geometry import DirectionBundle, extract_record

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.environ["PERSONA_QWEN_TOKENIZER_DIR"], local_files_only=True, trust_remote_code=False
    )
    contract = token_contract(tokenizer)
    assert contract["think_supported"] and contract["special_control_supported"]
    assert contract["tags"]["<think>"]["ids"] == [151667]
    assert contract["tags"]["</think>"]["ids"] == [151668]
    assert not contract["tags"]["<think>"]["tokenizer_special"]
    assert not contract["tags"]["<tool_response>"]["tokenizer_special"]
    messages = [{"role": "system", "content": "You are an AI Assistant."},
                {"role": "user", "content": "Explain <think>, </think>, and café thoughtfully."}]
    prompt = {"prompt_id": "real-tokenizer", "domain": "advice", "model": "tiny-random-qwen3", "messages": messages}
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    generated = tokenizer.encode("<think>\nFirst, consider café etiquette and the user's concern.\n</think>\n\nA clear answer is helpful.<|im_end|>", add_special_tokens=False)
    natural = build_transcript(tokenizer, prompt, "natural_think", generated, prefix, 0)
    assert natural["valid"]
    # Qwen has a '.\n' token: retain its whitespace to preserve the exact IDs.
    assert natural["think_text"].strip() == "First, consider café etiquette and the user's concern."
    assert natural["answer_text"] == "A clear answer is helpful."
    prefix_no_think = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    no_think = build_transcript(tokenizer, prompt, "natural_nothink", tokenizer.encode("A clear answer.<|im_end|>", add_special_tokens=False), prefix_no_think, 0)
    assert no_think["valid"] and no_think["think_token_count"] == 0
    transplants = build_transplants(natural, tokenizer)
    assert len(transplants) == 8
    for transplant in transplants:
        source = next(s for s in natural["spans"] if s["segment"] == transplant["source_content"])
        source_ids = natural["input_ids"][source["start"] : source["end"]]
        assert [row["token_id"] for row in annotate_transcript(transplant)] == source_ids

    config = transformers.Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=24,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=8, max_position_embeddings=256, attention_dropout=0.0)
    torch.manual_seed(19)
    model = transformers.Qwen3ForCausalLM(config).eval()
    bundle = DirectionBundle(np.zeros(16), {"assistant": np.arange(1, 17)})
    full = next(r for r in transplants if r["condition"] == "answer_as_cot")
    no_suffix = dict(full, input_ids=full["input_ids"][:full["spans"][0]["end"]])
    full_rows, _ = extract_record(model, full, bundle, layer=0)
    truncated_rows, _ = extract_record(model, no_suffix, bundle, layer=0)
    np.testing.assert_allclose([r["cos_assistant"] for r in full_rows],
                               [r["cos_assistant"] for r in truncated_rows], atol=1e-6)


def test_transplant_metadata_describes_transplant_not_original_generation():
    tokenizer, natural = record()
    natural.update(prefix_ids=natural['input_ids'][:natural['prefix_length']],
                   generated_ids=natural['input_ids'][natural['prefix_length']:],
                   finish_reason='stop', generation_seed=123)
    original = dict(natural)
    for r in build_transplants(natural, tokenizer):
        assert r['input_ids'] == r['prefix_ids'] + r['generated_ids']
        assert r['generated_text'] == tokenizer.decode(r['generated_ids'])
        assert r['finish_reason'] == 'teacher_forced'
        assert r['source_finish_reason'] == 'stop'
        assert r['generation_seed'] is None
        span = r['spans'][0]
        assert r[span['segment']+'_text'] == tokenizer.decode(r['input_ids'][span['start']:span['end']])
        assert r['answer_text' if span['segment']=='think' else 'think_text'] == ''
    assert natural == original
