"""Exact token-span bookkeeping for Qwen thinking and causal transplants.

All offsets are absolute, zero based, half-open token offsets. Tags in user or
system messages never enter the parser. Content IDs are copied into transplants;
decoding and re-encoding the content would silently change the experimental unit.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import hashlib
import json
from typing import Any


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(map(int, tokenizer.encode(text, add_special_tokens=False)))


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def _occurrences(ids: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [i for i in range(len(ids) - len(needle) + 1) if ids[i : i + len(needle)] == needle]


def token_contract(tokenizer: Any) -> dict[str, Any]:
    """Inspect atomic tags and the template instead of assuming special status.

    Qwen3's released tokenizer marks think and tool-response tags special=False,
    although they are dedicated single added tokens. The relevant matched control
    is therefore *atomic added tokens*, not the tokenizer's special-token flag.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    added = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    template = getattr(tokenizer, "chat_template", "") or ""
    if isinstance(template, dict):
        template = "\n".join(template.values())
    report: dict[str, Any] = {"tags": {}}
    for text in ("<think>", "</think>", "<scratch>", "</scratch>", "<tool_response>", "</tool_response>"):
        ids = _encode(tokenizer, text)
        atomic = len(ids) == 1 and _decode(tokenizer, ids) == text
        report["tags"][text] = {
            "ids": ids,
            "atomic": atomic,
            "added_token": text in added,
            "tokenizer_special": bool(atomic and ids[0] in special),
            "in_chat_template": text in template,
        }
    report["think_supported"] = all(report["tags"][tag]["atomic"] for tag in ("<think>", "</think>"))
    report["special_control_supported"] = all(
        report["tags"][tag]["atomic"]
        and report["tags"][tag]["added_token"]
        and report["tags"][tag]["in_chat_template"]
        for tag in ("<tool_response>", "</tool_response>")
    )
    report["special_control_interpretation"] = "template-supported atomic added-token block"
    return report


def _messages(prompt: dict[str, Any], role: str) -> list[dict[str, str]]:
    if "messages" in prompt:
        return deepcopy(prompt["messages"])
    messages = []
    system = prompt.get("system_prompt")
    if system:
        messages.append({"role": "system", "content": system})
    elif role != "default":
        raise ValueError("Role transcripts must provide the actual messages or system_prompt.")
    messages.append({"role": "user", "content": str(prompt.get("text", prompt.get("prompt", "")))})
    return messages


def _template(tokenizer: Any, messages: list[dict[str, str]], thinking: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    return _token_ids(ids)


def _token_ids(ids: Any) -> list[int]:
    """Normalize v4 list / v5 BatchEncoding template outputs without retokenizing."""
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one chat-template prefix.")
        ids = ids[0]
    return list(map(int, ids))


def _assistant_start(tokenizer: Any, prefix: list[int], messages: list[dict[str, str]]) -> int:
    header = _encode(tokenizer, "<|im_start|>assistant\n")
    starts = _occurrences(prefix, header)
    if starts:
        return starts[-1] + len(header)
    # A tokenizer with a different assistant header must still expose consistent
    # toggle templates. Their common prefix is the assistant-content boundary.
    enabled, disabled = _template(tokenizer, messages, True), _template(tokenizer, messages, False)
    common = 0
    while common < min(len(enabled), len(disabled)) and enabled[common] == disabled[common]:
        common += 1
    if common == 0 or enabled == disabled:
        raise ValueError("Cannot locate assistant content from this chat template.")
    if prefix not in (enabled, disabled):
        raise ValueError("Generation prefix does not match the tokenizer chat template.")
    return common


def _trim(tokenizer: Any, ids: list[int], start: int, end: int) -> tuple[int, int]:
    # Never trim part of a BPE token (e.g. Qwen's single '.\n' token). Such
    # inseparable whitespace remains in every exact-ID matched transplant.
    while start < end and _decode(tokenizer, ids[start : start + 1]).isspace():
        start += 1
    while end > start and _decode(tokenizer, ids[end - 1 : end]).isspace():
        end -= 1
    return start, end


def _identity(record: dict[str, Any]) -> str:
    fields = [record.get(k) for k in ("prompt_id", "condition", "role", "seed", "model", "input_ids")]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()[:24]


def build_transcript(
    tokenizer: Any,
    prompt: dict[str, Any],
    condition: str,
    generated_ids: list[int],
    prefix_ids: list[int],
    seed: int,
    role: str = "default",
    experiment: str = "E1",
) -> dict[str, Any]:
    """Build a valid exact-token record, or an auditable invalid record.

    Malformed / unterminated think blocks are never relabeled as answers. The
    caller must separately exclude generation-limit completions (finish_reason)
    and apply the preregistered minimum thinking length.
    """
    contract = token_contract(tokenizer)
    if not contract["think_supported"]:
        raise ValueError("Thinking delimiters must be dedicated atomic tokens.")
    messages = _messages(prompt, role)
    prefix = _token_ids(prefix_ids)
    ids = prefix + list(map(int, generated_ids))
    start = _assistant_start(tokenizer, prefix, messages)
    record = {
        "prompt_id": str(prompt.get("prompt_id", prompt.get("id", ""))),
        "domain": prompt.get("domain", "unknown"),
        "condition": condition,
        "role": role,
        "model": prompt.get("model", "unknown"),
        "seed": int(seed),
        "experiment": experiment,
        "messages": messages,
        "input_ids": ids,
        "prefix_length": len(prefix),
        "assistant_start": start,
        "text": _decode(tokenizer, ids[start:]),
        "generated_text": _decode(tokenizer, list(generated_ids)),
        "spans": [],
        "valid": True,
        "errors": [],
        "tokenizer_contract": contract,
    }
    for key in ("run_id", "synthetic", "role_score", "source", "question_id"):
        if key in prompt:
            record[key] = prompt[key]
    if not record["prompt_id"]:
        raise ValueError("Every prompt requires an id or prompt_id.")
    stops = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Only end-of-turn/end-of-text terminate content; other special tokens can
    # occur in arbitrary text and must not prematurely terminate a transcript.
    stops = {i for i in stops if _decode(tokenizer, [i]) in ("<|im_end|>", "<|endoftext|>")}
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        stops.add(int(eos))
    end = next((i for i in range(start, len(ids)) if ids[i] in stops), len(ids))
    open_id = contract["tags"]["<think>"]["ids"][0]
    close_id = contract["tags"]["</think>"]["ids"][0]
    opens = [i for i in range(start, end) if ids[i] == open_id]
    closes = [i for i in range(start, end) if ids[i] == close_id]
    no_think = condition in ("natural_nothink", "nothink_stepbystep")
    if len(opens) != 1 or len(closes) != 1 or (opens and closes and opens[0] >= closes[0]):
        record["errors"].append("Expected exactly one ordered, complete think block in assistant content.")
    else:
        opening, closing = opens[0], closes[0]
        before = _trim(tokenizer, ids, start, opening)
        if before[0] < before[1]:
            record["errors"].append("Unexpected assistant content before the think block.")
        think_start, think_end = _trim(tokenizer, ids, opening + 1, closing)
        answer_start, answer_end = _trim(tokenizer, ids, closing + 1, end)
        if no_think and think_end > think_start:
            record["errors"].append("No-think generation contained a nonempty think block.")
        if not no_think and think_end <= think_start:
            record["errors"].append("Thinking generation contained an empty think block.")
        if answer_end <= answer_start:
            record["errors"].append("Transcript has no answer content.")
        record["boundary_idx"] = closing
        if think_end > think_start:
            record["spans"].append({"segment": "think", "start": think_start, "end": think_end})
        record["spans"].append({"segment": "boundary", "start": closing, "end": closing + 1})
        if answer_end > answer_start:
            record["spans"].append({"segment": "answer", "start": answer_start, "end": answer_end})
    record["valid"] = not record["errors"]
    record["failure_reason"] = "; ".join(record["errors"]) or None
    for segment in ("think", "answer"):
        record[f"{segment}_token_count"] = sum(s["end"] - s["start"] for s in record["spans"] if s["segment"] == segment)
        record[f"{segment}_text"] = "".join(
            _decode(tokenizer, ids[s["start"] : s["end"]])
            for s in record["spans"] if s["segment"] == segment
        )
    record["record_id"] = record["transcript_id"] = _identity(record)
    return record


def annotate_transcript(record: dict[str, Any], tokenizer: Any = None) -> list[dict[str, Any]]:
    """One metadata row per content/boundary token; no prompt, opening tag or EOS."""
    if not record.get("valid", True):
        return []
    common = {key: record.get(key) for key in (
        "record_id", "transcript_id", "prompt_id", "domain", "condition", "role", "model", "seed", "experiment", "source_content", "source_record_id", "run_id", "synthetic", "role_score", "question_id"
    )}
    boundary = record.get("boundary_idx")
    rows = []
    seen: set[int] = set()
    for span in record["spans"]:
        segment, start, end = span["segment"], int(span["start"]), int(span["end"])
        if not 0 <= start < end <= len(record["input_ids"]):
            raise ValueError(f"Invalid span: {span}")
        for offset, index in enumerate(range(start, end)):
            if index in seen:
                raise ValueError("Transcript spans overlap.")
            seen.add(index)
            first5 = segment == "answer" and offset < 5
            row = dict(common, token_idx=index, token_id=int(record["input_ids"][index]),
                       segment=segment, is_think=segment == "think", is_boundary=segment == "boundary",
                       first5=first5, is_first5_answer=first5, segment_token_idx=offset, segment_idx=offset,
                       norm_pos=offset / max(end - start - 1, 1),
                       pos_from_boundary=index - boundary if boundary is not None else None,
                       primary_include=segment in ("think", "answer"))
            if tokenizer is not None:
                row["token_text"] = _decode(tokenizer, [row["token_id"]])
            rows.append(row)
    return rows


def build_transplants(record: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Construct a matched content × context factorial, plus delimiter controls.

    Return exact-ID causal continuations; none require generated following text.
    Unsupported atomic controls are omitted and recorded on each returned record.
    """
    if not record.get("valid", False):
        return []
    content = {}
    for segment in ("think", "answer"):
        spans = [span for span in record["spans"] if span["segment"] == segment]
        if len(spans) != 1:
            raise ValueError("Transplants require one natural thinking and one answer span.")
        span = spans[0]
        content[segment] = record["input_ids"][span["start"] : span["end"]]
    messages = record["messages"]
    thinking_prefix = _template(tokenizer, messages, True)
    no_think_prefix = _template(tokenizer, messages, False)
    contract = token_contract(tokenizer)
    open_id, close_id = (_encode(tokenizer, tag)[0] for tag in ("<think>", "</think>"))
    assistant_start = _assistant_start(tokenizer, thinking_prefix, messages)
    no_think_tail = no_think_prefix[_assistant_start(tokenizer, no_think_prefix, messages) :]
    if no_think_tail.count(open_id) != 1 or no_think_tail.count(close_id) != 1:
        raise ValueError("No-think chat template must supply an empty think block.")
    oi, ci = no_think_tail.index(open_id), no_think_tail.index(close_id)
    if oi >= ci or _decode(tokenizer, no_think_tail[oi + 1 : ci]).strip():
        raise ValueError("No-think chat template does not contain an empty think block.")
    if thinking_prefix[assistant_start:]:
        tail = thinking_prefix[assistant_start:]
        if tail.count(open_id) != 1 or close_id in tail:
            raise ValueError("Thinking template must end before content, with at most an open think block.")
        think_base = thinking_prefix
    else:
        think_base = thinking_prefix + _encode(tokenizer, "<think>\n")
    controls = [("as_cot", think_base, _encode(tokenizer, "\n</think>"), "think"),
                ("as_answer", no_think_prefix, [], "answer"),
                ("in_scratch", no_think_prefix + _encode(tokenizer, "<scratch>\n"), _encode(tokenizer, "\n</scratch>"), "answer")]
    if contract["special_control_supported"]:
        controls.append(("in_special", no_think_prefix + _encode(tokenizer, "<tool_response>\n"), _encode(tokenizer, "\n</tool_response>"), "answer"))
    results = []
    for source_segment, source_ids in content.items():
        source_name = "cot" if source_segment == "think" else "answer"
        for context_name, prefix, suffix, segment in controls:
            transplant = deepcopy(record)
            transplant.update(
                condition=f"{source_name}_{context_name}", experiment="E2",
                input_ids=prefix + source_ids + suffix,
                prefix_ids=list(prefix), generated_ids=list(source_ids + suffix),
                prefix_length=len(prefix), assistant_start=assistant_start,
                source_content=source_segment, source_record_id=record["record_id"],
                content_token_ids=list(source_ids),
                spans=[{"segment": segment, "start": len(prefix), "end": len(prefix) + len(source_ids)}],
                tokenizer_contract=contract,
                unsupported_controls=[] if contract["special_control_supported"] else ["cot_in_special", "answer_in_special"],
                think_token_count=len(source_ids) if segment == "think" else 0,
                answer_token_count=len(source_ids) if segment == "answer" else 0,
                think_text=_decode(tokenizer, source_ids) if segment == "think" else "",
                answer_text=_decode(tokenizer, source_ids) if segment == "answer" else "",
                source_finish_reason=record.get("finish_reason"),
                finish_reason="teacher_forced", generation_seed=None,
            )
            # Boundary is the close tag in answer prefixes, and after content in
            # think prefixes. It is metadata only for transplant measured spans.
            if segment == "think":
                transplant["boundary_idx"] = transplant["input_ids"].index(close_id, len(prefix) + len(source_ids))
            else:
                transplant["boundary_idx"] = max(i for i in range(assistant_start, len(prefix)) if prefix[i] == close_id)
            transplant["text"] = _decode(tokenizer, transplant["input_ids"][assistant_start:])
            transplant["generated_text"] = _decode(tokenizer, source_ids + suffix)
            transplant["record_id"] = transplant["transcript_id"] = _identity(transplant)
            results.append(transplant)
    return results
