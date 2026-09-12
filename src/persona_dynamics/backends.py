"""Generation is deliberately separate from teacher-forced measurement."""
from __future__ import annotations

import gc
import re
import numpy as np

from .io import stable_seed


class FixtureTokenizer:
    """Reversible UTF-8 tokenizer for OFFLINE SYNTHETIC pipeline verification."""
    tags = {"<think>": 1, "</think>": 2, "<tool_response>": 3,
            "</tool_response>": 4, "<|im_start|>": 5, "<|im_end|>": 6}
    all_special_ids = [5, 6]
    eos_token_id = 6
    pad_token_id = 6
    chat_template = "<|im_start|>assistant <think> </think> <tool_response> </tool_response>"

    def encode(self, text, add_special_tokens=False):
        pieces = re.split("(" + "|".join(re.escape(x) for x in self.tags) + ")", text)
        ids = []
        for piece in pieces:
            ids.extend([self.tags[piece]] if piece in self.tags else [256 + b for b in piece.encode()])
        return ids

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        reverse = {v: k for k, v in self.tags.items()}
        parts, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i >= 256:
                buf.append(i - 256)
            else:
                if buf:
                    parts.append(buf.decode(errors="replace")); buf.clear()
                if not (skip_special_tokens and i in self.all_special_ids):
                    parts.append(reverse.get(i, ""))
        if buf:
            parts.append(buf.decode(errors="replace"))
        return "".join(parts)

    def get_added_vocab(self):
        return self.tags.copy()

    def convert_tokens_to_ids(self, token):
        return self.tags.get(token)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=True, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return self.encode(text) if tokenize else text


def tokenizer_for(config):
    if config.backend == "fixture":
        return FixtureTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.model, revision=config.revision,
                                         trust_remote_code=False)


def model_for(config):
    """Called only in extraction workers: vLLM has exited before this allocates."""
    import torch
    from transformers import AutoModelForCausalLM
    kwargs = {"revision": config.revision, "torch_dtype": getattr(torch, config.dtype),
              "trust_remote_code": False, "attn_implementation": "sdpa"}
    if config.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
    if config.device != "auto":
        model.to(config.device)
    model.eval()
    return model


def generate_batches(config, requests, tokenizer):
    """Yield (request, completion token IDs, finish reason), one sample per request."""
    if not requests:
        return
    if config.backend == "fixture":
        for r in requests:
            subject = r['prompt_id']
            think = (f"I should consider the user's question carefully. The request {subject} has several aspects. "
                     "The user wants a useful response, so I will compare possibilities and then choose a clear explanation. "
                     "First consider what is known. Next consider uncertainty. Finally I can give a practical answer.")
            answer = (f"Here is a considered response to {subject}. Start with the clearest facts, "
                      "acknowledge uncertainty, and choose one manageable next step. "
                      "A thoughtful explanation can help you decide what to do.")
            if r["role"] != "default":
                answer = f"I speak as {r['role']}. " + answer
            if r["enable_thinking"]:
                text = "<think>" + think + "</think>\n\n" + answer + "<|im_end|>"
            elif r["condition"] == "nothink_stepbystep":
                text = think + " " + answer + "<|im_end|>"
            else:
                text = answer + "<|im_end|>"
            yield r, tokenizer.encode(text), "stop"
        return
    if config.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=config.model, revision=config.revision, tokenizer_revision=config.revision,
                  dtype=config.dtype, tensor_parallel_size=config.tensor_parallel_size,
                  gpu_memory_utilization=config.gpu_memory_utilization,
                  enforce_eager=config.enforce_eager,
                  max_model_len=config.max_context, trust_remote_code=False, seed=config.seeds[0])
        for start in range(0, len(requests), config.batch_size):
            batch = requests[start:start + config.batch_size]
            params = [SamplingParams(temperature=config.temperature, top_p=config.top_p,
                       top_k=config.top_k, max_tokens=config.max_tokens,
                       seed=r["generation_seed"], skip_special_tokens=False) for r in batch]
            results = llm.generate([{"prompt_token_ids": r["prefix_ids"]} for r in batch], params)
            for r, result in zip(batch, results, strict=True):
                out = result.outputs[0]
                yield r, list(out.token_ids), out.finish_reason
        return
    import torch
    from transformers import set_seed
    model = model_for(config)
    for r in requests:
        set_seed(r["generation_seed"])
        ids = torch.tensor([r["prefix_ids"]], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 do_sample=True, temperature=config.temperature, top_p=config.top_p,
                                 top_k=config.top_k, max_new_tokens=config.max_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
        completion = out[0, ids.shape[1]:].tolist()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "stop" if completion and completion[-1] in eos else "length"
        yield r, completion, finish
    del model
    gc.collect()


def fixture_activations(record, dimension):
    """Designed latent signals, never model evidence. Deterministic by prompt/seed."""
    from .transcripts import annotate_transcript
    n = len(record["input_ids"])
    rng = np.random.default_rng(stable_seed(record["seed"], record["prompt_id"], record["role"]))
    hidden = rng.normal(0, .35, (n, dimension)).astype(np.float32)
    hidden[:, -1] += 10
    score = float(record.get("role_score", 1.0))
    is_default = record.get("role", "default") == "default"
    role_shift = 0 if is_default else 1.0 + 3.0 / (1.0 + max(score, 0))
    for row in annotate_transcript(record):
        i = row["token_idx"]
        is_think = row["segment"] == "think"
        hidden[i, 0] += (0.6 if is_think else 2.0) - role_shift * (.18 if is_think else 1.0)
        # Enough rank for a small persona PCA; no hidden tensor is saved to disk.
        if not is_default:
            for d in range(3, min(dimension - 1, 12)):
                hidden[i, d] += np.sin((d + 1) * (score + .2)) * (.25 if is_think else .6)
    return hidden
