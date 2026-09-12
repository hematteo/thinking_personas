"""Small real-CUDA runtime check; produces no scientific results."""
import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from vllm import LLM, SamplingParams

assert torch.cuda.is_available()
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 8 and all("A40" in name for name in names), names
torch.manual_seed(0)
config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16)
model = Qwen3ForCausalLM(config).eval().to(device="cuda:0", dtype=torch.bfloat16)
with torch.inference_mode():
    output = model(torch.randint(0, 128, (1, 16), device="cuda:0"))
assert torch.isfinite(output.logits).all()
print(json.dumps(dict(passed=True, synthetic=True, gpus=names,
                      output_shape=list(output.logits.shape), torch=torch.__version__,
                      cuda=torch.version.cuda)))
