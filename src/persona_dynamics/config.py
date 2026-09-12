"""Strict, serializable configuration; scientific choices freeze at run creation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import yaml


@dataclass
class Config:
    name: str = "qwen3-32b"
    backend: str = "vllm"
    model: str = "Qwen/Qwen3-32B"
    revision: str = "9216db5781bf21249d130ec9da846c4624c16137"
    layer: int = 32
    dtype: str = "bfloat16"
    device: str = "auto"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    domains: list[str] = field(default_factory=lambda: ["math", "code", "advice", "ai_philosophy"])
    output_dir: str = "runs/qwen3-32b"
    prompts_path: str = "data/prepared/prompts.jsonl"
    roles_path: str = "data/prepared/roles.jsonl"
    axis_path: str = "data/assets/vectors/qwen-3-32b/assistant_axis.pt"
    default_vector_path: str = "data/assets/vectors/qwen-3-32b/default_vector.pt"
    control_paths: list[str] = field(default_factory=lambda: ["data/assets/vectors/qwen-3-32b/role_vectors/skeptic.pt", "data/assets/vectors/qwen-3-32b/role_vectors/judge.pt"])
    max_tokens: int = 2048
    max_context: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    batch_size: int = 8
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    target_per_domain: int = 50
    candidate_limit_per_domain: int = 100
    min_per_domain: int = 40
    min_think_tokens: int = 200
    first_answer_tokens: int = 5
    exclude_first_answer_tokens: bool = True
    calibration_count: int = 24
    e0_question_count: int = 5
    e0_roles: list[str] = field(default_factory=lambda: ["consultant", "analyst", "judge", "skeptic", "teacher", "hermit", "pilgrim", "mystic", "oracle", "poet"])
    role_count: int = 60
    role_question_count: int = 5
    e0_min_default_advantage: float = 0.0
    e0_min_rank_correlation: float = 0.3
    bootstrap_samples: int = 2000
    analysis_seed: int = 2026
    control_seed: int = 42
    pca: bool = True
    experiments: list[str] = field(default_factory=lambda: ["E1", "E2", "E3", "E4"])
    e2_stepbystep: bool = True
    synthetic_dimension: int = 24

    def validate(self):
        if self.backend not in {"fixture", "hf", "vllm"}:
            raise ValueError("backend must be fixture, hf, or vllm")
        if len(self.seeds) != len(set(self.seeds)) or not self.seeds:
            raise ValueError("seeds must be unique and nonempty")
        allowed_domains = {"math", "code", "advice", "ai_philosophy"}
        if not self.domains or len(set(self.domains)) != len(self.domains) or not set(self.domains) <= allowed_domains:
            raise ValueError("domains must be a nonempty unique subset of math/code/advice/ai_philosophy")
        if self.min_per_domain > self.target_per_domain or self.min_per_domain < 1:
            raise ValueError("Require 1 <= min_per_domain <= target_per_domain")
        if self.candidate_limit_per_domain < self.target_per_domain:
            raise ValueError("Candidate limit must be at least target_per_domain")
        if self.calibration_count < 4:
            raise ValueError("At least four independent calibration prompts are required")
        if self.backend != "fixture":
            if len(self.revision) != 40:
                raise ValueError("Pin a full model commit SHA before running research experiments")
            if len(self.control_paths) < 2:
                raise ValueError("At least two structured role controls required")
            if "fixture" in self.model:
                raise ValueError("Fixture model requires fixture backend")
        if self.layer < 0 or self.temperature <= 0 or self.max_tokens < 1:
            raise ValueError("Invalid layer or generation settings")
        if self.max_context <= self.max_tokens:
            raise ValueError("max_context must leave room for prompt tokens")
        if "E4" in self.experiments and "E3" not in self.experiments:
            raise ValueError("E4 requires E3 generations")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=()) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError("Overrides use key=value")
        raw[key] = yaml.safe_load(value)
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**raw).validate()
