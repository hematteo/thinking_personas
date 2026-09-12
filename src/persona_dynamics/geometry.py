"""Mean-centered residual geometry with scalar-only forward hooks.

Layer numbers denote zero-based decoder blocks, measured after the MLP residual
addition and before the model's final normalization. No vocabulary logits or
per-token hidden states are saved. Calibration and E4 may retain segment means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .transcripts import annotate_transcript


def unit(vector: Any, *, label: str = "direction") -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite one-dimensional vector.")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero or numerically degenerate norm.")
    return vector / norm


def orthogonalize(vector: Any, against: Any) -> np.ndarray:
    """Remove the assistant component and normalize; reject degenerate controls."""
    axis = unit(against, label="assistant axis")
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != axis.shape:
        raise ValueError("Control and assistant dimensions differ.")
    return unit(vector - np.dot(vector, axis) * axis, label="orthogonalized control")


@dataclass
class DirectionBundle:
    center: np.ndarray
    directions: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        if self.center.ndim != 1 or not np.isfinite(self.center).all():
            raise ValueError("Center must be a finite one-dimensional vector.")
        if "assistant" not in self.directions:
            raise ValueError("Directions must include the assistant axis.")
        normalized = {}
        for name, vector in self.directions.items():
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid direction name {name!r}.")
            normalized[name] = unit(vector, label=name)
            if normalized[name].shape != self.center.shape:
                raise ValueError(f"Direction {name} and center dimensions differ.")
        self.directions = normalized

    @property
    def hidden_size(self) -> int:
        return len(self.center)

    def project(self, activations: Any) -> dict[str, np.ndarray]:
        """Project NumPy or torch activations; convert only scalars to CPU.

        Undefined cosine for exactly centered-zero vectors is NaN, never zero.
        Raw dot products intentionally use the uncentered residual vector.
        """
        if hasattr(activations, "detach"):
            import torch

            x = activations.detach().float()
            if x.shape[-1] != self.hidden_size:
                raise ValueError("Activation dimension does not match directions.")
            center = torch.as_tensor(self.center, device=x.device, dtype=x.dtype)
            matrix = torch.as_tensor(np.stack(list(self.directions.values()), axis=1), device=x.device, dtype=x.dtype)
            centered = x - center
            norm = torch.linalg.vector_norm(centered, dim=-1)
            denom = torch.where(norm > 1e-12, norm, torch.full_like(norm, float("nan")))
            cos = (centered @ matrix) / denom.unsqueeze(-1)
            dot = x @ matrix
            result = {"resid_norm": torch.linalg.vector_norm(x, dim=-1).cpu().numpy(), "centered_norm": norm.cpu().numpy()}
            for index, name in enumerate(self.directions):
                result[f"cos_{name}"] = cos[..., index].cpu().numpy()
                result[f"dot_{name}"] = dot[..., index].cpu().numpy()
            return result
        x = np.asarray(activations, dtype=np.float64)
        if x.shape[-1] != self.hidden_size:
            raise ValueError("Activation dimension does not match directions.")
        centered = x - self.center
        norm = np.linalg.norm(centered, axis=-1)
        denom = np.where(norm > 1e-12, norm, np.nan)
        result = {"resid_norm": np.linalg.norm(x, axis=-1), "centered_norm": norm}
        for name, direction in self.directions.items():
            result[f"cos_{name}"] = (centered @ direction) / denom
            result[f"dot_{name}"] = x @ direction
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata, direction_names=list(self.directions), hidden_size=self.hidden_size,
                        cosine="(h-center) dot unit_direction / norm(h-center)", raw_dot="h dot unit_direction")
        # One self-contained, pickle-free artifact avoids metadata/array mismatch.
        with path.open("wb") as handle:
            np.savez_compressed(handle, center=self.center,
                                directions=np.stack(list(self.directions.values())),
                                metadata=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "DirectionBundle":
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            names = metadata.pop("direction_names")
            values = artifact["directions"]
            if values.ndim != 2 or len(names) != len(values) or len(names) != len(set(names)):
                raise ValueError("Invalid direction artifact schema.")
            return cls(artifact["center"], dict(zip(names, values)), metadata)


def calibration_geometry(prompt_means: Any, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Equal-prompt center and reproducible random-half difference control.

    Inputs must be independent default no-think answer means, one per prompt.
    Equal prompt weighting prevents long calibration answers dominating the ruler.
    """
    means = np.asarray(prompt_means, dtype=np.float64)
    if means.ndim != 2 or len(means) < 4 or not np.isfinite(means).all():
        raise ValueError("Calibration needs at least four finite prompt mean vectors.")
    order = np.random.default_rng(seed).permutation(len(means))
    split = len(order) // 2
    first, second = order[:split], order[split:]
    center = means.mean(axis=0)
    null = unit(means[first].mean(axis=0) - means[second].mean(axis=0), label="calibration null")
    return center, null, {"center_weighting": "equal prompt", "calibration_count": len(means),
                          "null_seed": seed, "null_half_a": first.tolist(), "null_half_b": second.tolist()}


def _base_and_layers(model: Any) -> tuple[Any, Any]:
    # Qwen3ForCausalLM.model is Qwen3Model; forwarding it omits the LM head.
    base = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    if not hasattr(base, "layers"):
        raise ValueError("Expected a decoder base model exposing .layers (e.g. Qwen3Model).")
    return base, base.layers


def extract_record(
    model: Any,
    record: dict[str, Any],
    bundle: DirectionBundle,
    layer: int,
    keep_segment_means: bool = False,
    answer_skip: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One full teacher-forced forward; a hook retains only scalar projections.

    Optional segment means are uncentered vectors for calibration/E4, not token
    activations. The model must already be loaded on its intended device(s).
    """
    import torch

    if not record.get("valid", True):
        return [], {}
    if answer_skip < 0:
        raise ValueError("answer_skip must be nonnegative.")
    base, layers = _base_and_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"Layer {layer} outside decoder range [0, {len(layers) - 1}].")
    rows = annotate_transcript(record)
    if not rows:
        raise ValueError("No measured spans in valid transcript.")
    projection: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    calls = 0

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal calls
        calls += 1
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(record["input_ids"]):
            raise ValueError("Hook expected [1, full transcript length, hidden] residuals.")
        # Project measured tokens in bounded chunks. This avoids materializing
        # multiple full transcript × hidden float32 buffers on an 80 GB model.
        indices = [row["token_idx"] for row in rows]
        chunks: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(indices), 256):
            selected = hidden[0, indices[start : start + 256], :]
            for key, values in bundle.project(selected).items():
                chunks.setdefault(key, []).append(values)
        projection.update({key: np.concatenate(values) for key, values in chunks.items()})
        if keep_segment_means:
            for segment in ("think", "answer"):
                spans = [s for s in record["spans"] if s["segment"] == segment]
                total, count = None, 0
                remaining_skip = answer_skip if segment == "answer" else 0
                for span in spans:
                    skipped = min(remaining_skip, span["end"] - span["start"])
                    remaining_skip -= skipped
                    for start in range(span["start"] + skipped, span["end"], 256):
                        block = hidden[0, start : min(start + 256, span["end"]), :].float()
                        partial = block.sum(dim=0)
                        total = partial if total is None else total + partial
                        count += len(block)
                if count:
                    means[segment] = (total / count).cpu().numpy()

    if hasattr(base, "get_input_embeddings"):
        device = base.get_input_embeddings().weight.device
    else:
        device = next(base.parameters()).device
    ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    handle = layers[layer].register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                 output_hidden_states=False, output_attentions=False, return_dict=True)
    finally:
        handle.remove()
        model.train(was_training)
    if calls != 1:
        raise RuntimeError(f"Expected one decoder hook call, observed {calls}.")
    for index, row in enumerate(rows):
        row.update({key: float(values[index]) for key, values in projection.items()})
        row["layer"] = int(layer)
    return rows, means


def fit_persona_space(
    segment_records: Iterable[dict[str, Any]], n_components: int = 5
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit E4 on equal-role means; project both clouds in the answer PCA frame.

    Input rows: model, role, segment, vector, and optional prompt_id and seed.
    Repeated seeds are averaged within prompt before prompts within role. Only
    roles with paired think/answer vectors enter either PCA; default is displayed
    but does not define the role frame. One model per call prevents invalid frames.
    """
    rows = list(segment_records)
    if not rows:
        raise ValueError("No segment vectors supplied for PCA.")
    models = {row.get("model", "unknown") for row in rows}
    if len(models) != 1:
        raise ValueError("Persona PCA must be fitted independently for each model.")
    model = next(iter(models))
    by_prompt: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for index, row in enumerate(rows):
        if row["segment"] not in ("think", "answer"):
            continue
        vector = np.asarray(row["vector"], dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("PCA vectors must be finite one-dimensional means.")
        key = (str(row["role"]), row["segment"], str(row.get("prompt_id", index)))
        by_prompt.setdefault(key, []).append(vector)
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for (role, segment, _prompt), vectors in by_prompt.items():
        grouped.setdefault((role, segment), []).append(np.mean(vectors, axis=0))
    role_means = {key: np.mean(vectors, axis=0) for key, vectors in grouped.items()}
    paired = sorted(role for role, segment in role_means if segment == "answer" and role != "default" and (role, "think") in role_means)
    if len(paired) < 3:
        raise ValueError("Persona PCA needs at least three paired non-default roles.")
    answer = np.stack([role_means[role, "answer"] for role in paired])
    think = np.stack([role_means[role, "think"] for role in paired])
    count = min(int(n_components), len(paired) - 1, answer.shape[1])
    if count < 1:
        raise ValueError("n_components must be positive.")

    def pca(values: np.ndarray, allow_collapsed: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        center = values.mean(axis=0)
        _u, singular, vh = np.linalg.svd(values - center, full_matrices=False)
        components = vh[:count].copy()
        # Fix arbitrary signs for byte-stable plots and saved frames.
        for component in components:
            if component[np.argmax(np.abs(component))] < 0:
                component *= -1
        variance = singular**2
        rank = int(np.sum(singular > max(singular[0] * 1e-10, 1e-12)))
        if not allow_collapsed and rank < count:
            raise ValueError("Role cloud is rank deficient for requested PCA components.")
        explained = variance[:count] / variance.sum() if variance.sum() > 1e-24 else np.full(count, np.nan)
        # A collapsed thinking cloud is an H2 prediction, not a pipeline error.
        # Null-space singular vectors have no empirical meaning; mark them absent.
        if rank < count:
            components[rank:] = np.nan
            explained[rank:] = np.nan
        return center, components, explained, rank

    center, components, variance, answer_rank = pca(answer)
    think_center, think_components, think_variance, think_rank = pca(think, allow_collapsed=True)
    answer_spread = float(np.sqrt(np.mean(np.sum((answer - center) ** 2, axis=1))))
    think_spread = float(np.sqrt(np.mean(np.sum((think - think_center) ** 2, axis=1))))
    displacement = think - answer
    translation = displacement.mean(axis=0)
    residual = displacement - translation
    residual_spread = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    cloud_geometry = {
        "n_roles": len(paired),
        "space": "full hidden space; uncentered segment means; paired roles",
        "inference": "descriptive; no bootstrap; same roles used to fit PCA",
        "answer_rms_spread": answer_spread,
        "think_rms_spread": think_spread,
        "think_to_answer_spread_ratio": think_spread / answer_spread if answer_spread > 1e-12 else None,
        "translation_norm": float(np.linalg.norm(translation)),
        "mean_paired_displacement_norm": float(np.linalg.norm(displacement, axis=1).mean()),
        "rms_displacement_after_translation": residual_spread,
        "translation_residual_to_answer_spread_ratio": residual_spread / answer_spread if answer_spread > 1e-12 else None,
        "ratio_status": "available" if answer_spread > 1e-12 else "unavailable: degenerate answer spread",
    }
    coordinates = []
    for (role, segment), vector in sorted(role_means.items()):
        if role != "default" and role not in paired:
            continue
        scores = (vector - center) @ components.T
        coordinates.append(dict(model=model, role=role, segment=segment,
                                **{f"pc{i + 1}": float(value) for i, value in enumerate(scores)}))
    frame = {"center": center, "components": components, "explained_variance_ratio": variance,
             "think_center": think_center, "think_components": think_components,
             "think_explained_variance_ratio": think_variance}
    alignment = np.abs(components @ think_components.T)
    report = {"model": model, "coordinates": coordinates,
              "pc_alignment": [[float(value) if np.isfinite(value) else None for value in row] for row in alignment],
              "answer_explained_variance_ratio": variance.tolist(),
              "think_explained_variance_ratio": [float(value) if np.isfinite(value) else None for value in think_variance],
              "answer_rank": answer_rank, "think_rank": think_rank,
              "think_pca_status": "available" if think_rank >= count else "unavailable: collapsed cloud" if think_rank == 0 else "partially rank deficient",
              "cloud_geometry": cloud_geometry,
              "roles": paired, "n_components": count, "requested_components": int(n_components),
              "weighting": "seeds within prompt; prompts within role; equal roles",
              "frame_center": "answer-role mean for both coordinate clouds",
              "default_used_for_fit": False}
    return frame, report
