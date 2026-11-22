"""Separate adapter checkpoints and held-out bulk-to-set evaluation."""
from __future__ import annotations

import math
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn

from .bulk_distribution import profile_sliced_wasserstein_loss
from .bulk_embedding import (
    FrozenStackCellEncoder,
    embedding_sliced_wasserstein_loss,
)

Adapter = nn.Module
Baseline = Callable[[Mapping[str, object]], torch.Tensor]
DownstreamEvaluator = Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]]

__all__ = [
    "evaluate_bulk_adapter",
    "load_adapter_checkpoint",
    "preserve_stack_checkpoint",
    "save_adapter_checkpoint",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_stack_checkpoint(
    source: Union[str, Path],
    backup: Union[str, Path],
) -> str:
    """Create and verify an exact checkpoint copy without modifying its source."""
    source_path = Path(source)
    backup_path = Path(backup)
    if not source_path.is_file():
        raise FileNotFoundError(f"STACK checkpoint does not exist: {source_path}")
    if source_path.resolve() == backup_path.resolve():
        raise ValueError("STACK checkpoint backup must be a different path")
    source_hash = _sha256(source_path)
    if backup_path.exists():
        if not backup_path.is_file() or _sha256(backup_path) != source_hash:
            raise FileExistsError(
                f"Refusing to overwrite non-identical checkpoint backup: {backup_path}"
            )
        return source_hash

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
    )
    os.close(fd)
    try:
        shutil.copyfile(source_path, temporary_name)
        if _sha256(Path(temporary_name)) != source_hash:
            raise IOError("Copied STACK checkpoint does not match the source")
        os.replace(temporary_name, backup_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    if _sha256(source_path) != source_hash:
        raise IOError("STACK checkpoint changed while it was being preserved")
    return source_hash


def save_adapter_checkpoint(
    adapter: Adapter,
    path: Union[str, Path],
    *,
    config: Mapping[str, object],
    dataset: Mapping[str, object],
    random_seed: int,
    metadata: Optional[Mapping[str, object]] = None,
    stack_checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Atomically save adapter-only weights with run provenance.

    When ``stack_checkpoint_path`` is supplied, output is prohibited from
    targeting that file and its digest is verified unchanged after saving.
    """
    if not isinstance(config, Mapping) or not config:
        raise ValueError("config must be a nonempty mapping")
    if not isinstance(dataset, Mapping) or not dataset:
        raise ValueError("dataset must be a nonempty mapping")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    checkpoint_path = Path(path)
    stack_path = Path(stack_checkpoint_path) if stack_checkpoint_path is not None else None
    stack_hash = None
    if stack_path is not None:
        if not stack_path.is_file():
            raise FileNotFoundError(f"STACK checkpoint does not exist: {stack_path}")
        if stack_path.resolve() == checkpoint_path.resolve():
            raise ValueError("Adapter checkpoint path must not overwrite the STACK checkpoint")
        stack_hash = _sha256(stack_path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    run_metadata = dict(metadata or {})
    run_metadata.update(
        config=dict(config),
        dataset=dict(dataset),
        random_seed=random_seed,
    )
    payload = {
        "format": "stack-bulk-adapter",
        "version": 1,
        "adapter_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        },
        "metadata": run_metadata,
        "stack_checkpoint_sha256": stack_hash,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.", suffix=".tmp", dir=checkpoint_path.parent
    )
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, checkpoint_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    if stack_path is not None and _sha256(stack_path) != stack_hash:
        raise IOError("STACK checkpoint changed while saving the adapter checkpoint")


def load_adapter_checkpoint(
    adapter: Adapter,
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, object]:
    """Load only adapter state; the supplied STACK model is never touched."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # Older supported PyTorch releases do not accept weights_only.
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "stack-bulk-adapter":
        raise ValueError(f"{checkpoint_path} is not a STACK bulk-adapter checkpoint")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported adapter checkpoint version: {payload.get('version')!r}")
    state_dict = payload.get("adapter_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Adapter checkpoint is missing adapter_state_dict")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Adapter checkpoint metadata must be a mapping")
    adapter.load_state_dict(state_dict, strict=strict)
    return metadata


def _validate_eval_prediction(
    predicted: torch.Tensor,
    real: torch.Tensor,
) -> None:
    if not isinstance(predicted, torch.Tensor):
        raise TypeError("Adapter and baseline predictions must be tensors")
    if predicted.shape != real.shape:
        raise ValueError("predicted and real cell sets must have identical shapes")
    if not torch.is_floating_point(predicted):
        raise ValueError("predicted cell sets must be floating-point tensors")
    if not torch.isfinite(predicted).all() or torch.any(predicted < 0):
        raise ValueError("predicted cell sets must be finite and nonnegative")


def _batch_metrics(
    predicted: torch.Tensor,
    real: torch.Tensor,
    composition: torch.Tensor,
    scale_target: torch.Tensor,
    t_ref: torch.Tensor,
    encoder: FrozenStackCellEncoder,
    n_proj: int,
    downstream_evaluator: Optional[DownstreamEvaluator],
) -> Dict[str, float]:
    _validate_eval_prediction(predicted, real)
    if predicted.device != real.device or predicted.dtype != real.dtype:
        raise ValueError("predicted and real cell sets must share device and dtype")
    target_bulk = composition * scale_target.reshape(-1, 1)
    predicted_bulk = predicted.sum(dim=1)
    absolute_error = (predicted_bulk - target_bulk).abs()
    relative_error = absolute_error.sum(dim=-1) / target_bulk.abs().sum(dim=-1).clamp_min(1e-8)

    scale = t_ref.reshape(-1, 1, 1)
    normalized_real = real / scale
    profile_distance = profile_sliced_wasserstein_loss(
        predicted, normalized_real, n_proj=n_proj
    )
    predicted_embeddings = encoder(predicted)
    with torch.no_grad():
        real_embeddings = encoder(normalized_real)
    embedding_distance = embedding_sliced_wasserstein_loss(
        predicted_embeddings, real_embeddings, n_proj=n_proj
    )
    metrics = {
        "aggregate_mae": float(absolute_error.mean().item()),
        "aggregate_relative_l1": float(relative_error.mean().item()),
        "profile_sliced_wasserstein": float(profile_distance.item()),
        "embedding_sliced_wasserstein": float(embedding_distance.item()),
    }
    if downstream_evaluator is not None:
        downstream = downstream_evaluator(predicted, normalized_real)
        if not isinstance(downstream, Mapping):
            raise TypeError("downstream_evaluator must return a mapping of numeric metrics")
        for name, value in downstream.items():
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"Downstream metric {name!r} is not finite")
            metrics[f"downstream/{name}"] = score
    return metrics


def evaluate_bulk_adapter(
    adapter: Adapter,
    batches: Iterable[Mapping[str, object]],
    stack_model: nn.Module,
    *,
    stack_baseline: Optional[Baseline] = None,
    downstream_evaluator: Optional[DownstreamEvaluator] = None,
    device: Optional[Union[str, torch.device]] = None,
    n_proj: int = 64,
) -> Dict[str, object]:
    """Evaluate an adapter on held-out donor batches against explicit baselines.

    Each batch must contain Stage 1's ``composition``, ``scale_target``,
    ``cell_set``, ``t_ref``, and ``donor_id``. The adapter must implement
    ``forward_batch(batch)``. ``stack_baseline`` may provide an unmodified STACK
    prediction from the same batch; absent such a predictor, its report entry is
    ``None`` rather than a fabricated comparison. The bulk baseline allocates
    the target aggregate profile uniformly over N cells and has zero aggregate
    reconstruction error by construction.

    An optional downstream evaluator receives ``(predicted_cell_set, real_set)``
    and returns scalar metrics; these should correspond to a held-out downstream
    task and can operate on cell sets or their own embeddings.
    """
    if not callable(getattr(adapter, "forward_batch", None)):
        raise TypeError("adapter must implement forward_batch(batch)")
    if not callable(getattr(stack_model, "_reduce_and_tokenize", None)) or not callable(
        getattr(stack_model, "_run_attention_layers", None)
    ):
        raise TypeError("stack_model must expose STACK tensor encoder methods")
    if isinstance(n_proj, bool) or not isinstance(n_proj, int) or n_proj <= 0:
        raise ValueError("n_proj must be a positive integer")
    target_device = torch.device(device) if device is not None else next(
        adapter.parameters(), torch.empty(0)
    ).device
    adapter_parameters = list(adapter.parameters())
    adapter_original_device = adapter_parameters[0].device if adapter_parameters else target_device
    adapter_was_training = adapter.training
    adapter.to(target_device)
    adapter.eval()

    stack_parameters = list(stack_model.parameters())
    stack_requires_grad = [parameter.requires_grad for parameter in stack_parameters]
    stack_original_device = stack_parameters[0].device if stack_parameters else target_device
    stack_was_training = stack_model.training
    encoder = None
    results: Dict[str, list] = {"adapter": [], "bulk_baseline": [], "stack_baseline": []}
    weighted_counts: Dict[str, list] = {"adapter": [], "bulk_baseline": [], "stack_baseline": []}
    donors = set()
    n_samples = 0
    try:
        stack_model.to(target_device)
        encoder = FrozenStackCellEncoder(stack_model).to(target_device)
        with torch.no_grad():
            for batch in batches:
                required = {"composition", "scale_target", "cell_set", "t_ref", "donor_id"}
                missing = required.difference(batch)
                if missing:
                    raise KeyError(f"Evaluation batch missing fields: {', '.join(sorted(missing))}")
                tensor_batch = {
                    key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                real = tensor_batch["cell_set"]
                if not isinstance(real, torch.Tensor) or real.ndim != 3:
                    raise ValueError("cell_set must have shape [batch, cells, genes]")
                composition = tensor_batch["composition"]
                scale_target = tensor_batch["scale_target"]
                t_ref = torch.as_tensor(
                    tensor_batch["t_ref"], device=target_device, dtype=real.dtype
                )
                if t_ref.ndim == 0:
                    t_ref = t_ref.expand(real.shape[0])
                if t_ref.ndim == 2 and t_ref.shape == (real.shape[0], 1):
                    t_ref = t_ref.squeeze(-1)
                if t_ref.shape != (real.shape[0],):
                    raise ValueError("t_ref must be scalar or have shape [batch]")
                if not torch.isfinite(t_ref).all() or torch.any(t_ref <= 0):
                    raise ValueError("t_ref must be finite and positive")

                predicted = adapter.forward_batch(tensor_batch)
                count = real.shape[0]
                adapters = {
                    "adapter": predicted,
                    "bulk_baseline": (
                        composition * scale_target.reshape(-1, 1)
                    ).unsqueeze(1).expand(-1, real.shape[1], -1) / real.shape[1],
                }
                if stack_baseline is not None:
                    adapters["stack_baseline"] = stack_baseline(tensor_batch)
                for name, prediction in adapters.items():
                    metrics = _batch_metrics(
                        prediction,
                        real,
                        composition,
                        scale_target,
                        t_ref,
                        encoder,
                        n_proj,
                        downstream_evaluator,
                    )
                    results[name].append(metrics)
                    weighted_counts[name].append(count)
                donors_in_batch = tensor_batch["donor_id"]
                if isinstance(donors_in_batch, (list, tuple)):
                    donors.update(str(donor) for donor in donors_in_batch)
                else:
                    donors.add(str(donors_in_batch))
                n_samples += count
    finally:
        if encoder is not None:
            del encoder
        for parameter, requires_grad in zip(stack_parameters, stack_requires_grad):
            parameter.requires_grad_(requires_grad)
        stack_model.to(stack_original_device)
        stack_model.train(stack_was_training)
        adapter.to(adapter_original_device)
        adapter.train(adapter_was_training)

    if n_samples == 0:
        raise ValueError("No held-out evaluation samples were provided")

    def average(name: str) -> Optional[Dict[str, float]]:
        batches_for_metric = results[name]
        if not batches_for_metric:
            return None
        metric_names = batches_for_metric[0].keys()
        if any(record.keys() != metric_names for record in batches_for_metric[1:]):
            raise ValueError(f"Metric names changed between {name} batches")
        total_weight = sum(weighted_counts[name])
        return {
            metric: sum(
                record[metric] * weight
                for record, weight in zip(batches_for_metric, weighted_counts[name])
            ) / total_weight
            for metric in metric_names
        }

    return {
        "n_samples": n_samples,
        "heldout_donors": sorted(donors),
        "adapter": average("adapter"),
        "bulk_baseline": average("bulk_baseline"),
        "unmodified_stack_baseline": average("stack_baseline"),
        "stack_baseline_status": (
            "evaluated" if stack_baseline is not None else "not provided"
        ),
    }
