"""Joint Stage 2/4/5 training objective for bulk-to-set adapters."""
from __future__ import annotations

from typing import Dict, Mapping, Union

import torch
import torch.nn as nn

from .bulk_adapter import BulkToSetAdapter, bulk_reconstruction_loss
from .bulk_distribution import profile_sliced_wasserstein_loss
from .bulk_embedding import StackEmbeddingSetLoss


class BulkToSetTrainingObjective(nn.Module):
    """Connect Stage 1 batches, the Stage 2 adapter, and Stage 4/5 losses.

    The Stage 1 ``cell_set`` contains raw counts, while adapter predictions
    are in training-reference units. This objective divides real cells by the
    training-only ``t_ref`` before profile and frozen-STACK embedding matching.
    Stage 1's composition and scale target supervise bulk reconstruction.
    """

    def __init__(
        self,
        adapter: BulkToSetAdapter,
        stack_model: nn.Module,
        *,
        bulk_weight: float = 1.0,
        profile_weight: float = 1.0,
        embedding_weight: float = 1.0,
        n_proj: int = 64,
    ) -> None:
        super().__init__()
        for name, weight in (
            ("bulk_weight", bulk_weight),
            ("profile_weight", profile_weight),
            ("embedding_weight", embedding_weight),
        ):
            if not torch.isfinite(torch.tensor(weight)) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(n_proj, bool) or not isinstance(n_proj, int) or n_proj <= 0:
            raise ValueError("n_proj must be a positive integer")
        self.adapter = adapter
        self.stack_loss = StackEmbeddingSetLoss(stack_model, n_proj=n_proj)
        self.bulk_weight = float(bulk_weight)
        self.profile_weight = float(profile_weight)
        self.embedding_weight = float(embedding_weight)
        self.n_proj = n_proj

    def forward(self, batch: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        required = {"composition", "scale_target", "cell_set", "t_ref"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"batch is missing required fields: {', '.join(sorted(missing))}")
        composition = batch["composition"]
        scale_target = batch["scale_target"]
        real_cell_set = batch["cell_set"]
        if not isinstance(composition, torch.Tensor) or not isinstance(scale_target, torch.Tensor):
            raise TypeError("composition and scale_target must be tensors")
        if not isinstance(real_cell_set, torch.Tensor) or real_cell_set.ndim != 3:
            raise ValueError("cell_set must have shape [batch, cells, genes]")
        if composition.ndim != 2 or composition.shape != (
            real_cell_set.shape[0],
            real_cell_set.shape[2],
        ):
            raise ValueError("composition must match the cell_set batch and gene dimensions")
        if real_cell_set.shape[1] == 0:
            raise ValueError("cell_set must contain at least one cell")
        if not torch.is_floating_point(real_cell_set):
            raise ValueError("cell_set must be floating point")
        if not torch.isfinite(real_cell_set).all() or torch.any(real_cell_set < 0):
            raise ValueError("cell_set must contain finite, nonnegative raw counts")

        reference = torch.as_tensor(
            batch["t_ref"], device=real_cell_set.device, dtype=real_cell_set.dtype
        )
        if reference.ndim == 0:
            reference = reference.expand(real_cell_set.shape[0])
        elif reference.ndim == 2 and reference.shape == (real_cell_set.shape[0], 1):
            reference = reference.squeeze(-1)
        if reference.shape != (real_cell_set.shape[0],):
            raise ValueError("t_ref must be a scalar or have shape [batch] or [batch, 1]")
        if not torch.isfinite(reference).all() or torch.any(reference <= 0):
            raise ValueError("t_ref must contain finite, positive values")

        model_batch = {
            key: value
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        predicted = self.adapter.forward_batch(model_batch)
        if predicted.shape != real_cell_set.shape:
            raise ValueError("adapter output must have the same shape as cell_set")
        normalized_real = real_cell_set / reference[:, None, None]
        bulk_loss = bulk_reconstruction_loss(predicted, composition, scale_target)
        profile_loss = profile_sliced_wasserstein_loss(
            predicted, normalized_real, n_proj=self.n_proj
        )
        embedding_loss, predicted_embeddings, real_embeddings = self.stack_loss(
            predicted, normalized_real
        )
        total_loss = (
            self.bulk_weight * bulk_loss
            + self.profile_weight * profile_loss
            + self.embedding_weight * embedding_loss
        )
        return {
            "loss": total_loss,
            "bulk_reconstruction_loss": bulk_loss,
            "profile_distribution_loss": profile_loss,
            "stack_embedding_loss": embedding_loss,
            "predicted_cell_set": predicted,
            "normalized_real_cell_set": normalized_real,
            "predicted_embeddings": predicted_embeddings,
            "real_embeddings": real_embeddings,
        }


__all__ = ["BulkToSetTrainingObjective"]
