"""Frozen STACK cell embeddings and embedding-set objectives."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FrozenStackCellEncoder(nn.Module):
    """Run the unmasked STACK cell encoder while freezing its parameters.

    The encoder applies the same ``log1p`` transform, gene reduction/tokenizer,
    and attention layers as STACK's regular forward path, but stops before its
    expression decoder. Autograd remains enabled for the input so embedding
    losses can train an upstream adapter.
    """

    def __init__(self, stack_model: nn.Module) -> None:
        super().__init__()
        if not callable(getattr(stack_model, "_reduce_and_tokenize", None)):
            raise TypeError("stack_model must expose _reduce_and_tokenize")
        if not callable(getattr(stack_model, "_run_attention_layers", None)):
            raise TypeError("stack_model must expose _run_attention_layers")
        self.stack_model = stack_model
        self.stack_model.requires_grad_(False)
        self.stack_model.eval()

    def train(self, mode: bool = True) -> "FrozenStackCellEncoder":
        super().train(mode)
        self.stack_model.eval()
        return self

    def forward(self, cell_set: torch.Tensor) -> torch.Tensor:
        """Return unmasked per-cell embeddings with shape ``[B, N, H]``."""
        if cell_set.ndim != 3:
            raise ValueError("cell_set must have shape [batch, cells, genes]")
        if any(size == 0 for size in cell_set.shape):
            raise ValueError("cell_set dimensions must be nonempty")
        if not torch.is_floating_point(cell_set):
            raise ValueError("cell_set must be a floating-point tensor")
        if not torch.isfinite(cell_set).all() or torch.any(cell_set < 0):
            raise ValueError("cell_set must contain finite, nonnegative expression values")
        n_genes = getattr(self.stack_model, "n_genes", cell_set.shape[-1])
        if cell_set.shape[-1] != n_genes:
            raise ValueError(f"cell_set has {cell_set.shape[-1]} genes; STACK expects {n_genes}")

        encode_unmasked = getattr(self.stack_model, "encode_unmasked", None)
        if callable(encode_unmasked):
            return encode_unmasked(cell_set)
        tokens = self.stack_model._reduce_and_tokenize(torch.log1p(cell_set))
        embeddings = self.stack_model._run_attention_layers(tokens)
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim < 3:
            raise ValueError("STACK attention layers must return per-cell tensor embeddings")
        if embeddings.shape[:2] != cell_set.shape[:2]:
            raise ValueError("STACK embedding batch and cell dimensions must match cell_set")
        return embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)


def mean_pool_cell_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean-pool per-cell embeddings ``[B, N, H]`` to ``[B, H]``."""
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, cells, hidden]")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0 or embeddings.shape[2] == 0:
        raise ValueError("embedding dimensions must be nonempty")
    if not torch.is_floating_point(embeddings) or not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite floating-point values")
    return embeddings.mean(dim=1)


def embedding_sliced_wasserstein_loss(
    predicted_embeddings: torch.Tensor,
    real_embeddings: torch.Tensor,
    n_proj: int = 64,
) -> torch.Tensor:
    """Sliced-Wasserstein distance between matched per-cell embedding sets."""
    if predicted_embeddings.ndim != 3 or real_embeddings.ndim != 3:
        raise ValueError("embedding sets must have shape [batch, cells, hidden]")
    if predicted_embeddings.shape != real_embeddings.shape:
        raise ValueError("predicted and real embedding sets must have identical shapes")
    if any(size == 0 for size in predicted_embeddings.shape):
        raise ValueError("embedding dimensions must be nonempty")
    if predicted_embeddings.device != real_embeddings.device:
        raise ValueError("embedding sets must be on the same device")
    if predicted_embeddings.dtype != real_embeddings.dtype:
        raise ValueError("embedding sets must have the same dtype")
    if not torch.is_floating_point(predicted_embeddings):
        raise ValueError("embedding sets must be floating-point tensors")
    if not torch.isfinite(predicted_embeddings).all() or not torch.isfinite(real_embeddings).all():
        raise ValueError("embedding sets must contain finite values")
    if isinstance(n_proj, bool) or not isinstance(n_proj, int) or n_proj <= 0:
        raise ValueError("n_proj must be a positive integer")

    hidden_dim = predicted_embeddings.shape[-1]
    projections = torch.randn(
        n_proj,
        hidden_dim,
        device=predicted_embeddings.device,
        dtype=predicted_embeddings.dtype,
    )
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(predicted_embeddings.dtype).tiny
    )
    predicted_projected = predicted_embeddings @ projections.transpose(0, 1)
    real_projected = real_embeddings @ projections.transpose(0, 1)
    predicted_sorted = predicted_projected.sort(dim=1).values
    real_sorted = real_projected.sort(dim=1).values
    return (predicted_sorted - real_sorted).square().mean()


class StackEmbeddingSetLoss(nn.Module):
    """Match generated and real cell sets in frozen STACK embedding space."""

    def __init__(self, stack_model: nn.Module, n_proj: int = 64) -> None:
        super().__init__()
        if isinstance(n_proj, bool) or not isinstance(n_proj, int) or n_proj <= 0:
            raise ValueError("n_proj must be a positive integer")
        self.encoder = FrozenStackCellEncoder(stack_model)
        self.n_proj = n_proj

    def forward(
        self,
        predicted_cell_set: torch.Tensor,
        real_cell_set: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embedding loss and both per-cell embedding sets.

        Returns ``(loss, predicted_embeddings, real_embeddings)``; each
        embedding set has shape ``[B, N, H]``.
        """
        if predicted_cell_set.shape != real_cell_set.shape:
            raise ValueError("predicted and real cell sets must have identical shapes")
        predicted_embeddings = self.encoder(predicted_cell_set)
        with torch.no_grad():
            real_embeddings = self.encoder(real_cell_set)
        loss = embedding_sliced_wasserstein_loss(
            predicted_embeddings, real_embeddings, n_proj=self.n_proj
        )
        return loss, predicted_embeddings, real_embeddings


__all__ = [
    "FrozenStackCellEncoder",
    "StackEmbeddingSetLoss",
    "embedding_sliced_wasserstein_loss",
    "mean_pool_cell_embeddings",
]
