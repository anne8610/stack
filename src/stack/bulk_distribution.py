"""Profile-level distribution losses for bulk-to-set training."""
from __future__ import annotations

import math

import torch


def profile_sliced_wasserstein_loss(
    predicted_cell_set: torch.Tensor,
    real_cell_set: torch.Tensor,
    n_proj: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match generated and real cell-profile distributions with sliced W2.

    Both inputs must have shape ``[B, N, G]`` with equal dimensions. Profiles
    are normalized to unit library size per cell before projection, separating
    profile-shape matching from the bulk reconstruction loss, which supervises
    aggregate abundance. Zero-library cells remain zero profiles.
    """
    if predicted_cell_set.ndim != 3 or real_cell_set.ndim != 3:
        raise ValueError("cell sets must have shape [batch, cells, genes]")
    if predicted_cell_set.shape != real_cell_set.shape:
        raise ValueError("predicted and real cell sets must have identical shapes")
    if any(size == 0 for size in predicted_cell_set.shape):
        raise ValueError("cell sets must have nonempty batch, cell, and gene dimensions")
    if predicted_cell_set.device != real_cell_set.device:
        raise ValueError("predicted and real cell sets must be on the same device")
    if predicted_cell_set.dtype != real_cell_set.dtype:
        raise ValueError("predicted and real cell sets must have the same dtype")
    if not torch.is_floating_point(predicted_cell_set):
        raise ValueError("cell sets must be floating-point tensors")
    if not torch.isfinite(predicted_cell_set).all() or not torch.isfinite(real_cell_set).all():
        raise ValueError("cell sets must contain finite values")
    if torch.any(predicted_cell_set < 0) or torch.any(real_cell_set < 0):
        raise ValueError("cell sets must contain nonnegative expression values")
    if isinstance(n_proj, bool) or not isinstance(n_proj, int) or n_proj <= 0:
        raise ValueError("n_proj must be a positive integer")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    predicted_profiles = predicted_cell_set / predicted_cell_set.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    real_profiles = real_cell_set / real_cell_set.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    n_genes = predicted_cell_set.shape[-1]
    projections = torch.randn(
        n_proj,
        n_genes,
        device=predicted_cell_set.device,
        dtype=predicted_cell_set.dtype,
    )
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(eps)

    predicted_projected = predicted_profiles @ projections.transpose(0, 1)
    real_projected = real_profiles @ projections.transpose(0, 1)
    predicted_sorted = predicted_projected.sort(dim=1).values
    real_sorted = real_projected.sort(dim=1).values
    return (predicted_sorted - real_sorted).square().mean()


__all__ = ["profile_sliced_wasserstein_loss"]
