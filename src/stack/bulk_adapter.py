"""Bulk-to-cell-set adapter and bulk reconstruction objective."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BulkToSetAdapter(nn.Module):
    """Generate a nonnegative cell set conditioned on a bulk profile.

    Stage 1 batches provide ``composition`` shaped ``[B, G]``,
    ``scale_target`` shaped ``[B]`` or ``[B, 1]``, and paired ``cell_set``
    shaped ``[B, N, G]``. The real cell values are not model inputs:
    :meth:`forward_batch` uses them only to obtain the output cell count ``N``.

    Generated values are in reference-scale units. Summing over cells is
    supervised against ``composition * scale_target``. ``top_k_genes`` applies
    a hard per-cell support limit, making the generated set sparse.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 128,
        max_cells: int = 128,
        top_k_genes: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_genes <= 0 or hidden_dim <= 0 or max_cells <= 0:
            raise ValueError("n_genes, hidden_dim, and max_cells must be positive")
        if top_k_genes is not None and not 0 < top_k_genes <= n_genes:
            raise ValueError("top_k_genes must be in [1, n_genes]")

        self.n_genes = n_genes
        self.max_cells = max_cells
        self.top_k_genes = top_k_genes or n_genes
        self.bulk_encoder = nn.Sequential(
            nn.Linear(n_genes + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cell_slots = nn.Parameter(torch.randn(max_cells, hidden_dim) * 0.02)
        self.expression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(
        self,
        composition: torch.Tensor,
        scale_target: torch.Tensor,
        n_cells: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a generated cell set with shape ``[B, N, G]``."""
        if composition.ndim != 2 or composition.shape[1] != self.n_genes:
            raise ValueError(f"composition must have shape [batch, {self.n_genes}]")
        if scale_target.ndim == 1:
            scale_target = scale_target.unsqueeze(-1)
        if scale_target.ndim != 2 or scale_target.shape != (composition.shape[0], 1):
            raise ValueError("scale_target must have shape [batch] or [batch, 1]")
        if n_cells is None:
            n_cells = self.max_cells
        if (
            isinstance(n_cells, bool)
            or not isinstance(n_cells, int)
            or not 0 < n_cells <= self.max_cells
        ):
            raise ValueError(f"n_cells must be an integer in [1, {self.max_cells}]")
        if composition.device != scale_target.device:
            raise ValueError("composition and scale_target must be on the same device")
        if composition.dtype != scale_target.dtype or not torch.is_floating_point(composition):
            raise ValueError("composition and scale_target must have the same floating-point dtype")
        if not torch.isfinite(composition).all() or not torch.isfinite(scale_target).all():
            raise ValueError("composition and scale_target must contain finite values")
        if torch.any(composition < 0) or torch.any(scale_target < 0):
            raise ValueError("composition and scale_target must be nonnegative")
        if torch.any(composition.sum(dim=-1) <= 0):
            raise ValueError("each composition must have positive mass")

        context = self.bulk_encoder(
            torch.cat((composition, torch.log1p(scale_target)), dim=-1)
        )
        cell_context = context.unsqueeze(1) + self.cell_slots[:n_cells].unsqueeze(0)
        gene_logits = self.expression_head(cell_context)

        if self.top_k_genes < self.n_genes:
            top_indices = gene_logits.topk(self.top_k_genes, dim=-1).indices
            active = torch.zeros_like(gene_logits, dtype=torch.bool).scatter_(
                dim=-1, index=top_indices, value=True
            )
            gene_logits = gene_logits.masked_fill(~active, float("-inf"))
        rates = F.softplus(gene_logits)
        return rates * composition.unsqueeze(1) * scale_target.unsqueeze(1)

    def forward_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Generate a set from the Stage 1 batch contract."""
        required = {"composition", "scale_target", "cell_set"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"batch is missing required fields: {', '.join(sorted(missing))}")
        composition = batch["composition"]
        cell_set = batch["cell_set"]
        if composition.ndim != 2:
            raise ValueError("batch composition must have shape [batch, genes]")
        if cell_set.ndim != 3 or cell_set.shape[2] != self.n_genes:
            raise ValueError(f"cell_set must have shape [batch, cells, {self.n_genes}]")
        if cell_set.shape[0] != composition.shape[0]:
            raise ValueError("cell_set and composition batch dimensions must match")
        return self(composition, batch["scale_target"], n_cells=cell_set.shape[1])


def bulk_reconstruction_loss(
    predicted_cell_set: torch.Tensor,
    composition: torch.Tensor,
    scale_target: torch.Tensor,
) -> torch.Tensor:
    """MSE between generated cell sums and reference-scaled bulk targets."""
    if predicted_cell_set.ndim != 3:
        raise ValueError("predicted_cell_set must have shape [batch, cells, genes]")
    if composition.ndim != 2 or composition.shape != (
        predicted_cell_set.shape[0],
        predicted_cell_set.shape[2],
    ):
        raise ValueError("composition shape must match the predicted batch and gene dimensions")
    if scale_target.ndim == 1:
        scale_target = scale_target.unsqueeze(-1)
    if scale_target.shape != (predicted_cell_set.shape[0], 1):
        raise ValueError("scale_target must have shape [batch] or [batch, 1]")
    if composition.device != predicted_cell_set.device or scale_target.device != predicted_cell_set.device:
        raise ValueError("predictions and targets must be on the same device")
    if not (
        torch.is_floating_point(predicted_cell_set)
        and torch.is_floating_point(composition)
        and torch.is_floating_point(scale_target)
    ):
        raise ValueError("predictions and targets must be floating-point tensors")
    if predicted_cell_set.dtype != composition.dtype or predicted_cell_set.dtype != scale_target.dtype:
        raise ValueError("predictions and targets must have matching dtypes")
    if not torch.isfinite(predicted_cell_set).all() or not torch.isfinite(composition).all():
        raise ValueError("predictions and composition must contain finite values")
    if not torch.isfinite(scale_target).all():
        raise ValueError("scale_target must contain finite values")
    if torch.any(predicted_cell_set < 0) or torch.any(composition < 0) or torch.any(scale_target < 0):
        raise ValueError("predictions and targets must be nonnegative")
    if torch.any(composition.sum(dim=-1) <= 0):
        raise ValueError("each composition must have positive mass")

    target_bulk = composition * scale_target
    predicted_bulk = predicted_cell_set.sum(dim=1)
    return F.mse_loss(predicted_bulk, target_bulk)


__all__ = ["BulkToSetAdapter", "bulk_reconstruction_loss"]
