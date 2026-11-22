"""Donor-isolated pseudo-bulk datasets for bulk-to-set adapter training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import anndata
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from .gene_processing import safe_decode_array
from .training.datasets import load_gene_list

log = logging.getLogger(__name__)

AnnDataSource = Union[anndata.AnnData, str, Path]
GeneListSource = Union[Sequence[str], str, Path]

__all__ = [
    "BulkPseudoBulkDataset",
    "create_bulk_train_val_test_datasets",
    "prepare_bulk_inference_batch",
]


def prepare_bulk_inference_batch(
    bulk_counts: torch.Tensor,
    t_ref: Union[float, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Convert aligned raw bulk counts to the Stage 2 inference contract.

    The returned ``composition`` and ``scale_target`` use the same equations
    as Stage 1 pseudo-bulk samples: ``q = b / (sum(b) + eps)`` and
    ``scale_target = sum(b) / T_ref``. Inputs must already follow the same
    ordered gene vocabulary used to train the adapter.
    """
    if bulk_counts.ndim != 2 or bulk_counts.shape[0] == 0 or bulk_counts.shape[1] == 0:
        raise ValueError("bulk_counts must have nonempty shape [batch, genes]")
    if not torch.is_floating_point(bulk_counts):
        raise ValueError("bulk_counts must be a floating-point tensor")
    if not torch.isfinite(bulk_counts).all() or torch.any(bulk_counts < 0):
        raise ValueError("bulk_counts must contain finite, nonnegative counts")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    reference = torch.as_tensor(t_ref, device=bulk_counts.device, dtype=bulk_counts.dtype)
    if reference.ndim == 0:
        reference = reference.expand(bulk_counts.shape[0])
    elif reference.ndim == 2 and reference.shape == (bulk_counts.shape[0], 1):
        reference = reference.squeeze(-1)
    if reference.shape != (bulk_counts.shape[0],):
        raise ValueError("t_ref must be a scalar or have shape [batch] or [batch, 1]")
    if not torch.isfinite(reference).all() or torch.any(reference <= 0):
        raise ValueError("t_ref must contain finite, positive values")

    totals = bulk_counts.sum(dim=-1)
    if torch.any(totals <= 0):
        raise ValueError("bulk samples must have positive library sizes")
    return {
        "composition": bulk_counts / (totals.unsqueeze(-1) + eps),
        "scale_target": totals / reference,
    }


def _validate_gene_list(target_genes: GeneListSource) -> List[str]:
    if isinstance(target_genes, (str, Path)):
        target_genes = load_gene_list(str(target_genes))
    genes = [str(gene).upper() for gene in target_genes]
    if not genes:
        raise ValueError("target_genes must contain at least one gene")
    if any(not gene for gene in genes):
        raise ValueError("target_genes cannot contain empty gene names")
    if len(set(genes)) != len(genes):
        raise ValueError("target_genes contains duplicate names")
    return genes


def _load_and_align(
    source: AnnDataSource,
    target_genes: Sequence[str],
    donor_col: str,
    counts_layer: Optional[str],
    gene_name_col: Optional[str],
) -> Dict[str, Union[np.ndarray, sp.csr_matrix]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"AnnData file does not exist: {path}")
        adata = anndata.read_h5ad(path)
        source_name = str(path)
    elif isinstance(source, anndata.AnnData):
        adata = source
        source_name = "in-memory AnnData"
    else:
        raise TypeError(f"Expected an AnnData object or file path, got {type(source).__name__}")

    if donor_col not in adata.obs.columns:
        raise ValueError(f"{source_name} is missing donor column {donor_col!r}")

    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"{source_name} is missing counts layer {counts_layer!r}")
        matrix = adata.layers[counts_layer]
        gene_metadata = adata.var
        gene_names = adata.var_names
    elif adata.raw is not None:
        matrix = adata.raw.X
        gene_metadata = adata.raw.var
        gene_names = adata.raw.var_names
    else:
        matrix = adata.X
        gene_metadata = adata.var
        gene_names = adata.var_names

    if gene_name_col is not None:
        if gene_name_col not in gene_metadata.columns:
            raise ValueError(f"{source_name} is missing gene-name column {gene_name_col!r}")
        source_gene_names = safe_decode_array(gene_metadata[gene_name_col].to_numpy())
    else:
        source_gene_names = safe_decode_array(gene_names)
    source_gene_names = np.char.upper(source_gene_names)

    if matrix.shape[1] != len(source_gene_names):
        raise ValueError(f"{source_name} has inconsistent count and gene-name dimensions")
    if len(set(source_gene_names.tolist())) != len(source_gene_names):
        raise ValueError(f"{source_name} contains duplicate gene names; alignment is ambiguous")

    target_index = {gene: idx for idx, gene in enumerate(target_genes)}
    matched_source_indices = []
    matched_target_indices = []
    for source_idx, gene in enumerate(source_gene_names):
        target_idx = target_index.get(str(gene))
        if target_idx is not None:
            matched_source_indices.append(source_idx)
            matched_target_indices.append(target_idx)
    if not matched_source_indices:
        raise ValueError(f"{source_name} has no genes in common with the target gene list")

    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{source_name} contains non-numeric expression values")
    if not np.isfinite(values).all():
        raise ValueError(f"{source_name} contains non-finite expression values")
    if np.any(values < 0):
        raise ValueError(f"{source_name} contains negative values; raw counts must be nonnegative")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{source_name} contains non-integer values; raw counts are required")
    if np.any(values >= float(2**63)):
        raise ValueError(f"{source_name} contains counts too large for int64")

    if sp.issparse(matrix):
        alignment = sp.csr_matrix(
            (
                np.ones(len(matched_source_indices), dtype=np.int8),
                (matched_source_indices, matched_target_indices),
            ),
            shape=(matrix.shape[1], len(target_genes)),
        )
        aligned = (matrix.tocsr().astype(np.int64) @ alignment).tocsr()
    else:
        aligned = np.zeros((matrix.shape[0], len(target_genes)), dtype=np.int64)
        aligned[:, matched_target_indices] = np.asarray(matrix)[:, matched_source_indices].astype(np.int64)

    donor_values = adata.obs[donor_col]
    if donor_values.isna().any():
        raise ValueError(f"{source_name} contains cells with missing donor IDs in {donor_col!r}")
    donor_ids = donor_values.astype(str).str.strip().to_numpy()
    if np.any(donor_ids == ""):
        raise ValueError(f"{source_name} contains empty donor IDs in {donor_col!r}")

    donor_matrices = {
        str(donor_id): aligned[donor_ids == donor_id]
        for donor_id in np.unique(donor_ids)
    }
    log.info(
        "Aligned %s cells from %s to %s target genes across %s donors",
        matrix.shape[0],
        source_name,
        len(target_genes),
        len(donor_matrices),
    )
    return donor_matrices


def _combine_donor_matrices(
    sources: Sequence[AnnDataSource],
    target_genes: Sequence[str],
    donor_col: str,
    counts_layer: Optional[str],
    gene_name_col: Optional[str],
) -> Dict[str, Union[np.ndarray, sp.csr_matrix]]:
    if not sources:
        raise ValueError("At least one AnnData source is required")
    grouped: Dict[str, List[Union[np.ndarray, sp.csr_matrix]]] = {}
    for source in sources:
        for donor_id, matrix in _load_and_align(
            source, target_genes, donor_col, counts_layer, gene_name_col
        ).items():
            grouped.setdefault(donor_id, []).append(matrix)
    return {
        donor_id: (
            sp.vstack([sp.csr_matrix(matrix) for matrix in matrices], format="csr")
            if any(sp.issparse(matrix) for matrix in matrices)
            else np.concatenate(matrices, axis=0)
        )
        for donor_id, matrices in grouped.items()
    }


class BulkPseudoBulkDataset(Dataset):
    """Fixed pseudo-bulk samples drawn from an explicit set of donors."""

    def __init__(
        self,
        donor_matrices: Dict[str, Union[np.ndarray, sp.csr_matrix]],
        donor_ids: Sequence[str],
        sample_size: int,
        samples_per_donor: int,
        random_state: int,
        eps: float = 1e-8,
        t_ref: Optional[float] = None,
    ):
        if (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, (int, np.integer))
            or sample_size <= 0
        ):
            raise ValueError("sample_size must be a positive integer")
        if (
            isinstance(samples_per_donor, bool)
            or not isinstance(samples_per_donor, (int, np.integer))
            or samples_per_donor <= 0
        ):
            raise ValueError("samples_per_donor must be a positive integer")
        if eps <= 0 or not np.isfinite(eps):
            raise ValueError("eps must be finite and positive")
        self.donor_ids = tuple(sorted(donor_ids))
        if not self.donor_ids:
            raise ValueError("A split must contain at least one donor")
        self.sample_size = sample_size
        self.samples_per_donor = samples_per_donor
        self.eps = float(eps)
        self._donor_matrices = donor_matrices
        self.samples: List[Tuple[str, np.ndarray]] = []
        rng = np.random.default_rng(random_state)
        for donor_id in self.donor_ids:
            matrix = donor_matrices[donor_id]
            if matrix.shape[0] < sample_size:
                raise ValueError(
                    f"Donor {donor_id!r} has {matrix.shape[0]} cells, fewer than "
                    f"sample_size={sample_size}"
                )
            for _ in range(samples_per_donor):
                indices = rng.choice(matrix.shape[0], size=sample_size, replace=False)
                self.samples.append((donor_id, indices))
        totals = np.asarray([self._bulk_total(donor_id, indices) for donor_id, indices in self.samples])
        if t_ref is None:
            if np.any(totals <= 0):
                raise ValueError("Training pseudo-bulks must have positive total counts to estimate T_ref")
            self.t_ref = float(np.median(totals))
        else:
            if t_ref <= 0 or not np.isfinite(t_ref):
                raise ValueError("t_ref must be finite and positive")
            self.t_ref = float(t_ref)

    def _bulk_total(self, donor_id: str, indices: np.ndarray) -> int:
        return int(np.asarray(self._donor_matrices[donor_id][indices].sum(axis=0)).sum())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        donor_id, indices = self.samples[index]
        sampled = self._donor_matrices[donor_id][indices]
        cell_set = sampled.toarray() if sp.issparse(sampled) else np.asarray(sampled)
        bulk_counts = np.asarray(sampled.sum(axis=0)).reshape(-1).astype(np.float32)
        cell_set = cell_set.astype(np.float32, copy=False)
        total = float(bulk_counts.sum(dtype=np.float64))
        composition = bulk_counts / (total + self.eps)
        return {
            "bulk_counts": torch.from_numpy(bulk_counts),
            "composition": torch.from_numpy(composition.astype(np.float32, copy=False)),
            "scale_target": torch.tensor(total / self.t_ref, dtype=torch.float32),
            "cell_set": torch.from_numpy(cell_set),
            "donor_id": donor_id,
            "cell_indices": torch.from_numpy(indices.copy()),
            "t_ref": self.t_ref,
        }


def create_bulk_train_val_test_datasets(
    sources: Sequence[AnnDataSource],
    target_genes: GeneListSource,
    donor_col: str,
    sample_size: int,
    samples_per_donor: int = 8,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    random_state: int = 42,
    counts_layer: Optional[str] = None,
    gene_name_col: Optional[str] = None,
    eps: float = 1e-8,
) -> Tuple[BulkPseudoBulkDataset, Optional[BulkPseudoBulkDataset], Optional[BulkPseudoBulkDataset]]:
    """Split donors before sampling; return train, validation, and test datasets.

    ``T_ref`` is the median total count among generated training pseudo-bulks.
    Each sample's ``scale_target`` is its total count divided by this reference.
    Counts use ``counts_layer`` when supplied, otherwise ``.raw.X`` when available,
    then ``.X``. Count integrality and nonnegativity are checked explicitly.
    """
    genes = _validate_gene_list(target_genes)
    if not 0 <= test_ratio < 1 or not 0 <= val_ratio < 1 or test_ratio + val_ratio >= 1:
        raise ValueError("test_ratio and val_ratio must be nonnegative and sum to less than one")
    donor_matrices = _combine_donor_matrices(
        sources, genes, donor_col, counts_layer, gene_name_col
    )
    donor_ids = sorted(donor_matrices)
    if len(donor_ids) < 2:
        raise ValueError("At least two donors are required for a donor-level split")
    n_test = int(np.ceil(len(donor_ids) * test_ratio)) if test_ratio else 0
    n_val = int(np.ceil(len(donor_ids) * val_ratio)) if val_ratio else 0
    if n_test + n_val >= len(donor_ids):
        raise ValueError(
            f"Cannot reserve {n_test} test and {n_val} validation donors from "
            f"{len(donor_ids)} total donors while keeping training donors"
        )

    shuffled = np.random.default_rng(random_state).permutation(donor_ids).tolist()
    test_donors = shuffled[:n_test]
    val_donors = shuffled[n_test : n_test + n_val]
    train_donors = shuffled[n_test + n_val :]
    log.info(
        "Split %s donors before sampling: %s train, %s validation, %s test",
        len(donor_ids),
        len(train_donors),
        len(val_donors),
        len(test_donors),
    )
    train = BulkPseudoBulkDataset(
        donor_matrices, train_donors, sample_size, samples_per_donor, random_state, eps
    )
    val = (
        BulkPseudoBulkDataset(
            donor_matrices, val_donors, sample_size, samples_per_donor,
            random_state + 1, eps, t_ref=train.t_ref,
        )
        if val_donors else None
    )
    test = (
        BulkPseudoBulkDataset(
            donor_matrices, test_donors, sample_size, samples_per_donor,
            random_state + 2, eps, t_ref=train.t_ref,
        )
        if test_donors else None
    )
    return train, val, test
