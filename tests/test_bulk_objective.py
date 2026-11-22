import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from stack.bulk_adapter import BulkToSetAdapter
from stack.bulk_objective import BulkToSetTrainingObjective


class TinyStackEncoder(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.n_genes = n_genes
        self.projection = nn.Linear(n_genes, 4)

    def _reduce_and_tokenize(self, features):
        return self.projection(features)

    def _run_attention_layers(self, tokens):
        return tokens


def test_training_objective_connects_stage2_4_5_and_keeps_stack_frozen():
    torch.manual_seed(29)
    n_genes, n_cells = 5, 3
    adapter = BulkToSetAdapter(n_genes=n_genes, hidden_dim=8, max_cells=n_cells)
    stack_model = TinyStackEncoder(n_genes)
    objective = BulkToSetTrainingObjective(adapter, stack_model, n_proj=8)
    real_cells = torch.rand(2, n_cells, n_genes) * 10
    bulk = real_cells.sum(dim=1)
    total = bulk.sum(dim=-1, keepdim=True)
    batch = {
        "composition": bulk / total,
        "scale_target": torch.tensor([0.8, 1.2]),
        "cell_set": real_cells,
        "t_ref": torch.tensor([10.0, 12.0]),
    }

    result = objective(batch)
    result["loss"].backward()

    assert result["predicted_cell_set"].shape == (2, n_cells, n_genes)
    assert result["normalized_real_cell_set"].shape == (2, n_cells, n_genes)
    assert result["predicted_embeddings"].shape == (2, n_cells, 4)
    assert result["real_embeddings"].shape == (2, n_cells, 4)
    assert {
        "bulk_reconstruction_loss",
        "profile_distribution_loss",
        "stack_embedding_loss",
    }.issubset(result)
    assert all(torch.isfinite(result[name]) for name in (
        "loss",
        "bulk_reconstruction_loss",
        "profile_distribution_loss",
        "stack_embedding_loss",
    ))
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in adapter.parameters()
    )
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in stack_model.parameters())
