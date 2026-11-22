import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from stack.bulk_embedding import (
    FrozenStackCellEncoder,
    StackEmbeddingSetLoss,
    mean_pool_cell_embeddings,
)
from stack.bulk_adapter import BulkToSetAdapter


class TinyStackCellModel(nn.Module):
    def __init__(self, n_genes=5, hidden_dim=3):
        super().__init__()
        self.n_genes = n_genes
        self.projection = nn.Linear(n_genes, hidden_dim)

    def _reduce_and_tokenize(self, features):
        return self.projection(features)

    def _run_attention_layers(self, tokens):
        return tokens


def test_frozen_stack_encoder_returns_per_cell_embeddings_and_keeps_stack_frozen():
    stack_model = TinyStackCellModel()
    encoder = FrozenStackCellEncoder(stack_model)
    cells = torch.rand(2, 4, 5, requires_grad=True)

    embeddings = encoder(cells)
    embeddings.square().mean().backward()

    assert embeddings.shape == (2, 4, 3)
    assert cells.grad is not None and cells.grad.abs().sum() > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in stack_model.parameters())


def test_embedding_set_loss_backpropagates_only_to_generated_set_and_pools():
    stack_model = TinyStackCellModel()
    loss_module = StackEmbeddingSetLoss(stack_model, n_proj=16)
    adapter = nn.Sequential(nn.Linear(6, 20), nn.Softplus())
    bulk_input = torch.rand(2, 6)
    generated_cells = adapter(bulk_input).reshape(2, 4, 5)
    real_cells = torch.rand(2, 4, 5)

    loss, predicted_embeddings, real_embeddings = loss_module(generated_cells, real_cells)
    pooled = mean_pool_cell_embeddings(predicted_embeddings)
    loss.backward()

    assert predicted_embeddings.shape == (2, 4, 3)
    assert real_embeddings.shape == (2, 4, 3)
    assert pooled.shape == (2, 3)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in adapter.parameters()
    )
    assert all(parameter.grad is None for parameter in stack_model.parameters())
    assert not stack_model.training


def test_unmasked_stack_tensor_encoder_backpropagates_to_adapter_only():
    pytest.importorskip("scvi")
    from stack.models.core import StateICLModel

    torch.manual_seed(17)
    stack_model = StateICLModel(
        n_genes=5,
        n_hidden=4,
        token_dim=4,
        n_cells=3,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
    )
    adapter = BulkToSetAdapter(n_genes=5, hidden_dim=8, max_cells=3)
    cell_encoder = FrozenStackCellEncoder(stack_model)
    composition = torch.rand(2, 5)
    composition = composition / composition.sum(dim=-1, keepdim=True)
    generated_cells = adapter(composition, torch.tensor([1.0, 0.8]), n_cells=3)
    real_cells = torch.rand(2, 3, 5)

    generated_embeddings = cell_encoder(generated_cells)
    with torch.no_grad():
        real_embeddings = cell_encoder(real_cells)
    loss = (generated_embeddings - real_embeddings).square().mean()
    loss.backward()

    assert generated_embeddings.shape == (2, 3, stack_model.n_hidden * stack_model.token_dim)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in adapter.parameters()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in stack_model.parameters()
    )
    assert not stack_model.training


def test_mean_pool_validates_embedding_shape():
    with pytest.raises(ValueError, match="shape"):
        mean_pool_cell_embeddings(torch.ones(2, 3))
